import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import sklearn.metrics
from tqdm import tqdm
import os

import config
from utils import sort_for_packed_sequence
from models.gensen import GenSenSingle


class SentimentAnalysisBaseline(nn.Module):
    gensen_out_size = 2048  # output from gensen module

    def __init__(self,
                 train_vocab,
                 params,
                 train_gensen=False
                 ):
        super(SentimentAnalysisBaseline, self).__init__()
        self.params = params
        self.cuda_available = torch.cuda.is_available()
        self.train_gensen = train_gensen

        self.gensen = GenSenSingle(
            model_folder=os.path.join(config.MODELS_PATH, 'GenSen'),
            filename_prefix='nli_large',
            pretrained_emb=os.path.join(config.MODELS_PATH, 'embeddings/glove.840B.300d.h5'),
            cuda=self.cuda_available
        )
        self.gensen.vocab_expansion(list(train_vocab))
        self.word2id = self.gensen.task_word2id
        # freeze gensen's weights
        if not self.train_gensen:
            for param in self.gensen.parameters():
                param.requires_grad = False

        if self.params['sentence_encoder_num_layers'] > 0:
            self.sentence_encoder = nn.GRU(
                input_size=2048,
                hidden_size=params['hidden_size'],
                num_layers=params['sentence_encoder_num_layers'],
                batch_first=True,
                bidirectional=True
            )

        # Outputs
        sentence_rep_size = 1 + SentimentAnalysisBaseline.gensen_out_size \
            if self.params['sentence_encoder_num_layers'] == 0 else 1 + 2 * params['hidden_size']
        self.Iseen = nn.Linear(sentence_rep_size, 3)  # 3 classes : not seen, seen, did not mention
        self.Iliked = nn.Linear(sentence_rep_size, 3)  # 3 classes : disliked, liked, did not mention
        self.Rseen = nn.Linear(sentence_rep_size, 3)  # 3 classes : not seen, seen, did not mention
        self.Rliked = nn.Linear(sentence_rep_size, 3)  # 3 classes : disliked, liked, did not mention

        if params['use_dropout']:
            self.dropout = nn.Dropout(p=params['use_dropout'])

        if self.cuda_available:
            self.cuda()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                          lr=1e-4)
        self.criterion = None

    def forward(self, input_dict, pool="mean"):
        """

        :param input_dict: dictionary that contains the different inputs: dialogue, senders, movie_occurrences
        :return:
        """
        dialogue, senders, lengths = input_dict["dialogue"], input_dict["senders"], input_dict["lengths"]
        batch_size, max_conversation_length = dialogue.data.shape[:2]
        # order by descending utterance length
        lengths = lengths.reshape((-1))
        sorted_lengths, sorted_idx, rev = sort_for_packed_sequence(lengths, self.cuda_available)

        # reshape and reorder
        sorted_utterances = dialogue.view(batch_size * max_conversation_length, -1).index_select(0, sorted_idx)

        # consider sequences of length > 0 only
        num_positive_lengths = np.sum(lengths > 0)
        sorted_utterances = sorted_utterances[:num_positive_lengths]
        sorted_lengths = sorted_lengths[:num_positive_lengths]

        # apply GenSen model
        embedded, sentence_representations = self.gensen.get_representation_from_ordered(sorted_utterances,
                                                                                         lengths=sorted_lengths,
                                                                                         pool='last',
                                                                                         return_numpy=False)
        # (< batch_size * max conversation_length, 2048)
        # print("SENTENCE REP SHAPE", sentence_representations.data.shape)
        if self.params['sentence_encoder_num_layers'] > 0:
            packed_sentences = pack_padded_sequence(embedded, sorted_lengths, batch_first=True)
            # Apply encoder and get the final hidden states
            _, sentence_representations = self.sentence_encoder(packed_sentences)
            # (2*num_layers, < batch_size * max_conv_length, hidden_size) -> (< batch * conv_len, 2*hidden_size)
            # Concat the hidden states of the last layer (two directions of the GRU)
            sentence_representations = torch.cat((sentence_representations[-1], sentence_representations[-2]), 1)

        if self.params['use_dropout']:
            sentence_representations = self.dropout(sentence_representations)

        # Complete the missing sequences (of length 0)
        if num_positive_lengths < batch_size * max_conversation_length:
            tt = torch.cuda.FloatTensor if self.cuda_available else torch.FloatTensor
            pad_tensor = Variable(torch.zeros(
                batch_size * max_conversation_length - num_positive_lengths,
                SentimentAnalysisBaseline.gensen_out_size if self.params['sentence_encoder_num_layers'] == 0
                else 2 * self.params['hidden_size'],
                out=tt()
            ))
            sentence_representations = torch.cat((
                sentence_representations,
                pad_tensor
            ), 0)
        # print("SENTENCE REP SHAPE",
        #       sentence_representations.data.shape)  # (batch_size * max_conversation_length, 2048)
        # Retrieve original sentence order and reshape to (batch, conv, -1)
        sentence_representations = sentence_representations.index_select(0, rev).view(
            batch_size, max_conversation_length, -1)
        # Append sender information
        sentence_representations = torch.cat([sentence_representations, senders.unsqueeze(2)], 2)
        # print("SENTENCE REP SHAPE WITH SENDER INFO", sentence_representations.data.shape)
        #  (batch_size, max_conv_length, 2049)

        # for binary class output: compute sigmoid in the loss function (computational stability)
        # for 3-class outputs: compute log_softmax
        output = (F.log_softmax(self.Iseen(sentence_representations), dim=-1),
                  F.log_softmax(self.Iliked(sentence_representations), dim=-1),
                  F.log_softmax(self.Rseen(sentence_representations), dim=-1),
                  F.log_softmax(self.Rliked(sentence_representations), dim=-1))
        if pool == "best_confidence":
            # obtain the most confident prediction for each conversation
            # get the confidence for each sentence
            maxed = [torch.max(x, dim=2)[0] for x in output]  # array of (batch, conv)
            # get the indices of the sentences giving the maximum confidence, for each conversation in batch
            indices = [torch.max(x, dim=1)[1] for x in maxed]  # array of (batch)
            # gather the outputs corresponding to these sentences
            output = [torch.gather(x, dim=1, index=i.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, 3)).squeeze()
                      for x, i in zip(output, indices)]  # array of (batch, 3)
        elif pool == "mean":
            output = [torch.mean(x, dim=1) for x in output]
            output = [F.log_softmax(x, dim=-1) for x in output]
        else:
            raise ValueError("pool expected to be 'best_confidence' or 'mean'. Got {} instead".format(pool))
        return output  # tuple of (batch, 3) tensors

    def evaluate(self, batch_loader, print_matrices=False, subset="valid", pool="mean"):
        """
        Evaluate function
        :param print_matrices: if true, print confusion matrices at the end of evaluation
        :param subset: in {"valid", "train"}. Susbet on which to evaluate
        :return: the mean loss.
        """
        self.eval()
        batch_loader.batch_index[subset] = 0
        n_batches = batch_loader.n_batches[subset]

        total = 0
        correct = 0
        losses = []
        matrix_size = 18
        Iconfusion_matrix = np.zeros((matrix_size, matrix_size), dtype=int)
        Rconfusion_matrix = np.zeros((matrix_size, matrix_size), dtype=int)
        Iconfusion_matrix_no_disagreement = np.zeros((matrix_size, matrix_size), dtype=int)
        Rconfusion_matrix_no_disagreement = np.zeros((matrix_size, matrix_size), dtype=int)
        wrongs = 0
        wrongs_with_disagreement = 0
        for i in tqdm(range(n_batches)):
            # load batch
            batch = batch_loader.load_batch(subset, cut_dialogues=0)
            if self.cuda_available:
                batch["dialogue"] = batch["dialogue"].cuda()
                batch["forms"] = batch["forms"].cuda()
                batch["senders"] = batch["senders"].cuda()
            # compute output and loss
            target = batch["forms"].cpu().data.numpy()
            output = self.forward(batch, pool=pool)  # tuple of (batch, 3) tensors
            loss = self.criterion(output, batch["forms"])
            losses.append(loss.data[0])

            # get the arg max for the categorical output
            Iseen = torch.max(output[0], 1)[1].squeeze().cpu()
            Iliked = torch.max(output[1], 1)[1].squeeze().cpu()
            Rseen = torch.max(output[2], 1)[1].squeeze().cpu()
            Rliked = torch.max(output[3], 1)[1].squeeze().cpu()

            # increment number of wrong predictions (either seeker and recommender)
            wrongs += np.sum(1 * (Iliked.data.numpy() != target[:, 2]) + 1 * (Rliked.data.numpy() != target[:, 5]))
            # increment number of wrong predictions where the targets disagree (ambiguous dialogue or careless worker)
            wrongs_with_disagreement += np.sum(
                1 * (Iliked.data.numpy() != target[:, 2]) * (target[:, 2] != target[:, 5])
                + 1 * (Rliked.data.numpy() != target[:, 5]) * (target[:, 2] != target[:, 5]))

            if print_matrices:
                # Cartesian product of all three different targets
                Iclass = 2 * Iseen + 6 * Iliked
                Rclass = 2 * Rseen + 6 * Rliked
                Itargetclass = 2 * target[:, 1] + 6 * target[:, 2]
                Rtargetclass = 2 * target[:, 4] + 6 * target[:, 5]
                # marks examples where the targets agree
                filter_no_disagreement = target[:, 2] == target[:, 5]

                total += Itargetclass.shape[0]
                correct += (Iclass.data == torch.LongTensor(Itargetclass)).cpu().sum()
                # increment confusion matrices
                Iconfusion_matrix += sklearn.metrics.confusion_matrix(Itargetclass, Iclass.data.numpy(),
                                                                      labels=np.arange(18))
                Rconfusion_matrix += sklearn.metrics.confusion_matrix(Rtargetclass, Rclass.data.numpy(),
                                                                      labels=np.arange(18))
                # increment confusion matrices only taking the examples where the two workers agree
                Iconfusion_matrix_no_disagreement += sklearn.metrics.confusion_matrix(
                    Itargetclass[filter_no_disagreement],
                    Iclass.data.numpy()[filter_no_disagreement],
                    labels=np.arange(18))
                Rconfusion_matrix_no_disagreement += sklearn.metrics.confusion_matrix(
                    Rtargetclass[filter_no_disagreement],
                    Rclass.data.numpy()[filter_no_disagreement],
                    labels=np.arange(18))
        if print_matrices:
            # the reshape modelizes a block matrix.
            # then we sum the blocks to obtain marginal matrices.
            # confusion matrix for suggested/mentioned label
            Isugg_marginal = Iconfusion_matrix.reshape(9, 2, 9, 2).sum(axis=(0, 2))
            Rsugg_marginal = Rconfusion_matrix.reshape(9, 2, 9, 2).sum(axis=(0, 2))
            # confusion matrix that ignores the suggested/mentioned label
            I_marginal = Iconfusion_matrix.reshape(9, 2, 9, 2).sum(axis=(1, 3))
            R_marginal = Rconfusion_matrix.reshape(9, 2, 9, 2).sum(axis=(1, 3))
            # confusion matrix for the seen/not seen/did not say label
            Iseen_marginal = I_marginal.reshape(3, 3, 3, 3).sum(axis=(0, 2))
            Rseen_marginal = R_marginal.reshape(3, 3, 3, 3).sum(axis=(0, 2))
            # confusion matrix for the liked/disliked/did not say label
            Iliked_marginal = I_marginal.reshape(3, 3, 3, 3).sum(axis=(1, 3))
            Rliked_marginal = R_marginal.reshape(3, 3, 3, 3).sum(axis=(1, 3))
            Iliked_marginal_no_disagreement = Iconfusion_matrix_no_disagreement.reshape(3, 3, 2, 3, 3, 2) \
                .sum(axis=(1, 2, 4, 5))
            Rliked_marginal_no_disagreement = Rconfusion_matrix_no_disagreement.reshape(3, 3, 2, 3, 3, 2) \
                .sum(axis=(1, 2, 4, 5))
            print("marginals")
            print(I_marginal)
            print(R_marginal)
            print("Suggested marginals")
            print(Isugg_marginal)
            print(Rsugg_marginal)
            print("Seen marginals")
            print(Iseen_marginal)
            print(Rseen_marginal)
            print("Liked marginals")
            print(Iliked_marginal)
            print(Rliked_marginal)
            print("Liked marginals, excluding targets with disagreements")
            print(Iliked_marginal_no_disagreement)
            print(Rliked_marginal_no_disagreement)
        print("{} wrong answers for liked label, for {} of those there was a disagreement between workers"
              .format(wrongs, wrongs_with_disagreement))
        print("{} loss : {}".format(subset, np.mean(losses)))
        self.train()
        return np.mean(losses)


class SentimentAnalysisBaselineLoss(nn.Module):
    def __init__(self, class_weight, use_targets):
        super(SentimentAnalysisBaselineLoss, self).__init__()
        self.class_weight = class_weight
        # string that specifies which targets to considern and, if specified, the weights
        self.use_targets = use_targets
        if len(use_targets.split()) == 6:
            self.weights = [int(x) for x in use_targets.split()[:3]]
        else:
            self.weights = [1, 1, 1]
        self.suggested_criterion = nn.BCEWithLogitsLoss()
        self.seen_criterion = nn.NLLLoss()
        if self.class_weight and "liked" in self.class_weight:
            self.liked_criterion = nn.NLLLoss(weight=torch.Tensor(self.class_weight["liked"]))
        else:
            self.liked_criterion = nn.NLLLoss()
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, model_output, target):
        Iseen, Iliked, Rseen, Rliked = model_output
        loss = 0
        if "seen" in self.use_targets:
            loss += self.weights[1] * (self.seen_criterion(Iseen, target[:, 1])
                                       + self.seen_criterion(Rseen, target[:, 4]))
        if "liked" in self.use_targets:
            loss += self.weights[2] * (self.liked_criterion(Iliked, target[:, 2])
                                       + self.liked_criterion(Rliked, target[:, 5]))
        return loss
