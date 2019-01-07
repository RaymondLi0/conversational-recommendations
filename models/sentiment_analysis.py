import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sklearn.metrics
from tqdm import tqdm
from models.hierarchical_rnn import HRNN


class SentimentAnalysis(nn.Module):
    def __init__(self,
                 train_vocab,
                 params,
                 gensen=True,
                 train_gensen=False,
                 conv_bidrectional=False,
                 resume=None):
        super(SentimentAnalysis, self).__init__()
        self.params = params
        self.cuda_available = torch.cuda.is_available()
        self.train_gensen = train_gensen

        if resume is not None:
            if self.cuda_available:
                checkpoint = torch.load(resume)
            else:
                checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        self.encoder = HRNN(
            params=params['hrnn_params'],
            train_vocabulary=train_vocab,
            gensen=gensen,
            train_gensen=self.train_gensen,
            conv_bidirectional=conv_bidrectional,
        )

        encoder_output_size = self.params['hrnn_params']['conversation_encoder_hidden_size']
        # Outputs
        self.Isuggested = nn.Linear((1 + conv_bidrectional) * encoder_output_size, 1)
        # 3 classes : not seen, seen, did not mention
        self.Iseen = nn.Linear((1 + conv_bidrectional) * encoder_output_size, 3)
        # 3 classes : disliked, liked, did not mention
        self.Iliked = nn.Linear((1 + conv_bidrectional) * encoder_output_size, 3)
        self.Rsuggested = nn.Linear((1 + conv_bidrectional) * encoder_output_size, 1)
        # 3 classes : not seen, seen, did not mention
        self.Rseen = nn.Linear((1 + conv_bidrectional) * encoder_output_size, 3)
        # 3 classes : disliked, liked, did not mention
        self.Rliked = nn.Linear((1 + conv_bidrectional) * encoder_output_size, 3)

        if self.cuda_available:
            self.cuda()

        if resume is not None:
            # load weights from saved model
            self.load_state_dict(checkpoint['state_dict'])

    def forward(self, input, return_liked_probability=False):
        """

        :param input: dictionary that contains the different inputs: dialogue, senders, movie_occurrences
        :return:
        """
        if return_liked_probability:
            # return the liked probability at each utterance in the dialogue
            # (batch_size, max_conv_length, hidden_size*num_directions)
            conversation_representations = self.encoder(input, return_all=True)
            # return the probability of the "liked" class
            return F.softmax(self.Iliked(conversation_representations), dim=-1)[:, :, 1]
        else:
            # return the sentiment considering the whole dialogue
            # for binary class output: compute sigmoid in the loss function (computational stability)
            # for 3-class outputs: compute log_softmax
            # (batch_size, hidden_size*num_directions)
            conversation_representations = self.encoder(input, return_all=False)
            return (self.Isuggested(conversation_representations),
                    F.log_softmax(self.Iseen(conversation_representations), dim=-1),
                    F.log_softmax(self.Iliked(conversation_representations), dim=-1),
                    self.Rsuggested(conversation_representations),
                    F.log_softmax(self.Rseen(conversation_representations), dim=-1),
                    F.log_softmax(self.Rliked(conversation_representations), dim=-1))

    def evaluate(self, batch_loader, criterion, print_matrices=False, subset="valid"):
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
            batch = batch_loader.load_batch(subset)
            if self.cuda_available:
                batch["dialogue"] = batch["dialogue"].cuda()
                batch["forms"] = batch["forms"].cuda()
                batch["senders"] = batch["senders"].cuda()
                batch["movie_occurrences"] = batch["movie_occurrences"].cuda()
            # compute output and loss
            target = batch["forms"].cpu().data.numpy()
            output = self.forward(batch)
            loss = criterion(output, batch["forms"])
            losses.append(loss.data[0])

            Isugg = (output[0] > 0.5).squeeze().cpu().long()
            # get the arg max for the categorical output
            Iseen = torch.max(output[1], 1)[1].squeeze().cpu()
            Iliked = torch.max(output[2], 1)[1].squeeze().cpu()
            Rsugg = (output[3] > 0.5).squeeze().cpu().long()
            Rseen = torch.max(output[4], 1)[1].squeeze().cpu()
            Rliked = torch.max(output[5], 1)[1].squeeze().cpu()

            # increment number of wrong predictions (either seeker and recommender)
            wrongs += np.sum(1 * (Iliked.data.numpy() != target[:, 2]) + 1 * (Rliked.data.numpy() != target[:, 5]))
            # increment number of wrong predictions where the targets disagree (ambiguous dialogue or careless worker)
            wrongs_with_disagreement += np.sum(
                1 * (Iliked.data.numpy() != target[:, 2]) * (target[:, 2] != target[:, 5])
                + 1 * (Rliked.data.numpy() != target[:, 5]) * (target[:, 2] != target[:, 5]))

            if print_matrices:
                # Cartesian product of all three different targets
                Iclass = Isugg + 2 * Iseen + 6 * Iliked
                Rclass = Rsugg + 2 * Rseen + 6 * Rliked
                Itargetclass = target[:, 0] + 2 * target[:, 1] + 6 * target[:, 2]
                Rtargetclass = target[:, 3] + 2 * target[:, 4] + 6 * target[:, 5]
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


class SentimentAnalysisLoss(nn.Module):
    def __init__(self, class_weight, use_targets):
        super(SentimentAnalysisLoss, self).__init__()
        self.class_weight = class_weight
        # string that specifies which targets to consider and, if specified, the weights
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
        Isuggested, Iseen, Iliked, Rsuggested, Rseen, Rliked = model_output
        loss = 0
        if "suggested" in self.use_targets:
            loss += self.weights[0] * (self.suggested_criterion(Isuggested.squeeze(), target[:, 0].float())
                                       + self.suggested_criterion(Rsuggested.squeeze(), target[:, 3].float()))
        if "seen" in self.use_targets:
            loss += self.weights[1] * (self.seen_criterion(Iseen, target[:, 1])
                                       + self.seen_criterion(Rseen, target[:, 4]))
        if "liked" in self.use_targets:
            loss += self.weights[2] * (self.liked_criterion(Iliked, target[:, 2])
                                       + self.liked_criterion(Rliked, target[:, 5]))
        return loss
