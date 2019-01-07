import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm

from utils import sort_for_packed_sequence
from models.hierarchical_rnn import HRNN
from models.decoders import TextDecoder


class HRED(nn.Module):
    def __init__(self,
                 train_vocab,
                 params=None,):
        super(HRED, self).__init__()
        self.params = params
        self.train_vocab = train_vocab
        self.cuda_available = torch.cuda.is_available()

        # HRNN encoder
        # Conversation encoder not bidirectional
        self.encoder = HRNN(params=params['hrnn_params'],
                            gensen=True,
                            train_vocabulary=train_vocab,
                            train_gensen=False,
                            conv_bidirectional=False)
        self.decoder = TextDecoder(
            context_size=params['hrnn_params']['conversation_encoder_hidden_size'],
            vocab_size=len(train_vocab),
            **params['decoder_params']
        )
        if self.cuda_available:
            self.cuda()
        self.decoder.set_pretrained_embeddings(self.encoder.gensen.encoder.src_embedding.weight.data)

    def forward(self, input_dict):
        # encoder result: (batch_size, max_conv_length, hidden_size)
        conversation_representations = self.encoder(input_dict, return_all=True)
        batch_size, max_conversation_length, max_utterance_length = input_dict["dialogue"].data.shape

        # Decoder:
        utterances = input_dict["dialogue"].view(batch_size * max_conversation_length, -1)
        lengths = input_dict["lengths"]
        # order by descending utterance length
        lengths = lengths.reshape((-1))
        sorted_lengths, sorted_idx, rev = sort_for_packed_sequence(lengths, cuda=self.cuda_available)

        sorted_utterances = utterances.index_select(0, sorted_idx)

        # shift the context vectors one step in time
        tt = torch.cuda.FloatTensor if self.cuda_available else torch.FloatTensor
        pad_tensor = (Variable(torch.zeros(
            batch_size, 1, self.params['hrnn_params']['conversation_encoder_hidden_size'], out=tt())))

        conversation_representations = torch.cat((pad_tensor, conversation_representations), 1).narrow(
            1, 0, max_conversation_length)
        # and reshape+reorder the same way as utterances
        conversation_representations = conversation_representations.contiguous().view(
            batch_size * max_conversation_length, self.params['hrnn_params']['conversation_encoder_hidden_size']).\
            index_select(0, sorted_idx)

        # consider only lengths > 0
        num_positive_lengths = np.sum(lengths > 0)
        sorted_utterances = sorted_utterances[:num_positive_lengths]
        sorted_lengths = sorted_lengths[:num_positive_lengths]
        conversation_representations = conversation_representations[:num_positive_lengths]

        # Run decoder
        outputs = F.log_softmax(self.decoder(
            sorted_utterances,
            sorted_lengths,
            conversation_representations,
            log_probabilities=True
        ).transpose(0, 2)).transpose(0, 2)

        # Complete the missing sequences (of length 0)
        if num_positive_lengths < batch_size * max_conversation_length:
            tt = torch.cuda.FloatTensor if self.cuda_available else torch.FloatTensor
            pad_tensor = Variable(torch.zeros(
                batch_size * max_conversation_length - num_positive_lengths,
                max_utterance_length,
                len(self.train_vocab),
                out=tt()
            ))
            outputs = torch.cat((
                outputs,
                pad_tensor
            ), 0)

        # print("OUTPUT SHAPE :", outputs.data.shape) # (batch * max_conv_len, max_sentence_len, vocab)
        # retrieve original order
        outputs = outputs.index_select(0, rev). \
            view(batch_size, max_conversation_length, max_utterance_length, -1)
        # print("OUTPUT SHAPE RETRIEVED IN ORDER", outputs.data.shape) # (batch, max_conv_len, max_sentence_len, vocab)
        return outputs

    def train_iter(self, batch, criterion):
        outputs = self.forward(batch)

        batch_size, max_conv_length, max_seq_length, vocab_size = outputs.data.shape
        # indices of recommender's utterances(< batch * max_conv_len)
        idx = Variable(torch.nonzero((batch["senders"].view(-1) == -1).data).squeeze())
        # select recommender's utterances for the loss
        outputs = outputs.view(-1, max_seq_length, vocab_size).index_select(0, idx)
        target = batch["target"].view(-1, max_seq_length).index_select(0, idx)

        loss = criterion(outputs.view(-1, vocab_size), target.view(-1))
        # backward pass
        loss.backward()
        return loss.data[0]

    def evaluate(self, batch_loader, criterion, subset="valid"):
        """
        Evaluate function
        :param subset: in {"valid", "train"}. Susbet on which to evaluate
        :return: the mean loss.
        """
        self.eval()
        batch_loader.batch_index[subset] = 0
        n_batches = batch_loader.n_batches[subset]

        losses = []
        for _ in tqdm(range(n_batches)):
            # load batch
            batch = batch_loader.load_batch(subset=subset)
            if self.cuda_available:
                batch["dialogue"] = batch["dialogue"].cuda()
                batch["target"] = batch["target"].cuda()
                batch["senders"] = batch["senders"].cuda()
            # compute output and loss
            outputs = self.forward(batch)

            batch_size, max_conv_length, max_seq_length, vocab_size = outputs.data.shape
            # indices of recommender's utterances(< batch * max_conv_len)
            idx = Variable(torch.nonzero((batch["senders"].view(-1) == -1).data).squeeze())
            # select recommender's utterances for the loss
            outputs = outputs.view(-1, max_seq_length, vocab_size).index_select(0, idx)
            target = batch["target"].view(-1, max_seq_length).index_select(0, idx)

            loss = criterion(outputs.view(-1, vocab_size), target.view(-1))
            losses.append(loss.data[0])
        print("{} loss : {}".format(subset, np.mean(losses)))
        self.train()
        return np.mean(losses)
