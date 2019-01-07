import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import numpy as np
import os

import config
from models.gensen import GenSenSingle
from utils import sort_for_packed_sequence


class HRNN(nn.Module):
    """
    Hierarchical Recurrent Neural Network

    params.keys() ['use_gensen', 'use_movie_occurrences', 'sentence_encoder_hidden_size',
    'conversation_encoder_hidden_size', 'sentence_encoder_num_layers', 'conversation_encoder_num_layers', 'use_dropout',
    ['embedding_dimension']]

    Input: Input["dialogue"] (batch, max_conv_length, max_utterance_length) Long Tensor
           Input["senders"] (batch, max_conv_length) Float Tensor
           Input["lengths"] (batch, max_conv_length) list
           (optional) Input["movie_occurrences"] (batch, max_conv_length, max_utterance_length) for word occurence
                                                 (batch, max_conv_length) for sentence occurrence. Float Tensor
    """

    def __init__(self,
                 params,
                 gensen=False,
                 train_vocabulary=None,
                 train_gensen=True,
                 conv_bidirectional=False):
        super(HRNN, self).__init__()
        self.params = params
        self.use_gensen = bool(gensen)
        self.train_gensen = train_gensen
        self.conv_bidirectional = conv_bidirectional

        self.cuda_available = torch.cuda.is_available()

        # Use instance of gensen if provided
        if isinstance(gensen, GenSenSingle):
            # Assume that vocab expansion is already run on gensen
            self.gensen = gensen
            self.word2id = self.gensen.task_word2id
            # freeze gensen's weights
            if not self.train_gensen:
                for param in self.gensen.parameters():
                    param.requires_grad = False
        # Otherwise instantiate a new gensen module
        elif self.use_gensen:
            self.gensen = GenSenSingle(
                model_folder=os.path.join(config.MODELS_PATH, 'GenSen'),
                filename_prefix='nli_large',
                pretrained_emb=os.path.join(config.MODELS_PATH, 'embeddings/glove.840B.300d.h5'),
                cuda=self.cuda_available
            )
            self.gensen.vocab_expansion(list(train_vocabulary))
            self.word2id = self.gensen.task_word2id
            # freeze gensen's weights
            if not self.train_gensen:
                for param in self.gensen.parameters():
                    param.requires_grad = False
        else:
            self.src_embedding = nn.Embedding(
                num_embeddings=len(train_vocabulary),
                embedding_dim=params['embedding_dimension']
            )
            self.word2id = {word: idx for idx, word in enumerate(train_vocabulary)}
            self.id2word = {idx: word for idx, word in enumerate(train_vocabulary)}
        self.sentence_encoder = nn.GRU(
            input_size=2048 + (self.params['use_movie_occurrences'] == "word") if self.use_gensen
            else self.params['embedding_dimension'] + (self.params['use_movie_occurrences'] == "word"),
            hidden_size=self.params['sentence_encoder_hidden_size'],
            num_layers=self.params['sentence_encoder_num_layers'],
            batch_first=True,
            bidirectional=True
        )
        self.conversation_encoder = nn.GRU(
            input_size=2 * self.params['sentence_encoder_hidden_size']
                       + 1 + (self.params['use_movie_occurrences'] == "sentence"),
            # concatenation of 2 directions for sentence encoders + sender informations + movie occurences
            hidden_size=self.params['conversation_encoder_hidden_size'],
            num_layers=self.params['conversation_encoder_num_layers'],
            batch_first=True,
            bidirectional=conv_bidirectional
        )
        if self.params['use_dropout']:
            self.dropout = nn.Dropout(p=self.params['use_dropout'])

    def get_sentence_representations(self, dialogue, senders, lengths, movie_occurrences=None):
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

        if self.use_gensen:
            # apply GenSen model and use outputs as word embeddings
            embedded, _ = self.gensen.get_representation_from_ordered(sorted_utterances,
                                                                      lengths=sorted_lengths,
                                                                      pool='last',
                                                                      return_numpy=False)
        else:
            embedded = self.src_embedding(sorted_utterances)
        # (< batch_size * max conversation_length, max_sentence_length, embedding_size/2048 for gensen)
        # print("EMBEDDED SHAPE", embedded.data.shape)

        if self.params['use_dropout']:
            embedded = self.dropout(embedded)

        if self.params['use_movie_occurrences'] == "word":
            if movie_occurrences is None:
                raise ValueError("Please specify movie occurrences")
            # reshape and reorder movie occurrences by utterance length
            movie_occurrences = movie_occurrences.view(
                batch_size * max_conversation_length, -1).index_select(0, sorted_idx)
            # keep indices where sequence_length > 0
            movie_occurrences = movie_occurrences[:num_positive_lengths]
            embedded = torch.cat((embedded, movie_occurrences.unsqueeze(2)), 2)

        packed_sentences = pack_padded_sequence(embedded, sorted_lengths, batch_first=True)
        # Apply encoder and get the final hidden states
        _, sentence_representations = self.sentence_encoder(packed_sentences)
        # (2*num_layers, < batch_size * max_conv_length, hidden_size)
        # Concat the hidden states of the last layer (two directions of the GRU)
        sentence_representations = torch.cat((sentence_representations[-1], sentence_representations[-2]), 1)

        if self.params['use_dropout']:
            sentence_representations = self.dropout(sentence_representations)

        # Complete the missing sequences (of length 0)
        if num_positive_lengths < batch_size * max_conversation_length:
            tt = torch.cuda.FloatTensor if self.cuda_available else torch.FloatTensor
            pad_tensor = Variable(torch.zeros(
                batch_size * max_conversation_length - num_positive_lengths,
                2 * self.params['sentence_encoder_hidden_size'],
                out=tt()
            ))
            sentence_representations = torch.cat((
                sentence_representations,
                pad_tensor
            ), 0)
        # print("SENTENCE REP SHAPE",
        #       sentence_representations.data.shape)  # (batch_size * max_conversation_length, 2*hidden_size)
        # Retrieve original sentence order and Reshape to separate conversations
        sentence_representations = sentence_representations.index_select(0, rev).view(
            batch_size,
            max_conversation_length,
            2 * self.params['sentence_encoder_hidden_size'])
        # Append sender information
        sentence_representations = torch.cat([sentence_representations, senders.unsqueeze(2)], 2)
        # Append movie occurrence information if required
        if self.params['use_movie_occurrences'] == "sentence":
            if movie_occurrences is None:
                raise ValueError("Please specify movie occurrences")
            sentence_representations = torch.cat((sentence_representations, movie_occurrences.unsqueeze(2)), 2)
        # print("SENTENCE REP SHAPE WITH SENDER INFO", sentence_representations.data.shape)
        #  (batch_size, max_conv_length, 513 + self.params['use_movie_occurrences'])
        return sentence_representations

    def forward(self, input_dict, return_all=True, return_sentence_representations=False):
        movie_occurrences = input_dict["movie_occurrences"] if self.params['use_movie_occurrences'] else None
        # get sentence representations
        sentence_representations = self.get_sentence_representations(
            input_dict["dialogue"], input_dict["senders"], lengths=input_dict["lengths"],
            movie_occurrences=movie_occurrences)
        # (batch_size, max_conv_length, 2*sent_hidden_size + 1 + use_movie_occurences)
        # Pass whole conversation into GRU
        lengths = input_dict["conversation_lengths"]
        sorted_lengths, sorted_idx, rev = sort_for_packed_sequence(lengths, self.cuda_available)

        # reorder in decreasing sequence length
        sorted_representations = sentence_representations.index_select(0, sorted_idx)
        packed_sequences = pack_padded_sequence(sorted_representations, sorted_lengths, batch_first=True)
        conversation_representations, last_state = self.conversation_encoder(packed_sequences)

        # retrieve original order
        conversation_representations, _ = pad_packed_sequence(conversation_representations, batch_first=True)
        conversation_representations = conversation_representations.index_select(0, rev)
        # print("LAST STATE SHAPE", last_state.data.shape) # (num_layers * num_directions, batch, conv_hidden_size)
        last_state = last_state.index_select(1, rev)
        if self.params['use_dropout']:
            conversation_representations = self.dropout(conversation_representations)
            last_state = self.dropout(last_state)
        if return_all:
            if not return_sentence_representations:
                # return the last layer of the GRU for each t.
                # (batch_size, max_conv_length, hidden_size*num_directions
                return conversation_representations
            else:
                # also return sentence representations
                return conversation_representations, sentence_representations
        else:
            # get the last hidden state only
            if self.conv_bidirectional:
                # Concat the hidden states for the last layer (two directions of the GRU)
                last_state = torch.cat((last_state[-1], last_state[-2]), 1)
                # (batch_size, num_directions*hidden_size)
                return last_state
            else:
                # Return the hidden state from the last layers
                return last_state[-1]
