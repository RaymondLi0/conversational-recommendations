"""GenSen Encoder. Code from https://gist.github.com/MaximumEntropy/f27766f5bcffc10f1e147bc3612d4b06
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import h5py
from sklearn.linear_model import LinearRegression
import nltk
import numpy as np
import pickle
import os


class Encoder(nn.Module):
    """GenSen Encoder."""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        """Initialize params."""
        super(Encoder, self).__init__()
        self.src_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

        self.encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

    def set_pretrained_embeddings(self, embedding_matrix):
        """Set embedding weights."""
        self.src_embedding.weight.data.set_(torch.from_numpy(embedding_matrix))

    def forward(self, input, lengths, return_all=False, pool='last'):
        """Propogate input through the encoder."""
        embedding = self.src_embedding(input)
        src_emb = pack_padded_sequence(embedding, lengths, batch_first=True)
        h, h_t = self.encoder(src_emb)

        # Get last hidden state via max-pooling or h_t
        if pool == 'last':
            h_t = torch.cat((h_t[-1], h_t[-2]), 1)
        elif pool == 'max':
            h_tmp, _ = pad_packed_sequence(h, batch_first=True)
            h_t = torch.max(h_tmp, 1)[0].squeeze()
        else:
            raise ValueError("Pool %s is not valid " % (pool))

        # Return all or only the last hidden state
        if return_all:
            h, _ = pad_packed_sequence(h, batch_first=True)
            return h, h_t
        else:
            return h_t


class GenSen(nn.Module):
    """GenSen Wrapper."""

    def __init__(
            self, model_folder, filename_prefix_1,
            filename_prefix_2, pretrained_emb, cuda=False
    ):
        """Initialize params."""
        super(GenSen, self).__init__()
        self.model_folder = model_folder
        self.filename_prefix_1 = filename_prefix_1
        self.filename_prefix_2 = filename_prefix_2
        self.pretrained_emb = pretrained_emb
        self.cuda = cuda
        self._load_params()
        self.vocab_expanded = False

    def _load_params(self):
        """Load pretrained params."""
        # Read vocab pickle files
        model_1_vocab = pickle.load(
            open(os.path.join(
                self.model_folder,
                '%s_vocab.pkl' % (self.filename_prefix_1)
            ))
        )

        model_2_vocab = pickle.load(
            open(os.path.join(
                self.model_folder,
                '%s_vocab.pkl' % (self.filename_prefix_2)
            ))
        )

        # Word to index mappings
        self.word2id_1 = model_1_vocab['word2id']
        self.word2id_2 = model_2_vocab['word2id']

        self.id2word_1 = model_1_vocab['id2word']
        self.id2word_2 = model_2_vocab['id2word']

        encoder_1_model = torch.load(os.path.join(
            self.model_folder,
            '%s.model' % (self.filename_prefix_1)
        ))
        encoder_2_model = torch.load(os.path.join(
            self.model_folder,
            '%s.model' % (self.filename_prefix_2)
        ))

        # Initialize encoders
        self.encoder_1 = Encoder(
            vocab_size=encoder_1_model['src_embedding.weight'].size(0),
            embedding_dim=encoder_1_model['src_embedding.weight'].size(1),
            hidden_dim=encoder_1_model['encoder.weight_hh_l0'].size(1),
            num_layers=1 if len(encoder_1_model) < 10 else 2
            # This is hack that works only for bidirectional
            # encoders with either 1 or 2 layers
        )

        self.encoder_2 = Encoder(
            vocab_size=encoder_2_model['src_embedding.weight'].size(0),
            embedding_dim=encoder_2_model['src_embedding.weight'].size(1),
            hidden_dim=encoder_2_model['encoder.weight_hh_l0'].size(1),
            num_layers=1 if len(encoder_2_model) < 10 else 2
            # This is hack that works only for bidirectional
            # encoders with either 1 or 2 layers
        )

        # Load pretrained sentence encoder weights
        self.encoder_1.load_state_dict(encoder_1_model)
        self.encoder_2.load_state_dict(encoder_2_model)

        # Set encoders in eval model.
        self.encoder_1.eval()
        self.encoder_2.eval()

    def first_expansion(self):
        """Traing linear regression model for the first time."""
        # Read pre-trained word embedding h5 file
        print('Loading pretrained word embeddings')
        pretrained_embeddings = h5py.File(self.pretrained_emb)
        pretrained_embedding_matrix = pretrained_embeddings['embedding'].value
        pretrain_vocab = \
            pretrained_embeddings['words_flatten'].value.split('\n')
        pretrain_word2id = {
            word: ind for ind, word in enumerate(pretrain_vocab)
        }

        # Set up training data for vocabulary expansion
        model_train_1 = []
        model_train_2 = []
        pretrain_train_1 = []
        pretrain_train_2 = []

        for word in pretrain_word2id:
            if word in self.word2id_1:
                model_train_1.append(
                    self.model_embedding_matrix_1[self.word2id_1[word]]
                )
                pretrain_train_1.append(
                    pretrained_embedding_matrix[pretrain_word2id[word]]
                )

        for word in pretrain_word2id:
            if word in self.word2id_2:
                model_train_2.append(
                    self.model_embedding_matrix_2[self.word2id_2[word]]
                )
                pretrain_train_2.append(
                    pretrained_embedding_matrix[pretrain_word2id[word]]
                )

        print('Training vocab expansion on model 1')
        lreg_1 = LinearRegression()
        lreg_1.fit(pretrain_train_1, model_train_1)

        print('Training vocab expansion on model 2')
        lreg_2 = LinearRegression()
        lreg_2.fit(pretrain_train_2, model_train_2)

        self.lreg_1 = lreg_1
        self.lreg_2 = lreg_2
        self.pretrain_word2id = pretrain_word2id
        self.pretrained_embedding_matrix = pretrained_embedding_matrix

    def vocab_expansion(self, task_vocab):
        """Expand the model's vocabulary with pretrained word embeddings."""
        self.task_word2id = {word: idx for idx, word in enumerate(task_vocab)}
        self.task_id2word = {idx: word for idx, word in enumerate(task_vocab)}

        if not self.vocab_expanded:
            self.model_embedding_matrix_1 = \
                self.encoder_1.src_embedding.weight.data.cpu().numpy()
            self.model_embedding_matrix_2 = \
                self.encoder_2.src_embedding.weight.data.cpu().numpy()
            self.first_expansion()

        # Expand vocabulary using the linear regression model
        task_embeddings_1 = []
        task_embeddings_2 = []
        oov_pretrain = 0
        oov_task = 0

        for word in task_vocab:
            if word in self.word2id_1:
                task_embeddings_1.append(
                    self.model_embedding_matrix_1[self.word2id_1[word]]
                )
            elif word in self.pretrain_word2id:
                oov_task += 1
                task_embeddings_1.append(self.lreg_1.predict(
                    self.pretrained_embedding_matrix[self.pretrain_word2id[word]].reshape(1, -1)
                ).squeeze().astype(np.float32))
            else:
                oov_pretrain += 1
                oov_task += 1
                task_embeddings_1.append(
                    self.model_embedding_matrix_1[self.word2id_1['<unk>']]
                )

        for word in task_vocab:
            if word in self.word2id_2:
                task_embeddings_2.append(
                    self.model_embedding_matrix_2[self.word2id_2[word]]
                )
            elif word in self.pretrain_word2id:
                oov_task += 1
                task_embeddings_2.append(self.lreg_2.predict(
                    self.pretrained_embedding_matrix[self.pretrain_word2id[word]].reshape(1, -1)
                ).squeeze().astype(np.float32))
            else:
                oov_pretrain += 1
                oov_task += 1
                task_embeddings_2.append(
                    self.model_embedding_matrix_2[self.word2id_2['<unk>']]
                )

        print('Found %d task OOVs ' % (oov_task))
        print('Found %d pretrain OOVs ' % (oov_pretrain))

        task_embeddings_1 = np.stack(task_embeddings_1)
        task_embeddings_2 = np.stack(task_embeddings_2)

        self.encoder_1.set_pretrained_embeddings(task_embeddings_1)
        self.encoder_2.set_pretrained_embeddings(task_embeddings_2)

        self.vocab_expanded = True

        if self.cuda:
            self.encoder_1 = self.encoder_1.cuda()
            self.encoder_2 = self.encoder_2.cuda()

    def get_minibatch(self, sentences, tokenize=False):
        """Prepare minibatch."""
        if tokenize:
            sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
        else:
            sentences = [sentence.split() for sentence in sentences]

        sentences = [['<s>'] + sentence + ['</s>'] for sentence in sentences]

        lens = [len(sentence) for sentence in sentences]
        sorted_idx = np.argsort(lens)[::-1]
        sorted_sentences = [sentences[idx] for idx in sorted_idx]
        rev = np.argsort(sorted_idx)
        sorted_lens = [len(sentence) for sentence in sorted_sentences]
        max_len = max(sorted_lens)

        sentences = [
            [self.task_word2id[w] if w in self.task_word2id else self.task_word2id['<unk>'] for w in sentence] +
            [self.task_word2id['<pad>']] * (max_len - len(sentence))
            for sentence in sorted_sentences
        ]

        sentences = Variable(torch.LongTensor(sentences), volatile=True)
        rev = Variable(torch.LongTensor(rev), volatile=True)
        lengths = sorted_lens

        if self.cuda:
            sentences = sentences.cuda()
            rev = rev.cuda()

        return {
            'sentences': sentences,
            'lengths': lengths,
            'rev': rev
        }

    def get_representation(
            self, sentences, return_all=False,
            pool='last', tokenize=False, return_numpy=True
    ):
        """Get model representations."""
        if not self.vocab_expanded:
            raise ValueError("Please call vocab_expansion first")

        minibatch = self.get_minibatch(sentences, tokenize=tokenize)
        if return_all:
            h_1, h_t_1 = self.encoder_1(
                input=minibatch['sentences'], lengths=minibatch['lengths'],
                return_all=return_all, pool=pool
            )
            h_2, h_t_2 = self.encoder_2(
                input=minibatch['sentences'], lengths=minibatch['lengths'],
                return_all=return_all, pool=pool
            )
            h = torch.cat([h_1, h_2], 2).index_select(0, minibatch['rev'])
            h_t = torch.cat(
                [h_t_1, h_t_2], 1
            ).index_select(0, minibatch['rev'])
            if return_numpy:
                return h.data.cpu().numpy(), h_t.data.cpu().numpy()
            else:
                return h, h_t
        else:
            h_t_1 = self.encoder_1(
                input=minibatch['sentences'], lengths=minibatch['lengths'],
                return_all=return_all, pool=pool
            )
            h_t_2 = self.encoder_2(
                input=minibatch['sentences'], lengths=minibatch['lengths'],
                return_all=return_all, pool=pool
            )
            h_t = torch.cat(
                [h_t_1, h_t_2], 1
            ).index_select(0, minibatch['rev'])
            if return_numpy:
                return h_t.data.cpu().numpy()
            else:
                return h_t


class GenSenSingle(nn.Module):
    """GenSen Wrapper."""

    def __init__(
            self, model_folder, filename_prefix,
            pretrained_emb, cuda=False
    ):
        """Initialize params."""
        super(GenSenSingle, self).__init__()
        self.model_folder = model_folder
        self.filename_prefix = filename_prefix
        self.pretrained_emb = pretrained_emb
        self.cuda = cuda
        self._load_params()
        self.vocab_expanded = False

    def _load_params(self):
        """Load pretrained params."""
        # Read vocab pickle files
        model_vocab = pickle.load(
            open(os.path.join(
                self.model_folder,
                '%s_vocab.pkl' % (self.filename_prefix)
            ))
        )

        # Word to index mappings
        self.word2id = model_vocab['word2id']
        self.id2word = model_vocab['id2word']

        if self.cuda:
            encoder_model = torch.load(os.path.join(
                self.model_folder,
                '%s.model' % (self.filename_prefix)
            ))
        else:
            encoder_model = torch.load(os.path.join(
                self.model_folder,
                '%s.model' % (self.filename_prefix)
            ), map_location=lambda storage, loc: storage)

        # Initialize encoders
        self.encoder = Encoder(
            vocab_size=encoder_model['src_embedding.weight'].size(0),
            embedding_dim=encoder_model['src_embedding.weight'].size(1),
            hidden_dim=encoder_model['encoder.weight_hh_l0'].size(1),
            num_layers=1 if len(encoder_model) < 10 else 2
            # This is hack that works only for bidirectional encoders
            # with either 1 or 2 layers
        )

        # Load pretrained sentence encoder weights
        self.encoder.load_state_dict(encoder_model)

        # Set encoders in eval model.
        self.encoder.eval()

    def first_expansion(self):
        """Traing linear regression model for the first time."""
        # Read pre-trained word embedding h5 file
        print('Loading pretrained word embeddings')
        pretrained_embeddings = h5py.File(self.pretrained_emb)
        pretrained_embedding_matrix = pretrained_embeddings['embedding'].value
        pretrain_vocab = \
            pretrained_embeddings['words_flatten'].value.split('\n')
        pretrain_word2id = {
            word: ind for ind, word in enumerate(pretrain_vocab)
        }

        # Set up training data for vocabulary expansion
        model_train = []
        pretrain_train = []

        for word in pretrain_word2id:
            if word in self.word2id:
                model_train.append(
                    self.model_embedding_matrix[self.word2id[word]]
                )
                pretrain_train.append(
                    pretrained_embedding_matrix[pretrain_word2id[word]]
                )

        print('Training vocab expansion on model')
        lreg = LinearRegression()
        lreg.fit(pretrain_train, model_train)
        self.lreg = lreg
        self.pretrain_word2id = pretrain_word2id
        self.pretrained_embedding_matrix = pretrained_embedding_matrix

    def vocab_expansion(self, task_vocab):
        """Expand the model's vocabulary with pretrained word embeddings."""
        self.task_word2id = {word: idx for idx, word in enumerate(task_vocab)}
        self.task_id2word = {idx: word for idx, word in enumerate(task_vocab)}

        if not self.vocab_expanded:
            self.model_embedding_matrix = \
                self.encoder.src_embedding.weight.data.cpu().numpy()
            self.first_expansion()

        # Expand vocabulary using the linear regression model
        task_embeddings = []
        oov_pretrain = 0
        oov_task = 0

        for word in task_vocab:
            if word in self.word2id:
                task_embeddings.append(
                    self.model_embedding_matrix[self.word2id[word]]
                )
            elif word in self.pretrain_word2id:
                oov_task += 1
                task_embeddings.append(self.lreg.predict(
                    self.pretrained_embedding_matrix[self.pretrain_word2id[word]].reshape(1, -1)
                ).squeeze().astype(np.float32))
            else:
                oov_pretrain += 1
                oov_task += 1
                task_embeddings.append(
                    self.model_embedding_matrix[self.word2id['<unk>']]
                )

        print('Found %d task OOVs ' % (oov_task))
        print('Found %d pretrain OOVs ' % (oov_pretrain))

        task_embeddings = np.stack(task_embeddings)

        self.encoder.set_pretrained_embeddings(task_embeddings)
        self.vocab_expanded = True

        # Move encoder to GPU if self.cuda
        if self.cuda:
            self.encoder = self.encoder.cuda()

    def get_minibatch(self, sentences, tokenize=False):
        """Prepare minibatch."""
        if tokenize:
            sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
        else:
            sentences = [sentence.split() for sentence in sentences]

        sentences = [['<s>'] + sentence + ['</s>'] for sentence in sentences]

        lens = [len(sentence) for sentence in sentences]
        sorted_idx = np.argsort(lens)[::-1]
        sorted_sentences = [sentences[idx] for idx in sorted_idx]
        rev = np.argsort(sorted_idx)
        sorted_lens = [len(sentence) for sentence in sorted_sentences]
        max_len = max(sorted_lens)

        sentences = [
            [self.task_word2id[w] if w in self.task_word2id else self.task_word2id['<unk>'] for w in sentence] +
            [self.task_word2id['<pad>']] * (max_len - len(sentence))
            for sentence in sorted_sentences
        ]

        sentences = Variable(torch.LongTensor(sentences), volatile=False)
        rev = Variable(torch.LongTensor(rev), volatile=False)
        lengths = sorted_lens

        if self.cuda:
            sentences = sentences.cuda()
            rev = rev.cuda()

        return {
            'sentences': sentences,
            'lengths': lengths,
            'rev': rev
        }

    def get_representation(self, sentences, pool='last', tokenize=False, return_numpy=True):
        """Get model representations."""
        if not self.vocab_expanded:
            raise ValueError("Please call vocab_expansion first")

        minibatch = self.get_minibatch(sentences, tokenize=tokenize)
        h, h_t = self.encoder(
            input=minibatch['sentences'], lengths=minibatch['lengths'],
            return_all=True, pool=pool
        )
        h = h.index_select(0, minibatch['rev'])
        h_t = h_t.index_select(0, minibatch['rev'])
        if return_numpy:
            return h.data.cpu().numpy(), h_t.data.cpu().numpy()
        else:
            return h, h_t

    def get_representation_ordered(self, sentences, pool='last', tokenize=False, return_numpy=True):
        """
        Get model representations, returned ordered by sentence length.
        h.index_select(0, rev) to retrieve original order,
        returns [h,] h_t, lengths, rev
        """
        if not self.vocab_expanded:
            raise ValueError("Please call vocab_expansion first")

        minibatch = self.get_minibatch(sentences, tokenize=tokenize)
        h, h_t = self.encoder(
            input=minibatch['sentences'], lengths=minibatch['lengths'],
            return_all=True, pool=pool
        )
        # keep sentences ordered by sentence length
        if return_numpy:
            return h.data.cpu().numpy(), h_t.data.cpu().numpy(), minibatch['lengths'], minibatch['rev']
        else:
            return h, h_t, minibatch['lengths'], minibatch['rev']

    def get_representation_from_ordered(self, sentences, lengths, pool='last', return_numpy=False):
        if not self.vocab_expanded:
            raise ValueError("Please call vocab_expansion first")
        h, h_t = self.encoder(input=sentences, lengths=lengths, return_all=True, pool=pool)

        if return_numpy:
            return h.data.cpu().numpy(), h_t.data.cpu().numpy()
        else:
            return h, h_t


if __name__ == '__main__':
    # Sentences need to be lowercased.
    sentences = [
        'hello world .',
        'the quick brown fox jumped over the lazy dog .',
        'this is a sentence .'
    ]
    vocab = [
        'the', 'quick', 'brown', 'fox', 'jumped', 'over', 'lazy', 'dog',
        'hello', 'world', '.', 'this', 'is', 'a', 'sentence', '<s>',
        '</s>', '<pad>', '<unk>'
    ]
    # gensen = GenSen(
    #     model_folder='GenSenLargeBothSkip',
    #     filename_prefix_1='nli_large_bothskip',
    #     filename_prefix_2='nli_large_bothskip_parse',
    #     pretrained_emb='/data/milatmp1/subramas/embeddings/gigaword_300.h5'
    # )
    # gensen.vocab_expansion(vocab)
    # reps_h, reps_h_t = gensen.get_representation(sentences, return_all=True, pool='last', numpy=True)
    # # reps_h contains the hidden states for all words in all sentences (padded to the max length of sentences) (batch_size x seq_len x 4096)
    # # reps_h_t contains only the last hidden state for all sentences in the minibatch (batch_size x 4096)
    # print(reps_h.shape, reps_h_t.shape)

    gensen = GenSenSingle(
        model_folder='GenSen',
        filename_prefix='nli_large',
        pretrained_emb='embeddings/glove.840B.300d.h5'
    )
    gensen.vocab_expansion(vocab)
    reps_h, reps_h_t = gensen.get_representation(sentences, pool='last', return_numpy=True)
    # reps_h contains the hidden states for all words in all sentences (padded to the max length of sentences) (batch_size x seq_len x 2048)
    # reps_h_t contains only the last hidden state for all sentences in the minibatch (batch_size x 2048)
    print(reps_h.shape, reps_h_t.shape)
