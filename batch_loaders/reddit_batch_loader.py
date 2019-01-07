import config
import os
import time
import numpy as np
from random import shuffle
import pickle

import torch
from torch.autograd import Variable

from utils import tokenize

import sys

reload(sys)
sys.setdefaultencoding('utf8')


def load_data(path):
    """

    :param path:
    :return:
    """
    data = []
    with open(path, "r") as f:
        for line in f.readlines():
            if line == "\n":
                continue
            if int(line[0]) == 1:
                data.append(line[2:].split("\t"))
            else:
                data[-1] += line[2:].split("\t")
    return data


class RedditBatchLoader(object):
    def __init__(
            self, batch_size,
            conversation_length_limit=config.CONVERSATION_LENGTH_LIMIT,
            utterance_length_limit=config.UTTERANCE_LENGTH_LIMIT,
            data_path=config.REDDIT_PATH,
            train_path=config.REDDIT_TRAIN_PATH,
            valid_path=config.REDDIT_VALID_PATH,
            test_path=config.REDDIT_TEST_PATH,
            vocab_path=config.VOCAB_PATH,
            shuffle_data=False,
    ):
        self.batch_size = batch_size
        self.batch_index = {"train": 0, "valid": 0, "test": 0}
        self.conversation_length_limit = conversation_length_limit
        self.utterance_length_limit = utterance_length_limit
        self.data_path = {"train": os.path.join(data_path, train_path),
                          "valid": os.path.join(data_path, valid_path),
                          "test": os.path.join(data_path, test_path)}
        self.vocab_path = vocab_path
        self.word2id = None
        self.shuffle_data = shuffle_data

        self._get_dataset_characteristics()

    def _get_dataset_characteristics(self):
        # load data
        print("Loading and processing reddit data")
        self.conversation_data = {key: load_data(val) for key, val in self.data_path.items()}
        # load vocabulary
        self.train_vocabulary = pickle.load(open(self.vocab_path))
        print("Vocabulary size : {} words.".format(len(self.train_vocabulary)))
        if self.shuffle_data:
            # shuffle each subset
            for _, val in self.conversation_data.items():
                shuffle(val)

        self.N_train_data = len(self.conversation_data["train"])
        self.n_batches = {key: len(val) // self.batch_size for key, val in self.conversation_data.items()}

    def token2id(self, token):
        """
        :param token: string or movieId
        :return: corresponding ID
        """
        if token in self.word2id:
            return self.word2id[token]
        return self.word2id['<unk>']

    def extract_dialogue(self, conversation):
        dialogue = []
        target = []
        for message in conversation:
            message = tokenize(message)
            dialogue.append(['<s>'] + message + ['</s>'])
            target.append(message + ['</s>', '</s>'])
        dialogue, target = self.truncate(dialogue, target)
        senders = [-1, 1] * 100
        senders = senders[:len(dialogue)]
        return dialogue, target, senders

    def truncate(self, dialogue, target):
        # truncate conversations that have too many utterances
        if len(dialogue) > self.conversation_length_limit:
            dialogue = dialogue[:self.conversation_length_limit]
            target = target[:self.conversation_length_limit]
        # truncate utterances that are too long
        for (i, utterance) in enumerate(dialogue):
            if len(utterance) > self.utterance_length_limit:
                dialogue[i] = dialogue[i][:self.utterance_length_limit]
                target[i] = target[i][:self.utterance_length_limit]
        return dialogue, target

    def set_word2id(self, word2id):
        self.word2id = word2id
        self.id2word = {id: word for (word, id) in self.word2id.items()}

    def _load_dialogue_batch(self, subset):
        batch = {"senders": [], "dialogue": [], "lengths": [], "target": []}
        # sender of message. 1 or -1
        # utterance lengths
        # get batch
        batch_data = self.conversation_data[subset][self.batch_index[subset] * self.batch_size:
                                                    (self.batch_index[subset] + 1) * self.batch_size]

        for i, conversation in enumerate(batch_data):
            dialogue, target, senders = self.extract_dialogue(conversation)

            batch["lengths"].append([len(message) for message in dialogue])
            batch["dialogue"].append(dialogue)
            batch["senders"].append(senders)
            batch["target"].append(target)

        max_utterance_len = max([max(x) for x in batch["lengths"]])
        max_conv_len = max([len(conv) for conv in batch["dialogue"]])
        batch["conversation_lengths"] = np.array([len(x) for x in batch["lengths"]])
        # replace text with ids and pad sentences
        batch["lengths"] = np.array(
            [lengths + [0] * (max_conv_len - len(lengths)) for lengths in batch["lengths"]]
        )
        batch["dialogue"] = Variable(torch.LongTensor(
            self.text_to_ids(batch["dialogue"], max_utterance_len, max_conv_len)))
        batch["target"] = Variable(torch.LongTensor(
            self.text_to_ids(batch["target"], max_utterance_len, max_conv_len)))
        batch["senders"] = Variable(torch.FloatTensor(
            [senders + [0] * (max_conv_len - len(senders)) for senders in batch["senders"]]))
        batch["movie_occurrences"] = {}
        return batch

    def load_batch(self, subset="train", verbose=False,):
        start_time = time.time()
        if verbose:
            print("loading {} batch index {}".format(subset, self.batch_index[subset]))
        batch = self._load_dialogue_batch(subset)
        self.batch_index[subset] = (self.batch_index[subset] + 1) % self.n_batches[subset]
        if verbose:
            print("batch loaded in : ", time.time() - start_time)
        return batch

    def text_to_ids(self, dialogue, max_utterance_len, max_conv_len):
        """
        replace with corresponding ids.
        Pad each utterance to max_utterance_len. And pad each conversation to max_conv_length
        :param dialogue: [[[word1, word2, ...]]]
        :param max_utterance_len:
        :param max_conv_len:
        :return: padded dialogue
        """
        dialogue = [[[self.token2id(w) for w in utterance] +
                     [self.word2id['<pad>']] * (max_utterance_len - len(utterance)) for utterance in conv] +
                    [[self.word2id['<pad>']] * max_utterance_len] * (max_conv_len - len(conv)) for conv in dialogue]
        return dialogue
