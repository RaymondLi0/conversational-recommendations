import config
import os
import numpy as np
import re
from random import shuffle
import csv
import json
from tqdm import tqdm
import pickle
import random
from collections import Counter

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
    for line in open(path):
        data.append(json.loads(line))
    return data


def get_movies(path):
    id2name = {}
    db2id = {}
    with open(path, 'r') as f:
        reader = csv.reader(f)
        # remove date from movie name
        date_pattern = re.compile(r'\(\d{4}\)')
        for row in reader:
            if row[0] != "index":
                id2name[int(row[0])] = date_pattern.sub('', row[1])
                db2id[int(row[2])] = int(row[0])
    del db2id[-1]
    print("loaded {} movies from {}".format(len(id2name), path))
    return id2name, db2id


def process_ratings(rating, ratings01=True):
    if ratings01:
        return rating
    if rating == 0:
        return 0.1
    elif rating == 1:
        return 0.8
    else:
        raise ValueError("Expected rating = 0 or 1. Instaed, received {}".format(rating))


def get_cut_indices(movie_mentions, cut_width):
    """
    Get the utterance indices to cut the dialogue around the movie mentions
    At cut_width=0, return the index of the first mention, and the index of the last mention
    Higher cut_width adds utterances before the first mention, and after the last mention.
    :param movie_mentions:
    :param cut_width:
    :return:
    """
    utterance_mentions = [sum(utterance) > 0 for utterance in movie_mentions]
    first = next((i for (i, x) in enumerate(utterance_mentions) if x), 0)  # index of first occurrence
    last = next((i for (i, x) in enumerate(reversed(utterance_mentions)) if x), 0)  # reversed index of last occurrence
    last = len(utterance_mentions) - last
    return max(0, first - cut_width), min(len(utterance_mentions), last + cut_width)


class DialogueBatchLoader(object):
    def __init__(self, sources, batch_size,
                 conversation_length_limit=config.CONVERSATION_LENGTH_LIMIT,
                 utterance_length_limit=config.UTTERANCE_LENGTH_LIMIT,
                 training_size=-1,
                 data_path=config.REDIAL_DATA_PATH,
                 train_path=config.TRAIN_PATH,
                 valid_path=config.VALID_PATH,
                 test_path=config.TEST_PATH,
                 movie_path=config.MOVIE_PATH,
                 vocab_path=config.VOCAB_PATH,
                 shuffle_data=False,
                 process_at_instanciation=False):
        # sources paramater: string "dialogue/sentiment_analysis [movie_occurrences] [movieIds_in_target]"
        self.sources = sources
        self.batch_size = batch_size
        self.batch_index = {"train": 0, "valid": 0, "test": 0}
        self.conversation_length_limit = conversation_length_limit
        self.utterance_length_limit = utterance_length_limit
        self.training_size = training_size
        self.data_path = {"train": os.path.join(data_path, train_path),
                          "valid": os.path.join(data_path, valid_path),
                          "test": os.path.join(data_path, test_path)}
        self.movie_path = movie_path
        self.vocab_path = vocab_path
        self.word2id = None
        self.shuffle_data = shuffle_data
        # if true, call extract_dialogue when loading the data. (for training on several epochs)
        # Otherwise, call extract_dialogue at batch loading. (faster for testing the code)
        self.process_at_instanciation = process_at_instanciation

        self._get_dataset_characteristics()

    def _get_dataset_characteristics(self):
        # load movies. "db" refers to the movieId in the ReDial dataset, whereas "id" refers to the global movieId
        # (after matching the movies with movielens in match_movies.py).
        # So db2id is a dictionary mapping ReDial movie Ids to global movieIds
        self.id2name, self.db2id = get_movies(self.movie_path)
        self.db2name = {db: self.id2name[id] for db, id in self.db2id.items()}
        self.n_movies = len(self.db2id.values())  # number of movies mentioned in ReDial
        print('{} movies'.format(self.n_movies))
        # load data
        print("Loading and processing data")
        self.conversation_data = {key: load_data(val) for key, val in self.data_path.items()}
        if self.training_size > 0:
            self.conversation_data["train"] = self.conversation_data["train"][:self.training_size]
        if "sentiment_analysis" in self.sources:
            self.form_data = {key: self.extract_form_data(val) for key, val in self.conversation_data.items()}
        if "ratings" in self.sources:
            self.ratings_data = {key: self.extract_ratings_data(val) for key, val in self.conversation_data.items()}
            train_mean = np.mean([np.mean(conv.values()) for conv in self.ratings_data["train"]])
            print("Mean training rating ", train_mean)
            print("validation MSE made by mean estimator: {}".format(
                np.mean([np.mean((np.array(conv.values()) - train_mean) ** 2)
                         for conv in self.ratings_data["valid"]])))
        # load vocabulary
        self.train_vocabulary = self._get_vocabulary()
        print("Vocabulary size : {} words.".format(len(self.train_vocabulary)))

        if "dialogue" in self.sources:
            data = self.conversation_data
        elif "sentiment_analysis" in self.sources:
            data = self.form_data
        elif "ratings" in self.sources:
            data = self.ratings_data

        if self.shuffle_data:
            # shuffle each subset
            for _, val in data.items():
                shuffle(val)

        self.N_train_data = len(data["train"])
        self.n_batches = {key: len(val) // self.batch_size for key, val in data.items()}
        # print("{} train batches and {} valid batches".format(self.n_batches["train"], self.n_batches["valid"]))

    def extract_form_data(self, data):
        """
        get form data from data. For sentiment analysis.
        :param data:
        :return: form_data. Array where each element is of the form (MovieId, Movie Name, answers, conversation_index)
        """
        form_data = []
        for (i, conversation) in enumerate(data):
            init_q = conversation["initiatorQuestions"]
            resp_q = conversation["respondentQuestions"]
            # get movies that are in both forms. Do not take empty movie names
            gen = (key for key in init_q if key in resp_q and not self.db2name[int(key)].isspace())
            for key in gen:
                answers = [init_q[key]["suggested"],
                           init_q[key]["seen"],
                           init_q[key]["liked"],
                           resp_q[key]["suggested"],
                           resp_q[key]["seen"],
                           resp_q[key]["liked"]]
                form_data.append((self.db2id[int(key)], self.db2name[int(key)], answers, i))
        return form_data

    def extract_ratings_data(self, data):
        """
        Get ratings data from each conversation.
        :param data:
        :return: array of dictionaries {movieId: rating}. One dictionary corresponds to one conversation
        """
        ratings_data = []
        for (i, conversation) in enumerate(data):
            conv_ratings = {}
            gen = (key for key in conversation["initiatorQuestions"] if not self.db2name[int(key)].isspace())
            for dbId in gen:
                movieId = self.db2id[int(dbId)]
                liked = int(conversation["initiatorQuestions"][dbId]["liked"])
                # Only consider "disliked" or "liked", ignore "did not say"
                if liked in [0, 1]:
                    conv_ratings[movieId] = process_ratings(liked)
            # Do not append empty ratings
            if conv_ratings:
                ratings_data.append(conv_ratings)
        return ratings_data

    def _get_vocabulary(self):
        """
        get the vocabulary from the train data
        :return: vocabulary
        """
        if os.path.isfile(self.vocab_path):
            print("Loading vocabulary from {}".format(self.vocab_path))
            return pickle.load(open(self.vocab_path))
        print("Loading vocabulary from data")
        counter = Counter()
        # get vocabulary from dialogues
        for subset in ["train", "valid", "test"]:
            for conversation in tqdm(self.conversation_data[subset]):
                for message in conversation["messages"]:
                    # remove movie Ids
                    pattern = re.compile(r'@(\d+)')
                    text = tokenize(pattern.sub(" ", message["text"]))
                    counter.update([word.lower() for word in text])
        # get vocabulary from movie names
        for movieId in self.db2name:
            tokenized_movie = tokenize(self.db2name[movieId])
            counter.update([word.lower() for word in tokenized_movie])
        # Keep the most common words
        kept_vocab = counter.most_common(15000)
        vocab = [x[0] for x in kept_vocab]
        print("Vocab covers {} word instances over {}".format(
            sum([x[1] for x in kept_vocab]),
            sum([counter[x] for x in counter])
        ))
        vocab += ['<s>', '</s>', '<pad>', '<unk>', '\n']
        with open(self.vocab_path, 'w') as f:
            pickle.dump(vocab, f)
        print("Saved vocabulary in {}".format(self.vocab_path))
        return vocab

    def extract_dialogue(self, conversation, flatten_messages=True):
        """
        :param conversation: conversation dictionary. keys : 'conversationId', 'respondentQuestions', 'messages',
         'movieMentions', 'respondentWorkerId', 'initiatorWorkerId', 'initiatorQuestions'
         :param flatten_messages
         :return:
        """
        dialogue = []
        target = []
        senders = []
        occurrences = None
        if "movie_occurrences" in self.sources:
            # initialize occurrences. Ignore empty movie names
            occurrences = {self.db2id[int(dbId)]: [] for dbId in conversation["movieMentions"]
                           if int(dbId) in self.db2name and not self.db2name[int(dbId)].isspace()}
        for message in conversation["messages"]:
            # role of the sender of message: 1 for seeker, -1 for recommender
            role = 1 if message["senderWorkerId"] == conversation["initiatorWorkerId"] else -1
            # remove "@" and add spaces around movie mentions to be sure to count them as single tokens
            # tokens that match /^\d{5,6}$/ are movie mentions
            pattern = re.compile(r'@(\d+)')
            message_text = pattern.sub(lambda m: " " + m.group(1) + " ", message["text"])
            text = tokenize(message_text)

            if "movie_occurrences" in self.sources:
                text, message_target, message_occurrences = self.replace_movies_in_tokenized(text)
            else:
                text = self.replace_movies_in_tokenized(text)
                message_target = text

            # if flatten messages, append message when the sender is the same as in the last message
            if flatten_messages and len(senders) > 0 and senders[-1] == role:
                dialogue[-1] += ["\n"] + text
                target[-1] += ["\n"] + message_target
                if "movie_occurrences" in self.sources:
                    for movieId in occurrences:
                        if movieId in message_occurrences:
                            occurrences[movieId][-1] += [0] + message_occurrences[movieId]
                        else:
                            occurrences[movieId][-1] += [0] * (len(text) + 1)
            # otherwise finish the previous utterance and add the new utterance
            else:
                if len(senders) > 0:
                    dialogue[-1] += ['</s>']
                    target[-1] += ['</s>', '</s>']
                    if "movie_occurrences" in self.sources:
                        for movieId in occurrences:
                            occurrences[movieId][-1] += [0]
                senders.append(role)
                dialogue.append(['<s>'] + text)
                target.append(message_target)
                if "movie_occurrences" in self.sources:
                    for movieId in occurrences:
                        if movieId in message_occurrences:
                            occurrences[movieId].append([0] + message_occurrences[movieId])
                        else:
                            occurrences[movieId].append([0] * (len(text) + 1))
        # finish the last utterance
        dialogue[-1] += ['</s>']
        target[-1] += ['</s>', '</s>']
        if "movie_occurrences" in self.sources:
            for movieId in occurrences:
                occurrences[movieId][-1] += [0]
        dialogue, target, senders, occurrences = self.truncate(dialogue, target, senders, occurrences)
        if "movie_occurrences" in self.sources:
            return dialogue, target, senders, occurrences
        return dialogue, target, senders, None

    def replace_movies_in_tokenized(self, tokenized):
        """
        replace movieId tokens in a single tokenized message.
        Eventually compute the movie occurrences and the target with (global) movieIds
        :param tokenized:
        :return:
        """
        output_with_id = tokenized[:]
        occurrences = {}
        pattern = re.compile(r'^\d{5,6}$')
        index = 0
        while index < len(tokenized):
            word = tokenized[index]
            # Check if word corresponds to a movieId.
            if pattern.match(word) and int(word) in self.db2id and not self.db2name[int(word)].isspace():
                # get the global Id
                movieId = self.db2id[int(word)]
                # add movie to occurrence dict
                if movieId not in occurrences:
                    occurrences[movieId] = [0] * len(tokenized)
                # remove ID
                del tokenized[index]
                # put tokenized movie name instead. len(tokenized_movie) - 1 elements are added to tokenized.
                tokenized_movie = tokenize(self.id2name[movieId])
                tokenized[index:index] = tokenized_movie

                # update output_with_id: replace word with movieId repeated as many times as there are words in the
                # movie name. Add the size-of-vocabulary offset.
                output_with_id[index:index + 1] = [movieId + len(self.word2id)] * len(tokenized_movie)

                # update occurrences
                if "movie_occurrences" in self.sources:
                    # extend the lists
                    for otherIds in occurrences:
                        # the zeros in occurrence lists can be appended at the end since all elements after index are 0
                        # occurrences[otherIds][index:index] = [0] * (len(tokenized_movie) - 1)
                        occurrences[otherIds] += [0] * (len(tokenized_movie) - 1)
                    # update list corresponding to the occurring movie
                    occurrences[movieId][index:index + len(tokenized_movie)] = [1] * len(tokenized_movie)

                # increment index
                index += len(tokenized_movie)

            else:
                # do nothing, and go to next word
                index += 1
        if "movie_occurrences" in self.sources:
            if "movieIds_in_target" in self.sources:
                return tokenized, output_with_id, occurrences
            return tokenized, tokenized, occurrences
        return tokenized

    def truncate(self, dialogue, target, senders, movie_occurrences):
        # truncate conversations that have too many utterances
        if len(dialogue) > self.conversation_length_limit:
            dialogue = dialogue[:self.conversation_length_limit]
            target = target[:self.conversation_length_limit]
            senders = senders[:self.conversation_length_limit]
            if "movie_occurrences" in self.sources:
                movie_occurrences = {
                    key: val[:self.conversation_length_limit] for key, val in movie_occurrences.items()
                }
        # truncate utterances that are too long
        for (i, utterance) in enumerate(dialogue):
            if len(utterance) > self.utterance_length_limit:
                dialogue[i] = dialogue[i][:self.utterance_length_limit]
                target[i] = target[i][:self.utterance_length_limit]
                if "movie_occurrences" in self.sources:
                    for movieId, value in movie_occurrences.items():
                        value[i] = value[i][:self.utterance_length_limit]
        return dialogue, target, senders, movie_occurrences

    def set_word2id(self, word2id):
        self.word2id = word2id
        self.id2word = {id: word for (word, id) in self.word2id.items()}

        if self.process_at_instanciation:
            # pre-process dialogues
            self.conversation_data = {key: [self.extract_dialogue(conversation, flatten_messages=True)
                                            for conversation in val]
                                      for key, val in self.conversation_data.items()}

    def token2id(self, token):
        """
        :param token: string or movieId
        :return: corresponding ID
        """
        if token in self.word2id:
            return self.word2id[token]
        if isinstance(token, int):
            return token
        return self.word2id['<unk>']

    def _load_dialogue_batch(self, subset, flatten_messages):
        batch = {"senders": [], "dialogue": [], "lengths": [], "target": []}
        if "movie_occurrences" in self.sources:
            # movie occurrences: Array of dicts
            batch["movie_occurrences"] = []

        # get batch
        batch_data = self.conversation_data[subset][self.batch_index[subset] * self.batch_size:
                                                    (self.batch_index[subset] + 1) * self.batch_size]

        for i, conversation in enumerate(batch_data):
            if self.process_at_instanciation:
                dialogue, target, senders, movie_occurrences = conversation
            else:
                dialogue, target, senders, movie_occurrences = self.extract_dialogue(conversation,
                                                                                     flatten_messages=flatten_messages)
            batch["lengths"].append([len(message) for message in dialogue])
            batch["dialogue"].append(dialogue)
            batch["senders"].append(senders)
            batch["target"].append(target)
            if "movie_occurrences" in self.sources:
                batch["movie_occurrences"].append(movie_occurrences)

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
        if "movie_occurrences" in self.sources:
            batch["movie_occurrences"] = [
                {key: [utterance + [0] * (max_utterance_len - len(utterance)) for utterance in value] +
                      [[0] * max_utterance_len] * (max_conv_len - len(value)) for key, value in conv.items()}
                for conv in batch["movie_occurrences"]
            ]
        return batch

    def _load_sentiment_analysis_batch(self, subset, flatten_messages=True, cut_dialogues=-1):
        batch = {"senders": [], "dialogue": [], "lengths": [], "forms": [], "movieIds": []}
        # forms: answer to movie forms (batch_size, 6)
        if "movie_occurrences" in self.sources:
            # movie occurrences (batch_size, max_conv_length, max_utt_length)
            batch["movie_occurrences"] = []

        # get batch
        batch_data = self.form_data[subset][self.batch_index[subset] * self.batch_size:
                                            (self.batch_index[subset] + 1) * self.batch_size]

        for i, example in enumerate(batch_data):
            # (movieId, movieName, answers, conversationIndex) = example
            conversation = self.conversation_data[subset][example[3]]
            if self.process_at_instanciation:
                dialogue, target, senders, movie_occurrences = conversation
            else:
                dialogue, target, senders, movie_occurrences = self.extract_dialogue(conversation,
                                                                                     flatten_messages=flatten_messages)
            if "movie_occurrences" in self.sources:
                movie_occurrences = movie_occurrences[example[0]]
                if cut_dialogues == "random":
                    cut_width = random.randint(1, len(dialogue))
                else:
                    cut_width = cut_dialogues
                # cut dialogues around occurrence
                if cut_width >= 0:
                    start_index, end_index = get_cut_indices(movie_occurrences, cut_width)
                    dialogue = dialogue[start_index:end_index]
                    senders = senders[start_index:end_index]
                    movie_occurrences = movie_occurrences[start_index:end_index]
            batch['movieIds'].append(example[0])
            batch["lengths"].append([len(message) for message in dialogue])
            batch["dialogue"].append(dialogue)
            batch["senders"].append(senders)
            batch["forms"].append(example[2])
            if "movie_occurrences" in self.sources:
                batch["movie_occurrences"].append(movie_occurrences)

        max_utterance_len = max([max(x) for x in batch["lengths"]])
        max_conv_len = max([len(conv) for conv in batch["dialogue"]])
        batch["conversation_lengths"] = np.array([len(x) for x in batch["lengths"]])
        # replace text with ids and pad sentences
        batch["lengths"] = np.array(
            [lengths + [0] * (max_conv_len - len(lengths)) for lengths in batch["lengths"]]
        )
        batch["dialogue"] = Variable(torch.LongTensor(
            self.text_to_ids(batch["dialogue"], max_utterance_len, max_conv_len)))  # (batch, conv_len, utt_len)
        batch["senders"] = Variable(torch.FloatTensor(
            [senders + [0] * (max_conv_len - len(senders)) for senders in batch["senders"]]))
        batch["forms"] = Variable(torch.LongTensor(batch["forms"]))  # (batch, 6)
        if "movie_occurrences" in self.sources:
            batch["movie_occurrences"] = Variable(torch.FloatTensor(
                [[utterance + [0] * (max_utterance_len - len(utterance)) for utterance in conv] +
                 [[0] * max_utterance_len] * (max_conv_len - len(conv)) for conv in batch["movie_occurrences"]]
            ))
        return batch

    def _load_ratings_batch(self, subset, batch_input, max_num_inputs=None):
        if batch_input == 'random_noise' and max_num_inputs is None:
            raise ValueError("batch_input set to random_noise, max_num_inputs should not be None")
        # One element in batch_data corresponds to one conversation
        batch_data = self.ratings_data[subset][self.batch_index[subset] * self.batch_size:
                                               (self.batch_index[subset] + 1) * self.batch_size]
        # WARNING : loading differs from the one used in autorec_batch_loader
        if batch_input == "full":
            if subset == "train":
                target = np.zeros((self.batch_size, self.n_movies)) - 1
                input = np.zeros((self.batch_size, self.n_movies))
                for (i, ratings) in enumerate(batch_data):
                    for (movieId, rating) in ratings.items():
                        input[i, movieId] = rating
                        target[i, movieId] = rating
            # if not training, for all ratings of conversation,
            # use this rating as target, and use all other ratings as inputs
            # so batch has as many examples as there are movie mentions in the `batch_size` conversations
            else:
                input = []
                target = []
                for ratings in batch_data:
                    complete_input = [0] * self.n_movies
                    # populate input with ratings
                    for movieId, rating in ratings.items():
                        complete_input[movieId] = rating
                    for movieId, rating in ratings.items():
                        # for each movie, zero out in the input and put target rating
                        input_tmp = complete_input[:]
                        input_tmp[movieId] = 0
                        target_tmp = [-1] * self.n_movies
                        target_tmp[movieId] = rating
                        input.append(input_tmp)
                        target.append(target_tmp)
                input = np.array(input)
                target = np.array(target)
            input = Variable(torch.from_numpy(input).float())
            target = Variable(torch.from_numpy(target).float())
            return {"input": input, "target": target}
        # take random inputs
        elif batch_input == "random_noise":
            input = np.zeros((self.batch_size, self.n_movies))
            target = np.zeros((self.batch_size, self.n_movies)) - 1
            for (i, ratings) in enumerate(batch_data):
                # randomly chose a number of inputs to keep
                max_nb_inputs = min(max_num_inputs, len(ratings) - 1)
                n_inputs = random.randint(1, max(1, max_nb_inputs))
                # randomly chose the movies that will be in the input
                input_keys = random.sample(ratings.keys(), n_inputs)
                # Create input
                for (movieId, rating) in ratings.items():
                    if movieId in input_keys:
                        input[i, movieId] = rating
                # Create target
                for (movieId, rating) in ratings.items():
                    target[i, movieId] = rating
            return {"input": Variable(torch.from_numpy(input).float()),
                    "target": Variable(torch.from_numpy(target).float())}

    def load_batch(self, subset="train",
                   flatten_messages=True, batch_input="random_noise", cut_dialogues=-1, max_num_inputs=None):
        """
        Get next batch
        :param batch_input:
        :param cut_dialogues:
        :param max_num_inputs:
        :param subset: "train", "valid" or "test"
        :param flatten_messages: if False, load the conversation messages as they are. If True, concatenate consecutive
        messages from the same sender and put a "\n" between consecutive messages.
        :return: batch
        """
        if "dialogue" in self.sources:
            if self.word2id is None:
                raise ValueError("word2id is not set, cannot load batch")
            batch = self._load_dialogue_batch(subset, flatten_messages)
        elif "sentiment_analysis" in self.sources:
            if self.word2id is None:
                raise ValueError("word2id is not set, cannot load batch")
            batch = self._load_sentiment_analysis_batch(subset, flatten_messages, cut_dialogues=cut_dialogues)
        elif "ratings" in self.sources:
            batch = self._load_ratings_batch(subset, batch_input=batch_input, max_num_inputs=max_num_inputs)

        self.batch_index[subset] = (self.batch_index[subset] + 1) % self.n_batches[subset]

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
