import os
from tqdm import tqdm
import numpy as np
import csv
import torch
from torch.autograd import Variable
import random

import config


def load_movies(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        movies = {row[0]: row[1] for row in reader if row[0] != "movieId"}
    return movies


def load_movies_merged(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        id2index = {row[3]: int(row[0]) for row in reader if row[0] != "index"}
    return id2index


def process_rating(rating, ratings01=False):
    if ratings01:
        # return 1 for ratings >= 2.5, 0 for lower ratings (this gives 87% of liked on movielens-latest)
        # return 1 for ratings >= 2, 0 for lower ratings (this gives 94% of liked on movielens-latest)
        return float(rating) >= 2
    # return a rating between 0 and 1
    return (float(rating) - .5) / 4.5


def load_ratings(path, as_array=True):
    """
    One data example per user.
    :param path:
    :param as_array:
    :return: if as_array = False, return a dictionary {userId: {movieId: rating}}
    otherwise, return an array [{movieId: rating}] where each element corresponds to one user.
    """
    data = {}
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for userId, movieId, rating, timestamp in tqdm(reader):
            if userId != "userId":  # avoid the first row
                if userId not in data:
                    data[userId] = {}
                data[userId][movieId] = rating
    if as_array:
        return data.values()
    return data


class MlBatchLoader(object):
    """
    Loads Movielens data
    """

    def __init__(self,
                 batch_size,
                 movie_path=config.MOVIE_PATH,
                 data_path=config.ML_DATA_PATH,
                 train_path=config.ML_TRAIN_PATH,
                 valid_path=config.ML_VALID_PATH,
                 test_path=config.ML_TEST_PATH,
                 ratings01=False
                 ):
        self.batch_size = batch_size
        self.movie_path = movie_path
        self.batch_index = {"train": 0, "valid": 0, "test": 0}
        self.data_path = {
            "train": os.path.join(data_path, train_path),
            "valid": os.path.join(data_path, valid_path),
            "test": os.path.join(data_path, test_path)
        }
        self.ratings01 = ratings01

        self.load_data()

        self.n_batches = {key: len(val) // self.batch_size for key, val in self.ratings.items()}

    def load_data(self):
        # self.id2movies = load_movies(self.movie_path)
        # self.id2index = {id: i for (i, id) in enumerate(self.id2movies)}
        self.id2index = load_movies_merged(self.movie_path)
        self.n_movies = np.max(self.id2index.values()) + 1
        print("Loading movie ratings from {}".format(self.data_path))
        self.ratings = {subset: load_ratings(path, as_array=False)
                        for subset, path in self.data_path.items()}
        # list of userIds for each subset
        self.keys = {subset: ratings.keys() for subset, ratings in self.ratings.items()}

    def load_batch(self, subset="train", batch_input="full", max_num_inputs=None, ratings01=None):
        if batch_input == 'random_noise' and max_num_inputs is None:
            raise ValueError("batch_input set to random_noise, max_num_inputs should not be None")
        if ratings01 is None:
            ratings01 = self.ratings01
        # list of users for the batch
        batch_data = self.keys[subset][self.batch_index[subset] * self.batch_size:
                                       (self.batch_index[subset] + 1) * self.batch_size]

        self.batch_index[subset] = (self.batch_index[subset] + 1) % self.n_batches[subset]

        # As inputs: ratings for the same user in the training set
        # As targets: ratings for that user in the subset
        input = np.zeros((self.batch_size, self.n_movies))
        # unobserved ratings are -1 (those don't influence the loss)
        target = np.zeros((self.batch_size, self.n_movies)) - 1
        for (i, userId) in enumerate(batch_data):
            # Create input from training ratings
            if userId in self.ratings["train"]:
                train_ratings = self.ratings["train"][userId]
                if batch_input == 'random_noise':
                    # randomly chose a number of inputs to keep
                    max_nb_inputs = min(max_num_inputs, len(train_ratings) - 1)
                    n_inputs = random.randint(1, max(1, max_nb_inputs))
                    # randomly chose the movies that will be in the input
                    input_keys = random.sample(train_ratings.keys(), n_inputs)
                # Create input from training ratings
                for (movieId, rating) in train_ratings.items():
                    if batch_input == 'full' or (batch_input == 'random_noise' and movieId in input_keys):
                        # movie ratings in a [0.1, 1] range
                        input[i, self.id2index[movieId]] = process_rating(rating, ratings01=ratings01)
            # else:
            #     print("Warning user {} not in training set".format(userId))
            # Create targets
            for (movieId, rating) in self.ratings[subset][userId].items():
                target[i, self.id2index[movieId]] = process_rating(rating, ratings01=ratings01)
        input = Variable(torch.from_numpy(input).float())
        target = Variable(torch.from_numpy(target).float())
        return {"input": input, "target": target}
