import numpy as np
from nltk import ngrams
from collections import Counter


def get_best_beam(beams, normalization_alpha=0):
    best_index = 0
    best_score = -1e10
    for (i, beam) in enumerate(beams):
        normalized_score = beam.normalized_score(normalization_alpha)
        if normalized_score > best_score:
            best_index = i
            best_score = normalized_score
    return beams[best_index]


def n_gram_repeats(sequence, n):
    """
    Returns true if sequence contains twice the same n-gram
    :param sequence:
    :param n:
    :return:
    """
    counts = Counter(ngrams(sequence, n))
    if len(counts) == 0:
        return False
    if counts.most_common()[0][1] > 1:
        return True
    return False


class Beam(object):
    end_token = -1

    def __init__(self, sequence, likelihood, mentioned_movies=None):
        self.finished = False
        self.sequence = sequence
        self.likelihood = likelihood
        if mentioned_movies is not None:
            self.mentioned_movies = mentioned_movies.copy()
        else:
            self.mentioned_movies = set()

    def get_updated(self, token, probability):
        if Beam.end_token == -1:
            raise ValueError("Beam class end_token static variable was not set")
        updated_beam = Beam(self.sequence + [token], self.likelihood * probability, self.mentioned_movies)
        if token == Beam.end_token:
            updated_beam.finished = True
        return updated_beam

    def __str__(self):
        finished_str = "" if self.finished else "not"
        return finished_str + " finished beam of likelihood {} : {}".format(self.likelihood, self.sequence)

    def get_string(self, id2word, verbose=False, alpha=0):
        if verbose:
            finished_str = "" if self.finished else "not"
            return finished_str + " finished beam of score {} : {}".format(
                self.normalized_score(alpha), " ".join([id2word[x] for x in self.sequence]))
        else:
            return " ".join([id2word[x] for x in self.sequence])

    def normalized_score(self, alpha):
        """
        Get score with a length penalty following
        Wu et al 'Google's neural machine translation system: Bridging the gap between human and machine translation'
        :param alpha:
        :return:
        """
        if alpha == 0:
            return np.log(self.likelihood)
        else:
            penalty = ((5 + len(self.sequence)) / 6) ** alpha
            return np.log(self.likelihood) / penalty


class BeamSearch(object):
    def __init__(self, beam_size, start_sentence, end_token):
        self.beam_size = beam_size
        self.end_token = end_token
        Beam.end_token = end_token
        self.beams = [Beam(sequence=start_sentence, likelihood=1)]

    def search(self, probabilities, n_gram_block=None):
        """
        One step of beam search
        :param n_gram_block:
        :param probabilities: list of beam_size probability tensors (one for each beam)
        :return: list of the new beams.
        """
        vocab_size = probabilities[0].data.shape[0]
        # compute the likelihoods for the next token
        # vector for finished beams. First dimension will be the likelihood of the finished beam, other dimensions are
        # zeros so this beam is counted only once in the top k
        finsished_beam_vec = np.zeros(vocab_size)
        finsished_beam_vec[0] = 1
        # (beam_size, vocab_size)
        new_probabilities = np.array([beam.likelihood * probability.data.numpy() if not beam.finished
                                      else beam.likelihood * finsished_beam_vec
                                      for beam, probability in zip(self.beams, probabilities)])
        # get the top-k (beam_size) probabilities
        ind = np.unravel_index(np.argsort(new_probabilities, axis=None), new_probabilities.shape)
        # inspect hypothesis in descending order of likelihood
        ind = (ind[0][::-1], ind[1][::-1])
        # get the list of top-k updated beams
        new_beams = []
        for beam_index, token in zip(*ind):
            # if finished, append the beam as is
            if self.beams[beam_index].finished:
                new_beams.append(self.beams[beam_index])
            # otherwise, update the beam with the chosen token
            else:
                # check n_gram blocking. Note that n_gram blocking is not used to produce the results in the article
                if n_gram_block is None or not n_gram_repeats(self.beams[beam_index].sequence + [token], n_gram_block):
                    # add extended hypothesis to new_beam list
                    new_beams.append(
                        self.beams[beam_index].get_updated(token, probabilities[beam_index][token].data.numpy()))

            # return when beam_size valid beams found
            if len(new_beams) >= self.beam_size:
                self.beams = new_beams
                return self.beams
        self.beams = new_beams
        return self.beams
