import torch
from torch.autograd import Variable

from beam_search import BeamSearch
from utils import tokenize


def replace_movie_with_words(token_id, movie_id2name, word2id):
    """
    If the ID corresponds to a movie, returns the sequence of tokens that correspond to this movie name
    :param token_id:
    :param word2id:
    :return: modified sequence
    """
    if token_id <= len(word2id):
        return token_id
    # retrieve the name of the movie associated with this ID. Substract the size-of-vocabulary offset
    tokenized_movie = tokenize(movie_id2name[token_id - len(word2id)])
    return [word2id[w] if w in word2id else word2id["<unk>"] for w in tokenized_movie]


class SequenceGenerator(object):
    """
    Generate sequences
    """

    def __init__(self, model,
                 beam_size,
                 word2id,
                 movie_id2name,
                 max_sequence_length=50,
                 n_gram_block=None):
        self.model = model
        self.beam_size = beam_size
        self.word2id = word2id
        self.movieId2name = movie_id2name
        self.max_sequence_length = max_sequence_length
        self.n_gram_block = n_gram_block

    def beam_search(self, initial_sequence, forbid_movies=None, temperature=1, **kwargs):
        """
        Beam search sentence generation
        :param initial_sequence: list giving the initial sequence of tokens
        :param kwargs: additional parameters to pass to model forward pass (e.g. a conditioning context)
        :return:
        """
        beam_search = BeamSearch(self.beam_size, initial_sequence, self.word2id["</s>"])
        beams = beam_search.beams
        for i in range(self.max_sequence_length):
            # compute probabilities for each beam
            probabilities = []
            for beam in beams:
                # add batch_dimension
                model_input = Variable(torch.LongTensor(beam.sequence)).unsqueeze(0)
                if self.model.cuda_available:
                    model_input = model_input.cuda()
                beam_forbidden_movies = forbid_movies.union(beam.mentioned_movies)
                prob = self.model(
                    input=model_input,
                    lengths=[len(beam.sequence)],
                    log_probabilities=False,
                    forbid_movies=beam_forbidden_movies,
                    temperature=temperature,
                    **kwargs
                )
                # get probabilities for the next token to generate
                probabilities.append(prob[0, -1, :].cpu())
            # update beams
            beams = beam_search.search(probabilities, n_gram_block=self.n_gram_block)
            # replace movie names with the corresponding words
            for beam in beams:
                if beam.sequence[-1] > len(self.word2id):
                    # update the list of movies mentioned for preventing repeated recommendations
                    beam.mentioned_movies.add(beam.sequence[-1] - len(self.word2id))
                    beam.sequence[-1:] = replace_movie_with_words(beam.sequence[-1], self.movieId2name, self.word2id)
        return beams

