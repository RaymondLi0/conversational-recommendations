import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DecoderGRU(nn.Module):
    """
    Conditioned GRU. The context vector is used as an initial hidden state at each layer of the GRU
    """

    def __init__(self,
                 hidden_size,
                 context_size,
                 num_layers,
                 vocab_size,
                 peephole,
                 embedding_dim=512):
        super(DecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.num_layers = num_layers
        # peephole: concatenate the context to the input at every time step
        self.peephole = peephole
        if not peephole and context_size != hidden_size:
            raise ValueError("peephole=False: the context size {} must match the hidden size {} in DecoderGRU".format(
                context_size, hidden_size))
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )
        self.gru = nn.GRU(
            input_size=embedding_dim + context_size * self.peephole,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

    def set_pretrained_embeddings(self, embedding_matrix):
        """Set embedding weights."""
        self.embedding.weight.data.set_(embedding_matrix)

    def forward(self, input_sequence, lengths, context=None, state=None):
        """
        If not peephole, use the context vector as initial hidden state at each layer.
        If peephole, concatenate context to embeddings at each time step instead.
        If context is not provided, assume that a state is given (for generation)
        :param state:
        :param input_sequence: (batch_size, seq_len)
        :param lengths: (batch_size)
        :param context: (batch, hidden_size) vector on which to condition
        :return: ouptut predictions (batch_size, seq_len, hidden_size) [, h_n (batch, num_layers, hidden_size)]
        """
        embedded = self.embedding(input_sequence)
        if context is not None:
            batch_size, context_size = context.data.shape
            seq_len = input_sequence.data.shape[1]
            if self.peephole:
                context_for_input = context.unsqueeze(1).expand(batch_size, seq_len, context_size)
                embedded = torch.cat((embedded, context_for_input), dim=2)
            packed = pack_padded_sequence(embedded, lengths, batch_first=True)

            if not self.peephole:
                # No peephole. Use context as initial hidden state
                # expand to the number of layers in the decoder
                context = context.unsqueeze(0).expand(
                    self.num_layers, batch_size, self.hidden_size).contiguous()

                output, _ = self.gru(packed, context)
            else:
                output, _ = self.gru(packed)
            return pad_packed_sequence(output, batch_first=True)[0]
        elif state is not None:
            output, h_n = self.gru(embedded, state)
            return output, h_n
        else:
            raise ValueError("Must provide at least state or context")


class TextDecoder(nn.Module):
    """
    Regular decoder. Add a fc layer on top of the DecoderGRU to predict the output (used in HRED for example)
    """

    def __init__(self,
                 hidden_size,
                 context_size,
                 num_layers,
                 vocab_size,
                 peephole,
                 embedding_dim=512):
        super(TextDecoder, self).__init__()
        self.num_layers = num_layers
        self.peephole = peephole
        self.decoder = DecoderGRU(hidden_size=hidden_size,
                                  context_size=context_size,
                                  num_layers=num_layers,
                                  vocab_size=vocab_size,
                                  embedding_dim=embedding_dim,
                                  peephole=peephole)
        self.out = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.cuda_available = torch.cuda.is_available()

    def set_pretrained_embeddings(self, embedding_matrix):
        """Set embedding weights."""
        self.decoder.set_pretrained_embeddings(embedding_matrix)

    def forward(self, input, lengths, context, log_probabilities):
        """

        :param log_probabilities:
        :param input: (batch, max_utterance_length)
        :param lengths:
        :param context: (batch, hidden_size)
        :return:
        """
        decoded = self.decoder(input, lengths, context=context)
        output = self.out(decoded)  # (batch, seq_len, vocab_size)
        if log_probabilities:
            return F.log_softmax(output.transpose(0, 2)).transpose(0, 2)
        else:
            return F.softmax(output.transpose(0, 2)).transpose(0, 2)


class SwitchingDecoder(nn.Module):
    """
    Decoder that takes the recommendations into account.
    A switch choses whether to output a movie or a word
    """

    def __init__(self,
                 hidden_size,
                 context_size,
                 num_layers,
                 vocab_size,
                 peephole,
                 ):
        super(SwitchingDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.num_layers = num_layers
        self.decoder = DecoderGRU(
            hidden_size=hidden_size,
            context_size=context_size,
            num_layers=num_layers,
            vocab_size=vocab_size,
            peephole=peephole
        )
        self.language_out = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.switch = nn.Linear(in_features=hidden_size + context_size, out_features=1)
        self.peephole = peephole
        self.cuda_available = torch.cuda.is_available()

    def set_pretrained_embeddings(self, embedding_matrix):
        """Set embedding weights."""
        self.decoder.set_pretrained_embeddings(embedding_matrix)

    def forward(self, input, lengths, context, movie_recommendations, log_probabilities, sample_movies,
                forbid_movies=None, temperature=1):
        """

        :param log_probabilities:
        :param temperature:
        :param input: (batch, max_utterance_length)
        :param lengths:
        :param context: (batch, hidden_size)
        :param movie_recommendations: (batch, n_movies) the movie recommendations that condition the utterances.
        Not necessarily in [0,1] range
        :param sample_movies: (for generation) If true, sample a movie for each utterance, returning one-hot vectors
        :param forbid_movies: (for generation) If provided, specifies movies that cannot be sampled
        :return: [log] probabilities (batch, max_utterance_length, vocab + n_movies)
        """
        batch_size, max_utterance_length = input.data.shape[:2]
        n_movies = movie_recommendations.data.shape[1]
        # Run language decoder
        # (batch, seq_len, hidden_size)
        hidden = self.decoder(input, lengths, context=context)
        # (batch, seq_len, vocab_size)
        language_output = self.language_out(hidden)
        # used in sampling
        max_probabilities, _ = torch.max(F.softmax(movie_recommendations, dim=1), dim=1)
        # expand context and movie_recommendations to each time step
        context = context.unsqueeze(1).expand(
            batch_size, max_utterance_length, self.context_size).contiguous()
        movie_recommendations = movie_recommendations.unsqueeze(1).expand(
            batch_size, max_utterance_length, n_movies).contiguous()

        # Compute Switch
        # (batch, seq_len, 2 * hidden_size)
        switch_input = torch.cat((context, hidden), dim=2)
        switch = self.switch(switch_input)

        # For generation: sample movies
        # (batch, seq_len, vocab_size + n_movies)
        if sample_movies:
            # Prepare probability vector for sampling
            movie_probabilities = F.softmax(movie_recommendations.view(-1, n_movies), dim=1)
            # zero out the forbidden movies
            if forbid_movies is not None:
                if batch_size > 1:
                    raise ValueError("forbid_movies functionality only implemented with batch_size=1 for now")
                for movieId in forbid_movies:
                    movie_probabilities[:, movieId] = 0
            # Sample movies
            sampled_movies = movie_probabilities.multinomial(1).view(
                batch_size, max_utterance_length).data.cpu().numpy()
            # Fill a new recommendations vector with sampled movies
            movie_recommendations = Variable(torch.zeros(batch_size, max_utterance_length, n_movies))
            for i in range(batch_size):
                for j in range(max_utterance_length):
                    # compensate bias towards sampled movie by putting the maximum probability of a movie instead of 1
                    movie_recommendations[i, j, sampled_movies[i, j]] = max_probabilities[i]
            if self.cuda_available:
                movie_recommendations = movie_recommendations.cuda()
            if log_probabilities:
                raise ValueError("Sample movies only works with log_probabilities=False for now.")
                # output = torch.cat((
                #     F.logsigmoid(switch) + F.log_softmax(language_output / temperature, dim=2),
                #     F.logsigmoid(-switch) + torch.log(movie_recommendations)
                # ), dim=2)
            else:
                output = torch.cat((
                    switch.sigmoid() * F.softmax(language_output / temperature, dim=2),
                    (-switch).sigmoid() * movie_recommendations
                ), dim=2)
            return output
        if log_probabilities:
            output = torch.cat((
                F.logsigmoid(switch) + F.log_softmax(language_output / temperature, dim=2),
                F.logsigmoid(-switch) + F.log_softmax(movie_recommendations / temperature, dim=2)
            ), dim=2)
        else:
            output = torch.cat((
                switch.sigmoid() * F.softmax(language_output / temperature, dim=2),
                (-switch).sigmoid() * F.softmax(movie_recommendations / temperature, dim=2)
            ), dim=2)
        return output

    def generate(self, input, state, context, movie_recommendations, log_probabilities, temperature=1):
        """
        One time step in generation. Use state if not None, otherwise use context to condition.
        :param log_probabilities:
        :param input: (batch)
        :param state: ()
        :param context: (batch, hidden_size)
        :param movie_recommendations: (batch, n_movies)
        :param temperature:
        :return: [log] probabilities (batch, vocab + n_movies), state ()
        """
        batch_size = input.data.shape[0]
        # add seq_len dimension (=1 in our case)
        input = input.unsqueeze(1)
        movie_recommendations = movie_recommendations.unsqueeze(1)
        # expand to the number of layers in the decoder
        context_expanded = context.unsqueeze(0).expand(
            self.num_layers, batch_size, self.hidden_size).contiguous()
        output, state = self.decoder(input, lengths=None, state=context_expanded if state is None else state)
        language_output = self.language_out(output)

        # (batch, 2*hidden_size)
        switch_input = torch.cat((context, output[:, -1, :]), dim=1)
        switch = self.switch(switch_input)
        if log_probabilities:
            output = torch.cat((
                F.logsigmoid(switch) + F.log_softmax(language_output.squeeze() / temperature),
                F.logsigmoid(-switch) + F.log_softmax(movie_recommendations.squeeze() / temperature)
            ), dim=1)
        else:
            output = torch.cat((
                F.sigmoid(switch) * F.softmax(language_output.squeeze() / temperature),
                F.sigmoid(-switch) * F.softmax(movie_recommendations.squeeze() / temperature)
            ), dim=1)
            return output, state
        return output[:, -1, :], state
