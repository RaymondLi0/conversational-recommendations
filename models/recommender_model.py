import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import os


from decoders import SwitchingDecoder
from utils import sort_for_packed_sequence

import config
from hierarchical_rnn import HRNN
from sentiment_analysis import SentimentAnalysis
from autorec import AutoRec
from gensen import GenSenSingle


class RecommendFromDialogue(nn.Module):
    """
    Recommender system that takes a dialogue as input. Runs sentiment analysis on all mentioned movies,
    then uses autorec to provide movie recommendations at each stage of the conversation
    """
    def __init__(self,
                 train_vocab,
                 n_movies,
                 params,
                 autorec_path=os.path.join(config.AUTOREC_MODEL, "model_best"),
                 sentiment_analysis_path=os.path.join(config.SENTIMENT_ANALYSIS_MODEL, "model_best"),
                 cuda=None,
                 gensen=True, ):
        super(RecommendFromDialogue, self).__init__()
        self.n_movies = n_movies
        if cuda is None:
            self.cuda_available = torch.cuda.is_available()
        else:
            self.cuda_available = cuda

        self.sentiment_analysis = SentimentAnalysis(
            params=params['sentiment_analysis_params'],
            train_vocab=train_vocab,
            gensen=gensen,
            resume=sentiment_analysis_path
        )
        self.autorec = AutoRec(
            params=params['autorec_params'],
            n_movies=self.n_movies,
            resume=autorec_path
        )

        # freeze sentiment analysis
        for param in self.sentiment_analysis.parameters():
            param.requires_grad = False

    def forward(
            self,
            dialogue,
            senders,
            lengths,
            conversation_lengths,
            movie_occurrences,
            recommend_new_movies,
            user_representation=None
    ):
        """

        :param dialogue: (batch, max_conv_length, max_utt_length) Variable containing the dialogue
        :param movie_occurrences: Array where each element corresponds to a conversation and is a dictionary
        {movieId: (max_conv_length, max_utt_length) array containing the movie mentions}
        :param recommend_new_movies: If true, zero out the movies already mentioned in the output
        :param user_representation: optional prior user representation (obtained from language)
        :return: (batch_size, max_conv_length, n_movies_total) movie preferences
        (ratings not necessarily between 0 and 1)
        """
        batch_size, max_conv_length = dialogue.data.shape[:2]
        if not movie_occurrences or len(movie_occurrences) == 0:
            tt = torch.cuda.FloatTensor if self.cuda_available else torch.FloatTensor
            return Variable(torch.zeros(batch_size, max_conv_length, self.n_movies, out=tt()))
        # indices to retrieve original order
        indices = [(i, movieId)
                   for (i, conv_movie_occurrences) in enumerate(movie_occurrences)
                   for movieId in conv_movie_occurrences]
        batch_indices = Variable(torch.LongTensor([i[0] for i in indices]))

        # flatten movie occurrences to shape (total_num_mentions_in_batch, max_conv_length, max_utt_length)
        flattened_movie_occurrences = [conv_movie_occurrences[movieId]
                                       for conv_movie_occurrences in movie_occurrences
                                       for movieId in conv_movie_occurrences]
        flattened_movie_occurrences = Variable(torch.FloatTensor(flattened_movie_occurrences))
        if self.cuda_available:
            batch_indices = batch_indices.cuda()
            flattened_movie_occurrences = flattened_movie_occurrences.cuda()

        # select the dialogues following the movie mentions
        dialogue = torch.index_select(dialogue, 0, batch_indices)
        senders = torch.index_select(senders, 0, batch_indices)
        lengths = lengths[[i[0] for i in indices]]
        conversation_lengths = conversation_lengths[[i[0] for i in indices]]
        # print("senders shape", senders.data.shape) # (total_num_mentions, max_conv_length)

        sentiment_analysis_input = {"dialogue": dialogue,
                                    "movie_occurrences": flattened_movie_occurrences,
                                    "senders": senders,
                                    "lengths": lengths,
                                    "conversation_lengths": conversation_lengths}

        # (total_num_mentions_in_batch, max_conv_length)
        movie_likes = self.sentiment_analysis(sentiment_analysis_input, return_liked_probability=True)

        # populate ratings input using the movie likes
        # (batch_size, max_conv_length, n_movies_total)
        tt = torch.cuda.FloatTensor if self.cuda_available else torch.FloatTensor
        autorec_input = Variable(torch.zeros(batch_size, max_conv_length, self.n_movies, out=tt()))
        # mask that tells from which utterance the movies have appeared in conversation
        mask = flattened_movie_occurrences.sum(dim=2) > 0
        mask = mask.cumsum(dim=1) > 0  # (total_num_mentions_in_batch, max_conv_length)
        # only use movie preferences after movies are mentioned
        movie_likes = movie_likes * mask.float()
        # print("movie likes shape", movie_likes.data.shape) # (total_num_mentions_in_batch, max_conv_length)
        for i, (batchId, movieId) in enumerate(indices):
            autorec_input[batchId, :, movieId] = movie_likes[i]

        # run recommendation model
        # (batch_size, max_conv_length, n_movies_total)
        output = self.autorec(autorec_input, additional_context=user_representation, range01=False)
        # use this at generation time: lower probability for movies already mentioned
        if recommend_new_movies:
            for batchId, movieId in indices:
                # (max_conv_length) mask that zeros out once the movie has been mentioned
                mask = np.sum(movie_occurrences[batchId][movieId], axis=1) > 0
                mask = Variable(torch.from_numpy((mask.cumsum(axis=0) == 0).astype(float))).float()
                if self.cuda_available:
                    mask = mask.cuda()
                output[batchId, :, movieId] = mask * output[batchId, :, movieId]
        return output


class Recommender(nn.Module):
    def __init__(self,
                 train_vocab,
                 n_movies,
                 params,
                 ):
        super(Recommender, self).__init__()
        self.params = params
        self.train_vocab = train_vocab
        self.n_movies = n_movies
        self.cuda_available = torch.cuda.is_available()

        # instantiate the gensen module that will be used in the encoder HRNN, and by the recommender module
        self.gensen = GenSenSingle(
            model_folder=os.path.join(config.MODELS_PATH, 'GenSen'),
            filename_prefix='nli_large',
            pretrained_emb=os.path.join(config.MODELS_PATH, 'embeddings/glove.840B.300d.h5'),
            cuda=self.cuda_available
        )
        self.gensen.vocab_expansion(list(train_vocab))

        # HRNN encoder
        # Conversation encoder not bidirectional
        self.encoder = HRNN(params=params['hrnn_params'],
                            train_vocabulary=train_vocab,
                            gensen=self.gensen,
                            train_gensen=False,
                            conv_bidirectional=False)
        self.recommender_module = RecommendFromDialogue(
            params=params['recommend_from_dialogue_params'],
            train_vocab=train_vocab,
            n_movies=n_movies,
            gensen=self.gensen,
        )

        if params['language_aware_recommender']:
            self.language_to_user = nn.Linear(in_features=params['hrnn_params']['conversation_encoder_hidden_size'],
                                              out_features=self.recommender_module.autorec.user_representation_size)
        # latent variable distribution parameters:
        latent_layer_sizes = params['latent_layer_sizes']
        if latent_layer_sizes is not None:
            latent_variable_size = latent_layer_sizes[-1]
            self.prior_hidden_layers = nn.ModuleList(
                [nn.Linear(in_features=params['hrnn_params']['conversation_encoder_hidden_size'],
                           out_features=latent_layer_sizes[0]) if i == 0
                 else nn.Linear(in_features=latent_layer_sizes[i - 1], out_features=latent_layer_sizes[i])
                 for i in range(len(latent_layer_sizes) - 1)])
            penultimate_size = params['hrnn_params']['conversation_encoder_hidden_size'] \
                if len(latent_layer_sizes) == 1 else latent_layer_sizes[-2]
            self.mu_prior = nn.Linear(penultimate_size, latent_variable_size)
            self.sigma_prior = nn.Linear(penultimate_size, latent_variable_size)

            # context size + size of sentence representations
            posterior_input_size = params['hrnn_params']['conversation_encoder_hidden_size'] +\
                                   2 * params['hrnn_params']['sentence_encoder_hidden_size'] + 1
            self.posterior_hidden_layers = nn.ModuleList(
                [nn.Linear(in_features=posterior_input_size,
                           out_features=latent_layer_sizes[0]) if i == 0
                 else nn.Linear(in_features=latent_layer_sizes[i - 1], out_features=latent_layer_sizes[i])
                 for i in range(len(latent_layer_sizes) - 1)])
            penultimate_size = posterior_input_size if len(latent_layer_sizes) == 1 else latent_layer_sizes[-2]
            self.mu_posterior = nn.Linear(penultimate_size, latent_variable_size)
            self.sigma_posterior = nn.Linear(penultimate_size, latent_variable_size)

        context_size = params['hrnn_params']['conversation_encoder_hidden_size']
        if latent_layer_sizes is not None:
            context_size += latent_layer_sizes[-1]
        self.decoder = SwitchingDecoder(
            context_size=context_size,
            vocab_size=len(train_vocab),
            **params['decoder_params']
        )

        if self.cuda_available:
            self.cuda()
        self.decoder.set_pretrained_embeddings(self.encoder.gensen.encoder.src_embedding.weight.data)

    def reparametrize(self, mu, logvariance):
        """
        Sample the latent variable
        :param mu:
        :param logvar:
        :return:
        """
        std = torch.exp(0.5 * logvariance)
        tt = torch.cuda.FloatTensor if self.cuda_available else torch.FloatTensor
        eps = Variable(torch.randn(std.data.shape, out=tt()))
        return mu + eps * std

    def forward(self, input_dict, return_latent=False):
        # encoder result: (batch_size, max_conv_length, conversation_encoder_hidden_size)
        conversation_representations, sentence_representations = self.encoder(
            input_dict, return_all=True, return_sentence_representations=True)
        batch_size, max_conversation_length, max_utterance_length = input_dict["dialogue"].data.shape

        # get movie_recommendations (batch, max_conv_length, n_movies)
        if self.params['language_aware_recommender']:
            user_rep_from_language = self.language_to_user(conversation_representations)
        movie_recommendations = self.recommender_module(
            dialogue=input_dict["dialogue"],
            senders=input_dict["senders"],
            lengths=input_dict["lengths"],
            conversation_lengths=input_dict["conversation_lengths"],
            movie_occurrences=input_dict["movie_occurrences"],
            recommend_new_movies=False,
            user_representation=user_rep_from_language if self.params['language_aware_recommender'] else None
        )

        # TODO: only decode recommender's utterances
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
            batch_size * max_conversation_length, self.params['hrnn_params']['conversation_encoder_hidden_size'])\
            .index_select(0, sorted_idx)

        # shift the movie recommendations one step in time
        pad_tensor = (Variable(torch.zeros(batch_size, 1, self.n_movies, out=tt())))
        movie_recommendations = torch.cat((pad_tensor, movie_recommendations), 1).narrow(
            1, 0, max_conversation_length)
        # and reshape+reorder movie_recommendations the same way as utterances
        movie_recommendations = movie_recommendations.contiguous().view(
            batch_size * max_conversation_length, -1).index_select(0, sorted_idx)

        # consider only lengths > 0
        num_positive_lengths = np.sum(lengths > 0)
        sorted_utterances = sorted_utterances[:num_positive_lengths]
        sorted_lengths = sorted_lengths[:num_positive_lengths]
        conversation_representations = conversation_representations[:num_positive_lengths]
        movie_recommendations = movie_recommendations[:num_positive_lengths]

        # Latent variable
        if self.params['latent_layer_sizes'] is not None:
            # remember that conversation_representations have been shifted one step in time
            h_prior = conversation_representations
            for layer in self.prior_hidden_layers:
                h_prior = F.relu(layer(h_prior))
            mu_prior = self.mu_prior(h_prior)
            logvar_prior = self.sigma_prior(h_prior)
            # posterior conditioned on current context, and representation of the next utterance (that is the
            # utterance about to be decoded)
            # reshape sentence representations the same way as utterances
            sentence_representations = sentence_representations.view(
                batch_size * max_conversation_length,
                2 * self.params['hrnn_params']['sentence_encoder_hidden_size'] + 1).index_select(0, sorted_idx)
            sentence_representations = sentence_representations[:num_positive_lengths]
            h_posterior = torch.cat((conversation_representations, sentence_representations), 1)
            for layer in self.posterior_hidden_layers:
                h_posterior = F.relu(layer(h_posterior))
            mu_posterior = self.mu_posterior(h_posterior)
            logvar_posterior = self.sigma_posterior(h_posterior)

            # In training, sample from the posterior distribution. At test time, sample from prior.
            mu, logvar = (mu_posterior, logvar_posterior) if self.training else (mu_prior, logvar_prior)
            z = self.reparametrize(mu, logvar)

            context = torch.cat((conversation_representations, z), 1)
        else:
            context = conversation_representations

        # Run decoder
        outputs = self.decoder(
            sorted_utterances,
            sorted_lengths,
            context,
            movie_recommendations,
            log_probabilities=True,
            sample_movies=False
        )

        # Complete the missing sequences (of length 0)
        if num_positive_lengths < batch_size * max_conversation_length:
            tt = torch.cuda.FloatTensor if self.cuda_available else torch.FloatTensor
            pad_tensor = Variable(torch.zeros(
                batch_size * max_conversation_length - num_positive_lengths,
                max_utterance_length,
                len(self.train_vocab) + self.n_movies,
                out=tt()
            ))
            outputs = torch.cat((
                outputs,
                pad_tensor
            ), 0)

        # print("OUTPUT SHAPE :", outputs.data.shape) # (batch * max_conv_len, max_sentence_len, vocab + n_movie)
        # retrieve original order
        outputs = outputs.index_select(0, rev). \
            view(batch_size, max_conversation_length, max_utterance_length, -1)
        # print("OUTPUT SHAPE RETRIEVED IN ORDER", outputs.data.shape)
        # (batch, max_conv_len, max_sentence_len, vocab + n_movie)
        if return_latent:
            if self.params['latent_layer_sizes'] is None:
                raise ValueError("Model has no latent variable, cannot return latent parameters.")
            return outputs, mu_prior, logvar_prior, mu_posterior, logvar_posterior
        return outputs

    def train_iter(self, batch, criterion, kl_coefficient=1):
        self.train()
        if self.params['latent_layer_sizes'] is not None:
            outputs, mu_prior, logvar_prior, mu_posterior, logvar_posterior = self.forward(batch, return_latent=True)
        else:
            outputs = self.forward(batch, return_latent=False)

        batch_size, max_conv_length, max_seq_length, vocab_size = outputs.data.shape
        # indices of recommender's utterances(< batch * max_conv_len)
        idx = Variable(torch.nonzero((batch["senders"].view(-1) == -1).data).squeeze())
        # select recommender's utterances for the loss
        outputs = outputs.view(-1, max_seq_length, vocab_size).index_select(0, idx)
        target = batch["target"].view(-1, max_seq_length).index_select(0, idx)

        loss = criterion(outputs.view(-1, vocab_size), target.view(-1))

        # variational loss = KL(posterior || prior)
        if self.params['latent_layer_sizes'] is not None:
            # for two normal distributions, kld(p1, p2) =
            # log(sig2 / sig1) + (sig1^2 + (mu1-mu2)^2) / (2 sig2^2) - 1/2
            # multivariate: (sig1 and sig2 the covariance matrices)
            # .5 * (tr(sig2^-1 sig1) + (mu2-mu1)T sig2^-1 (mu2-mu1) - k + ln(det(sig2) / det(sig1))
            # in the case where sig1 and sig2 are diagonal:
            # .5 * sum(sig1^2 / sig2^2 + (mu2-mu1)^2 / sig2^2 - 1 + ln(sig2^2) - ln(sig1^2))
            kld = .5 * (-1 + logvar_prior - logvar_posterior +
                        (torch.exp(logvar_posterior) + (mu_posterior - mu_prior).pow(2)) / torch.exp(logvar_prior))
            kld = torch.mean(torch.sum(kld, -1))
            # print("NLL loss {} KLD {}".format(loss.data, kld.data))
            loss += kl_coefficient + kld
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
