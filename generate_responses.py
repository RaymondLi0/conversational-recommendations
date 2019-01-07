from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import argparse

from models.recommender_model import Recommender
from sequence_generator import SequenceGenerator
from batch_loaders.batch_loader import DialogueBatchLoader
from utils import load_model
from beam_search import get_best_beam
import test_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path")
    parser.add_argument("--save_path")
    parser.add_argument("--beam_size", default=10)
    parser.add_argument("--n_examples", default=10)
    parser.add_argument("--only_best", default="True",
                        help="whether to display all the beam results, or only the best")
    parser.add_argument("--full_dialogue", default="True",
                        help="whether to display the full dialogue or only the answers from the model")
    parser.add_argument("--subset", default="test",
                        help="subset on which to condition the model")
    args = parser.parse_args()

    temperatures = [1]
    batch_loader = DialogueBatchLoader(
        sources="dialogue movie_occurrences movieIds_in_target",
        batch_size=1
    )
    rec = Recommender(
        batch_loader.train_vocabulary,
        batch_loader.n_movies,
        params=test_params.recommender_params
    )
    load_model(rec, args.model_path)
    batch_loader.set_word2id(rec.encoder.word2id)
    generator = SequenceGenerator(
        rec.decoder,
        beam_size=args.beam_size,
        word2id=batch_loader.word2id,
        movie_id2name=batch_loader.id2name,
        max_sequence_length=40
    )
    batch_loader.batch_index[args.subset] = 0

    # START
    with open(args.save_path, "w") as f:
        f.write("")
    for _ in tqdm(range(args.n_examples)):
        # Load batch
        batch_index = batch_loader.batch_index[args.subset]
        batch = batch_loader.load_batch(subset=args.subset)
        if rec.cuda_available:
            batch["dialogue"] = batch["dialogue"].cuda()
            batch["target"] = batch["target"].cuda()
            batch["senders"] = batch["senders"].cuda()

        # 1) Compute the contexts and recommendation vectors
        # encoder result: (conv_length, hidden_size)
        conversation_representations = rec.encoder(batch, return_all=True).squeeze()
        # get movie_recommendations
        movie_recommendations = rec.recommender_module(
            dialogue=batch["dialogue"],
            senders=batch["senders"],
            lengths=batch["lengths"],
            conversation_lengths=batch["conversation_lengths"],
            movie_occurrences=batch["movie_occurrences"],
            recommend_new_movies=True,
        ).squeeze()  # (conv_length, n_movies)
        conv_length = movie_recommendations.data.shape[0]

        # select contexts after seeker's utterances
        # indices of seeker's utterances(< conv_len)
        idx = Variable(torch.nonzero((batch["senders"].view(-1) == 1).data).squeeze())
        if rec.cuda_available:
            idx = idx.cuda()
        conversation_representations = conversation_representations.index_select(0, idx)
        movie_recommendations = movie_recommendations.index_select(0, idx)
        # if first utterance is recommender, add a 0-context at the beginning
        if batch["senders"].data.cpu()[0][0] == -1:
            tt = torch.cuda.FloatTensor if rec.cuda_available else torch.FloatTensor
            conversation_representations = torch.cat((
                Variable(torch.zeros((1, rec.params["hrnn_params"]["conversation_encoder_hidden_size"]), out=tt())),
                conversation_representations), 0)
            movie_recommendations = torch.cat((Variable(torch.zeros((1, rec.n_movies), out=tt())),
                                               movie_recommendations), 0)

        # Latent variable
        if rec.params['latent_layer_sizes'] is not None:
            # remember that conversation_representations have been shifted one step in time
            h_prior = conversation_representations
            for layer in rec.prior_hidden_layers:
                h_prior = F.relu(layer(h_prior))
            mu_prior = rec.mu_prior(h_prior)
            logvar_prior = rec.sigma_prior(h_prior)
            # No need of posterior for generation

            # In training, sample from the posterior distribution. At test time, sample from prior.
            mu, logvar = (mu_prior, logvar_prior)
            z = rec.reparametrize(mu, logvar)

            context = torch.cat((conversation_representations, z), 1)
        else:
            context = conversation_representations

        # 2) generate sentences conditioned on the contexts and recommendation vectors
        index = 0
        if args.full_dialogue:
            output_str = "CONVERSATION {} \n".format(batch_index)
        else:
            output_str = ""
        messages = [[batch_loader.id2word[w] for w in sentence[:length]]
                    for (sentence, length) in zip(batch["dialogue"][0].data.cpu().tolist(), batch["lengths"][0])]
        # keep track of movies mentioned by the model, so that it does not recommend twice the same movie
        mentioned_movies = set()
        for (i, msg) in enumerate(messages):
            if batch["senders"][0].data[i] == -1:  # sent by recommender: generate response
                # continue
                if args.full_dialogue:
                    output_str += "GROUND TRUTH: " + " ".join(msg) + "\n"
                for temperature in temperatures:
                    # BEAM SEARCH
                    beams = generator.beam_search(
                        [batch_loader.word2id["<s>"]],
                        forbid_movies=mentioned_movies,
                        # add batch dimension
                        context=context[index].unsqueeze(0),
                        movie_recommendations=movie_recommendations[index].unsqueeze(0),
                        sample_movies=True,
                        temperature=temperature
                    )
                    if args.only_best:
                        # add best beam
                        best_beam = get_best_beam(beams)
                        if args.full_dialogue:
                            output_str += "GENERATED T={}: ".format(temperature)
                        output_str += best_beam.get_string(batch_loader.id2word) + "\n"
                        # update set of mentioned movies
                        mentioned_movies.update(best_beam.mentioned_movies)
                        print("mentioned movies", mentioned_movies)
                    else:
                        # show all beams sorted by likelihood
                        sorted_beams = sorted(beams, key=lambda b: -b.likelihood)
                        for (beam_rank, beam) in enumerate(sorted_beams):
                            if args.full_dialogue:
                                output_str += "GENERATED T={}, nb {}: ".format(temperature, beam_rank)
                            output_str += beam.get_string(batch_loader.id2word) + "\n"
                index += 1
            else:  # sent by seeker
                if args.full_dialogue:
                    output_str += "SEEKER: " + " ".join(msg) + "\n"
        output_str += "\n"
        with open(args.save_path, "a") as f:
            f.write(output_str)

