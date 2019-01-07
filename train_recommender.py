import os
import shutil
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import config
from models.recommender_model import Recommender
from models.hred import HRED
from batch_loaders.batch_loader import DialogueBatchLoader
from batch_loaders.reddit_batch_loader import RedditBatchLoader
import test_params
from utils import create_dir


def train(model, batch_loader, nb_epochs, patience, save_path):
    def save_model(val_loss, best_loss, epoch, patience_count):
        # Save model
        is_best = val_loss < best_loss
        best_loss = min(best_loss, val_loss)
        save_checkpoint({
            "params": model.params,
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "best_loss": best_loss,
        }, is_best, save_path)

        # Patience
        if is_best:
            patience_count = 0
        else:
            patience_count += 1
        return best_loss, patience_count

    # set word2id in batchloader from encoder
    batch_loader.set_word2id(model.encoder.word2id)
    epoch = 0
    patience_count = 0
    best_loss = 1e10
    n_train_batches = batch_loader.n_batches["train"]

    training_losses = []
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    while epoch < nb_epochs:
        model.train()
        losses = []
        for i in tqdm(range(n_train_batches)):
            batch = batch_loader.load_batch(subset="train")
            # dummy_output = torch.Tensor(np.ones(target.data.shape))
            # dialogue and movie inputs are handled by gensen, so no need to send them to GPU
            if model.cuda_available:
                batch["dialogue"] = batch["dialogue"].cuda()
                batch["target"] = batch["target"].cuda()
                batch["senders"] = batch["senders"].cuda()

            optimizer.zero_grad()
            loss = model.train_iter(batch, criterion=criterion)
            optimizer.step()
            # keep losses in memory
            losses.append(loss)

            # intra epoch evaluations for long trainings
            if i > 0 and i % 10000 == 0:
                # Evaluate
                val_loss = model.evaluate(batch_loader=batch_loader, criterion=criterion)
                print('Epoch : {} Validation Loss : {}'.format(epoch + float(i) / n_train_batches, val_loss))
                # Write logs
                with open(os.path.join(save_path, "logs"), "a+") as f:
                    text = "EPOCH {} : losses {} {} \n". \
                        format(epoch + float(i) / n_train_batches, np.mean(losses), val_loss)
                    f.write(text)

                best_loss, patience_count = save_model(
                    val_loss, best_loss, epoch + float(i) / n_train_batches, patience_count)
                if patience_count >= patience:
                    print("Early stopping, {} epochs without best".format(patience_count))
                    return
        else:
            epoch += 1
            print('Epoch : {} Training Loss : {}'.format(epoch, np.mean(losses)))
            training_losses.append(np.mean(losses))

            # Evaluate
            val_loss = model.evaluate(batch_loader=batch_loader, criterion=criterion)

            print('Epoch : {} Validation Loss : {}'.format(epoch, val_loss))
            print('--------------------------------------------------------------')

            # Write logs
            with open(os.path.join(save_path, "logs"), "a+") as f:
                text = "EPOCH {} : losses {} {} \n". \
                    format(epoch, training_losses[-1], val_loss)
                f.write(text)

            best_loss, patience_count = save_model(val_loss, best_loss, epoch, patience_count)
            if patience_count >= patience:
                print("Early stopping, {} epochs without best".format(patience_count))
                return

    print("Training done.")
    return False


def save_checkpoint(state, is_best, path):
    torch.save(state, os.path.join(path, "checkpoint"))
    if is_best:
        shutil.copy(os.path.join(path, "checkpoint"), os.path.join(path, "model_best"))


def explore_params(params_seq, data="movie_dialogue", hred=False):
    """

    :param params_seq: sequence of tuples (save_folder, model_params, train_params)
    :return:
    """
    if hred:
        model_class = HRED
        sources = "dialogue"
    else:
        model_class = Recommender
        sources = "dialogue movie_occurrences movieIds_in_target"
    for (save_path, params, train_params) in params_seq:
        print("Saving in {} with parameters : {}, {}".format(save_path, params, train_params))
        create_dir(save_path)

        if data == "movie_dialogue_pretrained":
            # pre train on reddit data set
            batch_loader = RedditBatchLoader(batch_size=train_params["batch_size"])
            model = model_class(train_vocab=batch_loader.train_vocabulary, n_movies=batch_loader.n_movies,
                                params=params)
            train(
                model, batch_loader=batch_loader,
                nb_epochs=train_params["nb_epochs"], patience=20, save_path=save_path + "/reddit"
            )

            # train on our DB.
            batch_loader = DialogueBatchLoader(
                sources=sources,
                batch_size=train_params["batch_size"]
            )
            train(
                model, batch_loader=batch_loader,
                nb_epochs=train_params["nb_epochs"], patience=train_params["patience"], save_path=save_path
            )
        elif data == "movie_dialogue":
            # Just train on our DB
            batch_loader = DialogueBatchLoader(
                sources=sources,
                batch_size=train_params["batch_size"]
            )
            model = model_class(train_vocab=batch_loader.train_vocabulary, n_movies=batch_loader.n_movies,
                                params=params)
            train(
                model,
                batch_loader=batch_loader,
                nb_epochs=train_params["nb_epochs"],
                patience=train_params["patience"],
                save_path=save_path
            )
        elif data == "increasing_data_size":
            for size in [1000, 2000, 4000, 6000, -1]:
                batch_loader = DialogueBatchLoader(
                    sources=sources,
                    batch_size=train_params["batch_size"],
                    training_size=size
                )
                model = model_class(
                    train_vocab=batch_loader.train_vocabulary,
                    n_movies=batch_loader.n_movies,
                    params=params,
                )
                train(
                    model,
                    batch_loader=batch_loader,
                    nb_epochs=train_params["nb_epochs"],
                    patience=train_params["patience"],
                    save_path=save_path + "/{}training".format(size)
                )

        else:
            raise ValueError(
                "data parameter expected to be 'movie_dialogue' or 'movie_dialogue_pretrained'. Got '{}' instead".
                    format(data))


if __name__ == '__main__':
    params_seq = [(config.RECOMMENDER_MODEL, test_params.recommender_params, test_params.train_recommender_params)]
    explore_params(params_seq, data="movie_dialogue")
