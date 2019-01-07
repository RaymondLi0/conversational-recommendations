import os
import time
import shutil
import numpy as np
import torch
from tqdm import tqdm

import config
from models.autorec import AutoRec, ReconstructionLoss
from batch_loaders.ml_batch_loader import MlBatchLoader
from batch_loaders.batch_loader import DialogueBatchLoader
from utils import create_dir
import test_params


def train(model, batch_loader, nb_epochs, patience, batch_input, save_path,
          eval_at_beginning=True, max_num_inputs=None, weight_decay=0):
    """
    train model
    :param model: model to train
    :param batch_loader:
    :param batch_input: batch_input used in training (not in validation). "full" or "random_noise"
    :param save_path: path to save the model
    :param eval_at_beginning: if True, perform valid evaluation before beginning training.
    :return:
    """
    epoch = 0
    patience_count = 0
    best_loss = 1e10
    n_train_batches = batch_loader.n_batches["train"]

    training_losses = []
    validation_losses = []
    start_time = time.time()
    criterion = ReconstructionLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        weight_decay=weight_decay
    )

    if eval_at_beginning:
        # Evaluate
        val_loss = model.evaluate(batch_loader=batch_loader, criterion=criterion, subset="valid", batch_input="full")
        print('--------------------------------------------------------------')
        validation_losses.append(val_loss)
        # Write logs
        with open(os.path.join(save_path, "logs"), "a+") as f:
            text = "EPOCH {} : losses {} {} TIME {} s \n". \
                format(epoch, -1, val_loss, time.time() - start_time)
            f.write(text)
        # Save model
        is_best = val_loss < best_loss
        best_loss = min(best_loss, val_loss)
        save_checkpoint({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "params": model.params,
            "best_loss": best_loss,
        }, is_best, save_path)

    while epoch < nb_epochs:
        model.train()
        losses = []
        for _ in tqdm(range(n_train_batches)):
            batch = batch_loader.load_batch(subset="train", batch_input=batch_input, max_num_inputs=max_num_inputs)
            if model.cuda_available:
                batch["input"] = batch["input"].cuda()
                batch["target"] = batch["target"].cuda()

            # Train iteration: forward, backward and optimize
            optimizer.zero_grad()
            outputs = model(batch["input"])
            # reconstruction loss
            loss = criterion(outputs, batch["target"])
            loss.backward()
            optimizer.step()
            loss = loss.data[0]

            # keep losses in memory
            losses.append(loss)
        epoch_loss = criterion.normalize_loss_reset(np.sum(losses))
        print('Epoch : {} Training Loss : {}'.format(epoch, epoch_loss))
        training_losses.append(epoch_loss)

        # Evaluate
        val_loss = model.evaluate(batch_loader=batch_loader, criterion=criterion, subset="valid", batch_input="full")

        # print('Epoch : {} Validation Loss : {}'.format(epoch, val_loss))
        print('--------------------------------------------------------------')
        validation_losses.append(val_loss)
        epoch += 1

        # Write logs
        with open(os.path.join(save_path, "logs"), "a+") as f:
            text = "EPOCH {} : losses {} {} TIME {} s \n". \
                format(epoch, training_losses[-1], val_loss, time.time() - start_time)
            f.write(text)

        # Save model
        is_best = val_loss < best_loss
        best_loss = min(best_loss, val_loss)
        save_checkpoint({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "params": model.params,
            "best_loss": best_loss,
        }, is_best, save_path)

        # Early stopping
        if is_best:
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print("Early stopping, {} epochs without best".format(patience_count))
                break

    print("Training done.")
    return False


def save_checkpoint(state, is_best, path):
    torch.save(state, os.path.join(path, "checkpoint"))
    if is_best:
        shutil.copy(os.path.join(path, "checkpoint"), os.path.join(path, "model_best"))


def explore_params(params_seq, data="movielens"):
    """

    :param params_seq: sequence of tuples (save_folder, model_params, train_params)
    :return:
    """
    for (save_path, params, train_params) in params_seq:
        print("Saving in {} with parameters : {}, {}".format(save_path, params, train_params))
        create_dir(save_path)
        # for experiment on movielens only
        if data == "movielens":
            # train on five splits
            for (i, data_path) in enumerate(config.ML_SPLIT_PATHS):
                batch_loader = MlBatchLoader(
                    batch_size=train_params["batch_size"],
                    data_path=data_path
                )
                print("n movies", batch_loader.n_movies)
                model = AutoRec(n_movies=batch_loader.n_movies, params=params)
                create_dir(save_path + "/split{}".format(i))
                train(model, batch_loader=batch_loader, batch_input=train_params["batch_input"],
                      nb_epochs=train_params["nb_epochs"],
                      patience=train_params["patience"],
                      max_num_inputs=train_params["max_num_inputs"],
                      save_path=save_path + "/split{}".format(i))
        # for experiments on our data
        elif data == "db_pretrain":
            # pre-train on movielens
            batch_loader = MlBatchLoader(
                batch_size=train_params["batch_size"],
                ratings01=True
            )
            print("n movies", batch_loader.n_movies)
            model = AutoRec(n_movies=batch_loader.n_movies, params=params)
            create_dir(save_path + "/movielens")
            train(model, batch_loader=batch_loader, batch_input=train_params["batch_input"],
                  nb_epochs=train_params["nb_epochs"],
                  patience=train_params["patience"],
                  max_num_inputs=train_params["max_num_inputs"],
                  save_path=save_path + "/movielens")
            # train on our DB.
            # Re-create model and load it from pre-training folder
            batch_loader = DialogueBatchLoader(sources="ratings", batch_size=64)
            print("n movies", batch_loader.n_movies)
            model = AutoRec(n_movies=batch_loader.n_movies, resume=save_path + "/movielens/model_best", params=params)
            train(
                model,
                batch_loader=batch_loader,
                nb_epochs=train_params["nb_epochs"],
                patience=train_params["patience"],
                batch_input=train_params["batch_input"],
                max_num_inputs=train_params["max_num_inputs"],
                eval_at_beginning=True,
                save_path=save_path
            )
        elif data == "db":
            # No pre-training
            batch_loader = DialogueBatchLoader(sources="ratings", batch_size=16)
            model = AutoRec(n_movies=batch_loader.n_movies, params=params)
            train(
                model,
                batch_loader=batch_loader,
                nb_epochs=train_params["nb_epochs"],
                patience=train_params["patience"],
                batch_input=train_params["batch_input"],
                max_num_inputs=train_params["max_num_inputs"],
                eval_at_beginning=True,
                save_path=save_path
            )
        else:
            raise ValueError(
                "data parameter expected to be 'movielens', 'db_pretrain' or 'db'. Got '{}' instead".format(data))


if __name__ == '__main__':
    params = [(config.AUTOREC_MODEL, test_params.autorec_params, test_params.train_autorec_params)]
    explore_params(params_seq=params, data="db_pretrain")
