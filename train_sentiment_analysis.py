import os
import shutil
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import config
from models.sentiment_analysis import SentimentAnalysis, SentimentAnalysisLoss
from models.sentiment_analysis_baseline import SentimentAnalysisBaseline, SentimentAnalysisBaselineLoss
from batch_loaders.batch_loader import DialogueBatchLoader
import test_params
from utils import create_dir


# When changing loss function during training, nb of epochs before changing
change_at_epoch = 3


# function that freezes parts of the model
def freeze(model):
    for param in model.encoder.sentence_encoder.parameters():
        param.requires_grad = False
    for param in model.encoder.conversation_encoder.parameters():
        param.requires_grad = False


def train(model, batch_loader, baseline, save_path, nb_epochs, patience,
          targets="suggested seen liked", use_class_weights=True,
          start_with_class_weights=False, cut_dialogues=-1):
    """
    Train the SentimentAnalysis model
    :param cut_dialogues:
    :param patience:
    :param nb_epochs:
    :param save_path:
    :param baseline:
    :param batch_loader:
    :param model:
    :param start_with_class_weights: if True, use class weights at the beginning, and remove them after change_at_epoch
     epochs
    :return:
    """
    # set word2id in batchloader from encoder
    if baseline:
        batch_loader.set_word2id(model.gensen.task_word2id)
        loss_class = SentimentAnalysisBaselineLoss
    else:
        batch_loader.set_word2id(model.encoder.word2id)
        loss_class = SentimentAnalysisLoss

    epoch = 0
    patience_count = 0
    best_loss = 1e10
    n_train_batches = batch_loader.n_batches["train"]

    training_losses = []
    validation_losses = []

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    # Criterion. Weights for imbalanced classes (95 likes for 5 dislikes approx)
    if use_class_weights:
        criterion = loss_class(class_weight={"liked": use_class_weights},
                               use_targets=targets)
    else:
        criterion = loss_class(class_weight=None, use_targets=targets)

    while epoch < nb_epochs:
        model.train()
        # do not use weights anymore, freeze a part of the model
        if start_with_class_weights and epoch >= change_at_epoch:
            criterion.liked_criterion = nn.NLLLoss()
            freeze(model)
        losses = []
        for _ in tqdm(range(n_train_batches)):
            if cut_dialogues == "epoch":
                batch = batch_loader.load_batch(subset="train", cut_dialogues=epoch + 1)
            else:
                batch = batch_loader.load_batch(subset="train", cut_dialogues=cut_dialogues)
            if model.cuda_available:
                batch["dialogue"] = batch["dialogue"].cuda()
                batch["forms"] = batch["forms"].cuda()
                batch["senders"] = batch["senders"].cuda()
                if not baseline:
                    batch["movie_occurrences"] = batch["movie_occurrences"].cuda()

            # Train iteration: forward, backward and optimize
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch["forms"])
            # optimize
            loss.backward()
            optimizer.step()
            loss = loss.data[0]

            # keep losses in memory
            losses.append(loss)

        print('Epoch : {} Training Loss : {}'.format(epoch, np.mean(losses)))
        training_losses.append(np.mean(losses))

        # Evaluate
        val_loss = model.evaluate(batch_loader=batch_loader, criterion=criterion)

        # print('Epoch : {} Validation Loss : {}'.format(epoch, val_loss))
        print('--------------------------------------------------------------')
        validation_losses.append(val_loss)
        epoch += 1

        with open(os.path.join(save_path, "logs"), "a+") as f:
            text = "EPOCH {} : losses {} {} \n". \
                format(epoch, training_losses[-1], val_loss)
            f.write(text)
        # Keep track of best loss for early stopping (disabled if before the loss change)
        if not start_with_class_weights or epoch >= change_at_epoch:
            is_best = val_loss < best_loss
            best_loss = min(best_loss, val_loss)
        else:
            # if start with class weights == True and epoch < change_at_epoch, do not update best_loss.
            is_best = True
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


def evaluate(model, batch_loader, resume, baseline, subset="valid", show_wrong=False):
    # set word2id in batchloader from encoder
    if baseline:
        batch_loader.set_word2id(model.gensen.task_word2id)
    else:
        batch_loader.set_word2id(model.encoder.word2id)
    if not os.path.isfile(resume):
        raise ValueError("no checkpoint found at '{}'".format(resume))

    print("=> loading checkpoint '{}'".format(resume))
    if model.cuda_available:
        checkpoint = torch.load(resume)
    else:
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {} with loss {})"
          .format(resume, checkpoint['epoch'], checkpoint['best_loss']))

    model.evaluate(batch_loader, print_matrices=True, subset=subset, show_wrong=show_wrong)


def save_checkpoint(state, is_best, path):
    torch.save(state, os.path.join(path, "checkpoint"))
    if is_best:
        shutil.copy(os.path.join(path, "checkpoint"), os.path.join(path, "model_best"))


def explore_params(params_seq, baseline=False, data="standard"):
    """

    :param params_seq: sequence of tuples (save_folder, model_params, train_params)
    :return:
    """
    if baseline:
        model_class = SentimentAnalysisBaseline
        sources = "sentiment_analysis movie_occurrences"
    else:
        model_class = SentimentAnalysis
        sources = "sentiment_analysis movie_occurrences"
    for (save_path, params, train_params) in params_seq:
        create_dir(save_path)
        print("Saving in {} with parameters : {}, {}".format(save_path, params, train_params))
        if "start_with_class_weights" in train_params and train_params["start_with_class_weights"]:
            train_params["use_class_weights"] = True
            print("start_with_class_weights is set to True, setting use_class_weights=True")
        else:
            train_params["start_with_class_weights"] = False

        if train_params["use_class_weights"]:
            train_params["use_class_weights"] = [1. / 5, 1. / 80, 1. / 15]

        if data == "standard":
            batch_loader = DialogueBatchLoader(
                sources=sources,
                batch_size=train_params['batch_size']
            )
            sa = model_class(params=params, train_vocab=batch_loader.train_vocabulary)
            if baseline:
                train(
                    sa,
                    nb_epochs=train_params["nb_epochs"],
                    patience=train_params["patience"],
                    save_path=save_path,
                    baseline=baseline,
                    batch_loader=batch_loader,
                    targets=train_params["targets"],
                    use_class_weights=train_params['use_class_weights'],
                    start_with_class_weights=train_params["start_with_class_weights"],
                    cut_dialogues=0
                )
            else:
                train(
                    sa,
                    nb_epochs=train_params["nb_epochs"],
                    patience=train_params["patience"],
                    save_path=save_path,
                    baseline=baseline,
                    batch_loader=batch_loader,
                    targets=train_params["targets"],
                    use_class_weights=train_params['use_class_weights'],
                    start_with_class_weights=train_params["start_with_class_weights"],
                    cut_dialogues=train_params['cut_dialogues']
                )
        elif data == "increasing_data_size":
            for size in [1000, 2000, 4000, 6000, -1]:
                batch_loader = DialogueBatchLoader(
                    sources=sources,
                    batch_size=train_params['batch_size'],
                    training_size=size
                )
                sa = model_class(params=params, train_vocab=batch_loader.train_vocabulary)
                if baseline:
                    train(
                        sa,
                        nb_epochs=train_params["nb_epochs"],
                        patience=train_params["patience"],
                        save_path=save_path + "/{}training".format(size),
                        baseline=baseline,
                        batch_loader=batch_loader,
                        targets=train_params["targets"],
                        use_class_weights=train_params['use_class_weights'],
                        start_with_class_weights=train_params["start_with_class_weights"],
                        cut_dialogues=0
                    )
                else:
                    train(
                        sa,
                        nb_epochs=train_params["nb_epochs"],
                        patience=train_params["patience"],
                        save_path=save_path + "/{}training".format(size),
                        baseline=baseline,
                        batch_loader=batch_loader,
                        targets=train_params["targets"],
                        use_class_weights=train_params['use_class_weights'],
                        start_with_class_weights=train_params["start_with_class_weights"],
                        cut_dialogues=train_params['cut_dialogues']
                    )


if __name__ == '__main__':
    params_seq = [(config.SENTIMENT_ANALYSIS_MODEL, test_params.sentiment_analysis_params, test_params.train_sa_params)]
    explore_params(params_seq, baseline=False)
