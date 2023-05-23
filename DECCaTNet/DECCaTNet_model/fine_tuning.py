import pickle

import braindecode.augmentation.functional
import braindecode
import mne
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import SubsetRandomSampler
# from ray.experimental.tqdm_ray import tqdm
from tqdm import tqdm
from sklearn.model_selection import KFold
import os
import pickle as pkl
import math
from ray import tune
import copy
from pathlib import Path
from DECCaTNet_model.custom_dataset import PathDataset, ConcatPathDataset, FineTunePathDataset

import pandas as pd

from preprocessing.preprocess import _make_adjacent_groups, check_windows, run_preprocess
from .DECCaTNet_model import Encoder


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x


class FineTuneNet(nn.Module):
    def __init__(self, channel_groups, ds_channel_order, all_params, global_params):
        """
        encoder_path
        channel_group_size = 2
        embedding_size
        n_classes = 2
        """
        # self.patch_size = patch_size
        super().__init__()
        params = all_params['fine_tuning']

        self.magic = global_params['magic_constant']
        self.encoder_path = params["encoder_path"]
        self.channel_group_size = global_params["n_channels"]
        self.channel_groups = channel_groups  # Channel groups defined by names
        self.n_channel_groups = len(self.channel_groups)  # Number of channel groups
        self.embedding_size = global_params["embedding_size"]
        self.n_classes = params["n_classes"]

        self.encoder = Encoder(all_params['encoder_params'], global_params)

        #print(f'=============== debugging, this is encoder path {self.encoder_path} ==============')
        #print(f'=============== debugging, this is encoder path {os.getcwd()} ==============')

        if torch.cuda.is_available():
            self.encoder.load_state_dict(torch.load(self.encoder_path))
        else:
            self.encoder.load_state_dict(torch.load(self.encoder_path,map_location=torch.device('cpu'))) # saved in the hyperparameter tuning itself.

        self.encoder.requires_grad_(all_params['encoder_params']['FREEZE_ENCODER'])  # TODO Doesnt train the encoder during fine_tuning. (good for something i guess)

        self.out_layer_1 = all_params['downstream_params']['out_layer_1']
        self.out_layer_2 = all_params['downstream_params']['out_layer_2']
        self.dropout_1 = all_params['downstream_params']['dropout_1']
        self.dropout_2 = all_params['downstream_params']['dropout_2']

        self.ds_channel_order = ds_channel_order
        # Make dict witch translates channel names to index in preprocessed files
        self.channel_index = {}
        for i, ch in enumerate(self.ds_channel_order):
            self.channel_index[ch] = i
        # Use channel index dict to define channel groups by indexes instead of channel names
        self.channel_index_groups = []
        for group in self.channel_groups:
            self.channel_index_groups.append([self.channel_index[ch] for ch in group])

        # trans_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=8)
        # self.transformer = nn.TransformerEncoder(encoder_layer=trans_layer, num_layers=6)
        self.encoders = nn.ModuleList([self.encoder for i in range(len(self.channel_index_groups))])

        self.classifier = nn.Sequential(
            nn.Linear(in_features=int(self.embedding_size * self.n_channel_groups * self.magic),
                      out_features=self.out_layer_1),
            nn.ReLU(),
            nn.Dropout(self.dropout_1),
            nn.Linear(in_features=self.out_layer_1, out_features=self.out_layer_2),
            nn.ReLU(),
            nn.Dropout(self.dropout_2),
            nn.Linear(in_features=self.out_layer_2, out_features=1),
            #nn.Linear(in_features=self.out_layer_2, out_features=n_classes),
            # PrintLayer(),
            #nn.LogSoftmax(-1) # dont need this as we use cross entropy loss
        )

    def forward(self, X):
        if len(X.shape) == 3:
            X = X[:, None, :, :]
        # Split input into chunks that fit into the encoder
        # TODO: Decide what to do if n_channels does not fit into encoder size (Not even)
        # Do we discard the last channels? Do we make a overlap
        # TODO: Define splits from channel group index list from init to reduce run time
        channel_group_tensors = []

        for indexes in self.channel_index_groups:
            # print(indexes)
            # Select the correct channel for each window in batch, concatenate those to new tensor
            channel_group_tensors.append(
                torch.concat([X[..., i:i + 1, :] for i in indexes], dim=-2)
            )
            # print(f'channel_group: {channel_group_tensors[-1].shape}')

        # TODO: compare speed of the above with ll
        # encoder_inputs = torch.split(X, self.channel_group_size, dim=0)

        encoder_out = []
        # Run each group/pair of channels through the encoder

        for i,group in enumerate(channel_group_tensors):
            # print(f"Group shape: {group.shape}")
            encoder_out.append(self.encoders[i](group))  # Encode group

        # print(encoder_out[0].shape)
        X_encoded = torch.concat(encoder_out, dim=1)
        # print(X_encoded.shape)
        X_encoded = torch.reshape(X_encoded, (X_encoded.shape[0], X_encoded.shape[1], -1))
        # print(f'encoded shape after reshape {X_encoded.shape}')
        X_encoded = torch.flatten(X_encoded, start_dim=1)  # TODO used start_dim to keep batches seperate
        # print(f'encoded shape after reshape and flatten {X_encoded.shape}')
        x = self.classifier(X_encoded)
        return x


def n_correct_preds(y_pred, y):
    #print(y_pred,y)
    #print(torch.argmax(y_pred,dim=1))
    predicted_labels = (y_pred >= 0.5).long()
    #num_correct = (torch.argmax(y_pred, dim=1) == y).float().sum().item()
    num_correct = (predicted_labels == y).float().sum().item()
    num_total = len(y)
    #print(f'checking that n_correct_preds work: {y_pred} and y: {y}, gives num correct {num_correct}')
    # print(f'argmax pred {torch.argmax(y_pred, dim=1)} y {torch.argmax(y,dim=1)} results{torch.argmax(y_pred, dim=1) == torch.argmax(y,dim=1)}')
    return num_correct, num_total


def train_epoch(model, train_loader, device, loss_func, optimizer,disable):
    model.train()  # tells Pytorch Backend that model is trained (for example set dropout and have correct batchNorm)
    train_loss = 0
    correct_train_preds = 0
    num_train_preds = 0

    for x, y in tqdm(train_loader,disable=disable):
        #print(f'target variables before changign them {y}')
        y = torch.Tensor([ 1 if not elem else 0 for elem in y]).view(-1,1) # TODO: this is only works with n_classes = 2
        #print(f'target variables after changing: {y}')
        #y = y.type(torch.LongTensor)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        # forward pass
        pred = model(x)
        # compute loss
        loss = loss_func(pred, y)
        # update weights
        loss.backward()
        optimizer.step()

        correct, number = n_correct_preds(pred, y)
        correct_train_preds += correct
        num_train_preds += number

        # track loss
        train_loss += loss.item()

        # free up cuda memory
        del x
        del y
        del pred
        torch.cuda.empty_cache()

    print(f'=== RESULTS FROM ONE FINETUNE TRAIN EPOCH, correct preds: {correct_train_preds / num_train_preds}, train_loss: {train_loss / num_train_preds}')

    return model,train_loss, correct_train_preds, num_train_preds


def validate_epoch(model, val_loader, device, loss_func,disable):
    print('============================ RUNNING VALIDATION EPOCH ============================')
    correct_eval_preds = 0
    num_eval_preds = 0
    val_loss = 0
    with torch.no_grad():  # detach all gradients from tensors
        model.eval()  # tell model it is evaluation time
        for x, y in tqdm(val_loader, disable=disable):
            y = torch.Tensor([ 1 if not elem else 0 for elem in y]).view(-1,1)
            #y = y.type(torch.LongTensor)
            x, y = x.to(device), y.to(device)
            # get predictions
            pred = model(x)
            # get validation loss
            val_loss += loss_func(pred, y).item()
            # get correct preds and number of preds
            correct, number = n_correct_preds(pred, y)
            # update preds
            correct_eval_preds += correct
            num_eval_preds += number

            # cleare up memory
            del x
            del y
            del pred
            torch.cuda.empty_cache()

    print(f'==== VALIDATION RESULTS FROM FINETUNE,correct preds {correct_eval_preds / num_eval_preds}, loss: {val_loss / num_eval_preds}')

    return val_loss, correct_eval_preds, num_eval_preds


def train_model(epochs, model, train_loader, val_loader, test_loader, device, loss_func, optimizer, validate_test,
                early_stop=None,disable=False):
    loss = []
    val_loss = []
    test_loss = []
    train_acc = []
    val_acc = []
    test_acc = []
    print('=========================== TRAINING MODEL IN fine_tuning ===========================')
    for epoch in range(epochs):
        print('================== epoch number: ', epoch, 'of: ', epochs, ' in fine_tuning =========================')
        model, train_loss, correct_train_preds, num_train_preds = train_epoch(model, train_loader, device, loss_func,
                                                                       optimizer,disable=disable)

        val_loss_out, correct_eval_preds, num_eval_preds = validate_epoch(model, val_loader, device, loss_func,disable=disable)

        # calculate accuracies and losses and update
        loss.append(train_loss / len(train_loader))
        val_loss.append(val_loss_out / len(val_loader))
        train_acc.append(correct_train_preds / num_train_preds)
        val_acc.append(correct_eval_preds / num_eval_preds)
        if early_stop:
            if early_stop.early_stop(val_loss_out / len(val_loader)):
                print(f'reached stopping criteria in epoch {epoch}')
                break

    if validate_test:
        test_loss_out, correct_test_preds, num_test_preds = validate_epoch(model, test_loader, device, loss_func)
        test_loss.append(test_loss_out / len(test_loader))
        test_acc.append(correct_test_preds / num_test_preds)

    return loss, train_acc, val_loss, val_acc, test_loss, test_acc, model


def k_fold_training(epochs, model, dataset, batch_size, test_loader, device, loss_func, optimizer, validate_test,
                    n_folds, early_stop=None, random_state=422):
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    avg_loss = []
    avg_val_loss = []
    avg_test_loss = []
    train_acc = []
    val_acc = []
    test_acc = []
    print('=================================== PERFORMING K-FOLD-TRAINING in fine_tuning =============================')

    for fold, (train_idx, val_idx) in enumerate(folds.split(np.arange(len(dataset)))):
        print(f'Fold {fold + 1}')

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

        for epoch in range(epochs):
            print('epoch number: ', epoch, 'of: ', epochs)
            train_loss, correct_train_preds, num_train_preds = train_epoch(model, train_loader, device, loss_func,
                                                                           optimizer)
            val_loss, correct_eval_preds, num_eval_preds = validate_epoch(model, val_loader, device, loss_func)

            # calculate accuracies and losses and update
            avg_loss.append(train_loss / len(train_loader))
            avg_val_loss.append(val_loss / len(val_loader))
            train_acc.append(correct_train_preds / num_train_preds)
            val_acc.append(correct_eval_preds / num_eval_preds)
            if early_stop:
                if early_stop.early_stop(val_loss / len(val_loader)):
                    print(f'reached stopping criteria in epoch {epoch} for fold {fold} ')
                    break

    if validate_test:
        test_loss, correct_test_preds, num_test_preds = validate_epoch(model, test_loader, device, loss_func)
        avg_test_loss.append(test_loss / len(test_loader))
        test_acc.append(correct_test_preds / num_test_preds)

    return avg_loss, train_acc, avg_val_loss, val_acc, avg_test_loss, test_acc, model


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def run_fine_tuning(all_params, global_params, test_set=None):
    params = all_params['fine_tuning']
    print('=================== START FINE-TUNING ====================')
    if params['REDO_PREPROCESS']:
        print('It is not allowed to REDO_PREPROCESS when running HYPERSEARCH, skipping')

    idx = []
    # we need, idx, paths and dataset_params
    path = all_params['fine_tuning']['ds_path']
    # get splits path by first getting windows path and then removing last object
    preproc_path = os.path.join(path, 'first_preproc')
    indexes = os.listdir(preproc_path)
    for i in indexes:
        sub_dir = os.path.join(preproc_path, str(i))
        for i_window in range(
                int(pd.read_json(os.path.join(sub_dir, "description.json"), typ='series')['n_windows'])):
            idx.append((i, i_window))
    ds_params = all_params['fine_tuning']['fine_tuning_preprocess']

    dataset = FineTunePathDataset(idx, preproc_path, ds_params, global_params, ds_params['target_name'])
    epochs = params["max_epochs"]
    learning_rate = params['lr_rate']
    weight_decay = params['weight_decay']
    num_workers = global_params['n_jobs']
    perform_k_fold = params['PERFORM_KFOLD']
    n_folds = params['n_folds']

    dataset, _ = dataset.get_splits(all_params['hyper_search']['fine_tune_split'])

    # im thinking load one window
    ds_channel_order = dataset.__getitem__(0, window_order=True)

    for i in range(math.floor(len(idx) * all_params['hyper_search']['fine_tune_split'])):
        window_order = dataset.__getitem__(i, window_order=True)
        # if not window_order == ds_channel_order:
        #     changes = [ds_channel_order.index(ch_n) if ds_channel_order[i] != ch_n else i for i, ch_n in
        #                enumerate(window_order)]
        # print(ds_channel_order,'\n',windows_ds.windows.ch_names,'\n',changes)
        # assert window_order == ds_channel_order, f'{window_order} \n {ds_channel_order}' # TODO fix assertion

    channel_groups = _make_adjacent_groups(ds_channel_order, global_params['n_channels'])

    train, valid = dataset.get_splits(params['train_split'])

    train_loader = torch.utils.data.DataLoader(train, batch_size=params["batch_size"],
                                               num_workers=num_workers, shuffle=params['SHUFFLE'])
    val_loader = torch.utils.data.DataLoader(valid, batch_size=params['batch_size'],
                                             num_workers=num_workers, shuffle=params['SHUFFLE'])
    if test_set is not None:
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=params['batch_size'],
                                                  shuffle=params["SHUFFLE"],
                                                  num_workers=num_workers)
        validate_test = True
    else:
        validate_test = False
        test_loader = val_loader  # TODO remove this once we have test set

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        all_params_2 = copy.deepcopy(all_params)
        to_join = str(os.getcwd())
        all_params_2['fine_tuning']['encoder_path'] = to_join + '/' + all_params_2['fine_tuning']['encoder_path']
        model = FineTuneNet(channel_groups, ds_channel_order, all_params_2, global_params)
    except:
        assert all_params['hyper_search']['FINE_TUNING'] == True, (
            'assertion failed as this should only be accsessible when only finetuning')
        all_params['fine_tuning'][
            'encoder_path'] = '/lhome/oskarsto/repos/master-eeg-trans/DECCaTNet/' + \
                              all_params['fine_tuning']['encoder_path']
        model = FineTuneNet(channel_groups, ds_channel_order, all_params, global_params)

    if torch.cuda.is_available():
        model.cuda()
        print("Moved model to CUDA")

    loss_func = nn.CrossEntropyLoss()
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)  # TODO: check out betas for Adam and if Adam is the best choice

    early_stopper = EarlyStopper(params['early_stopper']['patience'],params['early_stopper']['min_delta'])

    if perform_k_fold:
        loss, train_acc, val_loss, val_acc, test_loss, test_acc, model = k_fold_training(epochs, model, dataset,
                                                                                         params["batch_size"],
                                                                                         test_loader, device, loss_func,
                                                                                         optimizer, validate_test,
                                                                                         n_folds, early_stopper)
    else:
        loss, train_acc, val_loss, val_acc, test_loss, test_acc, model = train_model(epochs, model, train_loader,
                                                                                     val_loader, test_loader, device,
                                                                                     loss_func, optimizer,
                                                                                     validate_test, early_stopper,disable=global_params['TQDM'])

    # TODO: implement savinf parts of models underway in training
    # save function for final model
    if not os.path.exists(params['save_dir_model']):
        os.makedirs(params['save_dir_model'])

    save_path_model = os.path.join(params['save_dir_model'], params['model_file_name'])
    torch.save(model.state_dict(), save_path_model)
    # here is the solution, have the model as several modules; then different modules can be saved seperately
    save_path_enocder = os.path.join(params['save_dir_model'], "encoder_" + params['model_file_name'])
    torch.save(model.encoder.state_dict(), save_path_enocder)

    # save all of parameters to pickelfile
    pickle_name = 'meta_data_and_params_' + params['model_file_name'] + '.pkl'

    meta_data_path = os.path.join(params['save_dir_model'], pickle_name)
    with open(meta_data_path, 'wb') as outfile:
        pkl.dump({
            "avg_train_losses": loss,
            'avg_train_acc': train_acc,
            'avg_val_losses': val_loss,
            'avg_val_acc': val_acc,
            'avg_test_loss': test_loss,
            'avg_test_acc': test_acc,
            "save_dir_for_model": params['save_dir_model'],
            "model_file_name": params['model_file_name'],
            "batch_size": params["batch_size"],
            "shuffle": params["SHUFFLE"],  # "num_workers": num_workers,
            "max_epochs": epochs,
            "learning_rate": learning_rate,
            # "beta_vals": beta_vals, # TODO: check out betavals
            "weight_decay": weight_decay,
            # "save_freq": save_freq, # TODO maybe implement
            'model_params': all_params['encoder_params'],
            "n_channels": global_params['n_channels'],  # TODO:check where number of channels need to be changed
            'dataset_names': params['ds_name'],
            'num_workers': num_workers,

        }, outfile)
        # TODO: remeber that some datasets (Abnormal/Normal) is already splitted, guessing this is implemented by Oskar.

        print("================ FINE-TUNING DONE in fine_tuning =========================")


def get_window_len(ds):
    diff = ds.windows.metadata['i_stop_in_trial'] - ds.windows.metadata['i_start_in_trial']
    return diff.to_numpy()
