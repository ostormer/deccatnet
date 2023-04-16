import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm
from sklearn.model_selection import KFold
import os
import pickle as pkl

from DECCaTNet.preprocessing.preprocess import _make_adjacent_pairs
from .DECCaTNet_model import Encoder


class EncodingClassifier(nn.Module):
    def __init__(self):
        pass

    def forward(self, X):
        return X


class FineTuneNet(nn.Module):
    def __init__(self, channel_groups, ds_channel_order, params):
        """
        encoder_path
        channel_group_size = 2
        embedding_size
        n_classes = 2
        """
        # self.patch_size = patch_size
        super().__init__()
        self.encoder_path = params["encoder_path"]
        self.channel_group_size = params["channel_group_size"]
        self.channel_groups = channel_groups  # Channel groups defined by names
        self.n_channel_groups = len(self.channel_groups)  # Number of channel groups
        self.embedding_size = params["embedding_size"]
        self.n_classes = params["n_classes"]

        self.encoder = Encoder(emb_size=self.embedding_size)
        self.encoder.load_state_dict(torch.load(self.encoder_path))
        self.encoder.requires_grad_(False)

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

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.embedding_size * self.n_channel_groups, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=self.n_classes),
            nn.LogSoftmax(self.n_classes)
        )

    def forward(self, X):
        print("New sample:!!!")
        print(X.shape)
        X = X[:, None, :, :]
        # Split input into chunks that fit into the encoder
        # TODO: Decide what to do if n_channels does not fit into encoder size (Not even)
        # Do we discard the last channels? Do we make a overlap
        # TODO: Define splits from channel group index list from init to reduce run time
        channel_group_tensors = []

        for indexes in self.channel_index_groups:
            print(indexes)
            # Select the correct channel for each window in batch, concatenate those to new tensor
            channel_group_tensors.append(
                torch.concat([X[..., i:i + 1, :] for i in indexes], dim=-2)
            )
            print(channel_group_tensors[-1].shape)

        # TODO: compare speed of the above with ll
        # encoder_inputs = torch.split(X, self.channel_group_size, dim=0)

        encoder_out = []
        # Run each group/pair of channels through the encoder
        for group in channel_group_tensors:
            print(f"Group shape: {group.shape}")
            encoder_out.append(self.encoder(group))  # Encode group

        print(encoder_out[0].shape)
        X_encoded = torch.concat(encoder_out, dim=0)
        print(X_encoded.shape)
        X_encoded = torch.reshape(X_encoded, (X_encoded.shape[0], -1))
        X_encoded = torch.flatten(X_encoded)
        x = self.classifier(X_encoded)
        return x


def n_correct_preds(y_pred, y):
    num_correct = (torch.argmax(y_pred, dim=1) == y).float().sum().item()
    num_total = len(y)
    return num_correct, num_total


def train_epoch(model, train_loader, device, loss_func, optimizer):
    model.train()  # tells Pytorch Backend that model is trained (for example set dropout and have correct batchNorm)
    train_loss = 0
    correct_train_preds = 0
    num_train_preds = 0

    for x, y, crop_inds in tqdm(train_loader, position=0, leave=True):
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

    return train_loss, correct_train_preds, num_train_preds


def validate_epoch(model, val_loader, device, loss_func):
    correct_eval_preds = 0
    num_eval_preds = 0
    val_loss = 0
    with torch.no_grad():  # detach all gradients from tensors
        model.eval()  # tell model it is evaluation time
        for x, y, crops_inds in tqdm(val_loader, position=0, leave=True):
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

    return val_loss, correct_eval_preds, num_eval_preds


def train_model(epochs, model, train_loader, val_loader, test_loader, device, loss_func, optimizer, validate_test,early_stop):
    loss = []
    val_loss = []
    test_loss = []
    train_acc = []
    val_acc = []
    test_acc = []
    for epoch in range(epochs):
        print('epoch number: ', epoch, 'of: ', epochs)
        train_loss, correct_train_preds, num_train_preds = train_epoch(model, train_loader, device, loss_func,
                                                                       optimizer)

        val_loss, correct_eval_preds, num_eval_preds = validate_epoch(model, val_loader, device, loss_func)

        # calculate accuracies and losses and update
        loss.append(train_loss / len(train_loader))
        val_loss.append(val_loss / len(val_loader))
        train_acc.append(correct_train_preds / num_train_preds)
        val_acc.append(correct_eval_preds / num_eval_preds)

        if early_stop.early_stop(val_loss / len(val_loader)):
            print(f'reached stopping criteria in epoch {epoch}')
            break

    if validate_test:
        test_loss, correct_test_preds, num_test_preds = validate_epoch(model, test_loader, device, loss_func)
        test_loss.append(test_loss / len(test_loader))
        test_acc.append(correct_test_preds / num_test_preds)

    return loss, train_acc, val_loss, val_acc, test_loss, test_acc, model


def k_fold_training(epochs, model, dataset, batch_size, test_loader, device, loss_func, optimizer, validate_test,
                    n_folds, early_stop,random_state=422):
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    loss = []
    val_loss = []
    test_loss = []
    train_acc = []
    val_acc = []
    test_acc = []

    for fold, (train_idx, val_idx) in tqdm(enumerate(folds.split(np.arange(len(dataset)))), position=0, leave=True):
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
            loss.append(train_loss / len(train_loader))
            val_loss.append(val_loss / len(val_loader))
            train_acc.append(correct_train_preds / num_train_preds)
            val_acc.append(correct_eval_preds / num_eval_preds)

            if early_stop.early_stop(val_loss / len(val_loader)):
                print(f'reached stopping criteria in epoch {epoch} for fold {fold} ')
                break

    if validate_test:
        test_loss, correct_test_preds, num_test_preds = validate_epoch(model, test_loader, device, loss_func)
        test_loss.append(test_loss / len(test_loader))
        test_acc.append(correct_test_preds / num_test_preds)

    return loss, train_acc, val_loss, val_acc, test_loss, test_acc, model

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

def run_fine_tuning(dataset, params, test_set=None, perform_k_fold=True, n_folds=0):
    epochs = params["epochs"]
    learning_rate = params['lr_rate']
    weight_decay = params['weight_decay']
    num_workers = params['num_workers']
    ds_channel_order = dataset.datasets[0].windows.ch_names
    for windows_ds in dataset.datasets:
        assert windows_ds.windows.ch_names == ds_channel_order
    print("All recordings have the correct channel order")

    early_stopper = EarlyStopper(params['patience'],params['min_delta'])

    channel_groups = _make_adjacent_pairs(ds_channel_order)
    model = FineTuneNet(channel_groups, ds_channel_order, params)

    split = dataset.split("train")
    train = split["True"]
    test = split["False"]
    # TODO: get train/val/test set, will use test as val for now

    train_loader = torch.utils.data.DataLoader(train, batch_size=params["batch_size"], shuffle=params["shuffle"],
                                               num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(test, batch_size=params['batch_size'], shuffle=params["shuffle"],
                                             num_workers=num_workers)
    if test_set is not None:
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=params['batch_size'], shuffle=params["shuffle"],
                                                  num_workers=num_workers)
        validate_test = True
    else:
        validate_test = False
        test_loader = torch.utils.data.DataLoader(test, batch_size=params['batch_size'], shuffle=params["shuffle"],
                                                  num_workers=num_workers)  # TODO remove this once we have test set

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        model.cuda()
        print("Moved model to CUDA")

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)  # TODO: check out betas for Adam and if Adam is the best choice

    if perform_k_fold:
        loss, train_acc, val_loss, val_acc, test_loss, test_acc, model = k_fold_training(epochs, model, dataset,
                                                                                         params["batch_size"],
                                                                                         test_loader, device, loss_func,
                                                                                         optimizer, validate_test,
                                                                                         n_folds,early_stopper)
    else:
        loss, train_acc, val_loss, val_acc, test_loss, test_acc, model = train_model(epochs, model, train_loader,
                                                                                     val_loader, test_loader, device,
                                                                                     loss_func, optimizer,
                                                                                     validate_test,early_stopper)

    # TODO: implement savinf parts of models underway in training
    # save function for final model
    save_path_model = os.path.join(params['save_dir_model'],params['model_file_name'])
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
            'avg_val_losses':val_loss,
            'avg_val_acc': val_acc,
            'avg_test_loss':test_loss,
            'avg_test_acc':test_acc,
            "save_dir_for_model": params['save_dir_model'],
            "model_file_name": params['model_file_name'],
            "batch_size": params["batch_size"],
            "shuffle": params["shuffle"],  # "num_workers": num_workers,
            "max_epochs": epochs,
            "learning_rate": learning_rate,
            #"beta_vals": beta_vals, # TODO: check out betavals
            "weight_decay": weight_decay,
            #"save_freq": save_freq, # TODO maybe implement
            'model_params': params['model_params'],
            "n_channels": n_channels, #TODO:check where number of channels need to be changed
            'n_models': n_models,
            'dataset_names':params['dataset_name'],
            'num_workers':num_workers,


        }, outfile)
        # TODO: remeber that some datasets (Abnormal/Normal) is already splitted, guessing this is implemented by Oskar.

        print("Training done")
