import ast
import copy
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm
import pickle as pkl
import torchplot as plt

import DECCaTNet_model.contrastive_framework as cf
import preprocessing.preprocess as pre
import DECCaTNet_model.fine_tuning as fn

from DECCaTNet_model import DECCaTNet_model as DECCaTNet
from DECCaTNet_model.custom_dataset import PathDataset, ConcatPathDataset,FineTunePathDataset
from ray import tune
import ray
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
from ray.air import session

from DECCaTNet_model.contrastive_framework import ContrastiveLoss, train_epoch, validate_epoch
from preprocessing.preprocess import _make_adjacent_groups, check_windows, run_preprocess
from DECCaTNet_model.fine_tuning import train_epoch, validate_epoch, FineTuneNet
from ray.util import inspect_serializability

# things we wnt to be able to hypersearch:n_channels, channels_selection, augmentation_selection, all losses, architectures, basically a lot of shit

def hyper_search(all_params, global_params):
    hyper_prams = all_params['hyper_search']
    configs = make_correct_config(hyper_prams, all_params, global_params)
    #ray.init(num_cpus=1)
    if hyper_prams['PRE_TRAINING']:
        scheduler = ASHAScheduler(
            metric="val_loss",
            mode="min",
            max_t=hyper_prams['max_t'],
            grace_period=1,
            reduction_factor=2)
        reporter = CLIReporter(
            # ``parameter_columns=["l1", "l2", "lr", "batch_size"]``,
            metric_columns=["val_loss", "train_loss", "training_iteration"])
    else:
        scheduler = ASHAScheduler(
            metric="val_acc",
            mode="max",
            max_t=hyper_prams['max_t'],
            grace_period=1,
            reduction_factor=2)
        reporter = CLIReporter(
            # ``parameter_columns=["l1", "l2", "lr", "batch_size"]``,
            metric_columns=["val_loss", "train_loss", 'val_acc', "training_iteration"],
            max_report_frequency=30)

    result = tune.run(
        partial(hyper_search_train, hyper_params=hyper_prams, all_params=copy.deepcopy(all_params), global_params=copy.deepcopy(global_params)),
        config=configs,
        num_samples=1,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir='../tune_results'
    )
    best_trial = result.get_best_trial()
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["val_acc"]))


def make_correct_config(hyper_params, all_params, global_params):
    config = {}
    for key in hyper_params['config']:
        config[key] = eval(hyper_params['config'][key])
    return config


def update_paths(config, all_params, global_params, change_on):
    for dataset in global_params['datasets']:
        all_params['preprocess'][dataset]['preprocess_root'] = all_params['preprocess'][dataset][
                                                                   'preprocess_root'] + '_' + change_on + '_' + str(
            config[
                change_on])
        all_params['pre_training'][dataset]['ds_path'] = all_params['preprocess'][dataset][
                                                             'preprocess_root'] + '/split'
        all_params['pre_training'][dataset]['ids_path'] = all_params['preprocess'][dataset][
                                                              'preprocess_root'] + '/pickles/split_idx_list.pkl'
    all_params['fine_tuning']['fine_tuning_preprocess']['preprocess_root'] = \
        all_params['fine_tuning']['fine_tuning_preprocess']['preprocess_root'] + '_' + change_on + '_' + str(
            config[change_on])
    all_params['fine_tuning']['ds_path'] = all_params['fine_tuning']['fine_tuning_preprocess'][
                                               'preprocess_root'] + '/pickles/windowed_ds.pkl'
    return all_params


def update_model_paths(config, all_params):
    # all datasets are chosen correctly, however selecting the correct model for encoding is not yet done.
    # for example: finding correct pretraining parameters for rpetraining loss
    # when testing for something in the final model, it is important that the correct encoder is chosen
    # when testing for something in the encoder, it is impo
    pass


def hyper_search_train(config, hyper_params=None, all_params=None, global_params=None):
    # check for global params in config
    for key in global_params:
        if key in config:
            global_params[key] = config[key]
    if hyper_params['PREPROCESS']:
        # should change names for preprocessing, only n_channels and channel_selection which can be changed
        if 'n_channels' in config:
            all_params = update_paths(config, all_params, global_params, 'n_channels')
        if 'channel_select_function' in config:
            all_params = update_paths(config, all_params, global_params, 'channel_select_function')
    if hyper_params['PERFORM_PREPROCESS']:
        pre.run_preprocess(all_params, global_params)
    if hyper_params['FINE_AND_PRE']:
        for key in all_params['pre_training']:
            if key in config:
                all_params['pre_training'][key] = config[key]
        cf.pre_train_model(all_params, global_params)
        fine_tuning_hypersearch(all_params, global_params)
    elif hyper_params['PRE_TRAINING']:
        for key in all_params['pre_training']:
            if key in config:
                all_params['pre_training'][key] = config[key]
        pre_train_hypersearch(all_params, global_params)
    elif hyper_params['FINE_TUNING']:
        for key in all_params['pre_training']:
            if key in config:
                all_params['pre_training'][key] = config[key]
        fine_tuning_hypersearch(copy.deepcopy(all_params), global_params)


def fine_tuning_hypersearch(all_params=None, global_params=None, test_set=None):
    params = all_params['fine_tuning']

    if params['REDO_PREPROCESS']:  # this should allways be true, will take up time
        all_params['preprocess'] = params['fine_tuning_preprocess']
        idx,path,ds_params,orig_dataset = run_preprocess(all_params, global_params, fine_tuning=True)[0]
        dataset = FineTunePathDataset(idx,path,ds_params,global_params,ds_params['target_name'])

    epochs = params["max_epochs"]
    learning_rate = params['lr_rate']
    weight_decay = params['weight_decay']
    num_workers = global_params['n_jobs']
    perform_k_fold = params['PERFORM_KFOLD']
    n_folds = params['n_folds']

    ds_channel_order = orig_dataset.datasets[0].windows.ch_names

    for i, windows_ds in enumerate(orig_dataset.datasets):
        if not windows_ds.windows.ch_names == ds_channel_order:
            changes = [ds_channel_order.index(ch_n) if ds_channel_order[i] != ch_n else i for i, ch_n in
                       enumerate(windows_ds.windows.ch_names)]
            # print(ds_channel_order,'\n',windows_ds.windows.ch_names,'\n',changes)
        assert windows_ds.windows.ch_names == ds_channel_order, f'{windows_ds.windows.ch_names} \n {ds_channel_order}'  # TODO remove comment, i cant pass this assertion. Might have something to do with paralellization

    channel_groups = _make_adjacent_groups(ds_channel_order, global_params['n_channels'])

    train,valid = dataset.get_splits(params['train_split'])

    train_loader = torch.utils.data.DataLoader(train, batch_size=params["batch_size"],
                                               num_workers=num_workers,shuffle=params['SHUFFLE'])
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
        model = FineTuneNet(channel_groups, ds_channel_order, all_params, global_params)
    except:
        all_params['fine_tuning']['encoder_path'] = 'C:/Users/Styrk/OneDrive-NTNU/Documents/Skole/Master/master_code/master-eeg-trans/DECCaTNet/' + all_params['fine_tuning']['encoder_path']
        model = FineTuneNet(channel_groups, ds_channel_order, all_params, global_params)

    if torch.cuda.is_available():
        model.cuda()
        print("Moved model to CUDA")

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)  # TODO: check out betas for Adam and if Adam is the best choice

    if perform_k_fold:
        k_fold_training(epochs, model, dataset,
                        params["batch_size"],
                        test_loader, device,
                        loss_func,
                        optimizer, validate_test,
                        n_folds)
    else:
        train_model(epochs, model, train_loader,
                    val_loader, test_loader,
                    device,
                    loss_func, optimizer,
                    validate_test)


def train_model(epochs, model, train_loader, val_loader, test_loader, device, loss_func, optimizer, validate_test,
                early_stop=None):
    for epoch in range(epochs):
        print('epoch number: ', epoch, 'of: ', epochs)
        train_loss, correct_train_preds, num_train_preds = train_epoch(model, train_loader, device, loss_func,
                                                                       optimizer)

        val_loss, correct_eval_preds, num_eval_preds = validate_epoch(model, val_loader, device, loss_func)

        session.report({'val_loss': val_loss / len(val_loader), 'train_loss': train_loss / len(train_loader),
                        'val_acc': correct_eval_preds / num_eval_preds})


def k_fold_training(epochs, model, dataset, batch_size, test_loader, device, loss_func, optimizer, validate_test,
                    n_folds, early_stop=None, random_state=422):
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

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

            session.report({'val_loss': val_loss / len(val_loader), 'train_loss': train_loss / len(train_loader),
                            'val_acc': correct_eval_preds / num_eval_preds})#, checkpoint=checkpoint)


def pre_train_hypersearch(all_params=None, global_params=None):
    params = all_params['pre_training']
    all_dataset = global_params['datasets']
    datasets_dict = {}
    for dataset in all_dataset:
        path_params = params[dataset]
        with open(path_params['ids_path'], 'rb') as fid:
            pre_train_ids = pickle.load(fid)
        datasets_dict[dataset] = (path_params['ds_path'], pre_train_ids)

    dataset = ConcatPathDataset(datasets_dict, all_params, global_params)

    batch_size = params['batch_size']
    train_split = params['train_split']
    shuffle = params['SHUFFLE']
    trained_model_path = params['pretrained_model_path']
    save_freq = params['save_freq']
    temperature = params['temperature']
    learning_rate = params['learning_rate']
    weight_decay = params['weight_decay']
    num_workers = global_params['n_jobs']
    max_epochs = params['max_epochs']
    batch_print_freq = params['batch_print_freq']
    save_dir_model = params['save_dir_model']
    model_file_name = params['model_file_name']
    time_process = params['TIME_PROCESS']

    n_channels = global_params['n_channels']

    if not os.path.exists(save_dir_model):
        os.makedirs(save_dir_model)

    # load dataset
    train_set, val_set = dataset.get_splits(train_split)

    # create data_loaders, here batch size is decided
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle,
                                               num_workers=num_workers, drop_last=True)
    # maybe alos num_workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    # check if cuda setup allowed:
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # init model and check if weights already given
    model = DECCaTNet.DECCaTNet(all_params, global_params)

    if torch.cuda.is_available():
        model.cuda()
        print("Moved model to CUDA")

    # get loss function and optimizer
    loss_func = ContrastiveLoss(temperature=temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)  # TODO: check out betas for Adam and if Adam is the best choice

    # track losses
    losses = []
    val_losses = []
    time_names = ['batch', 'to_device', 'encoding', 'loss_calculation', 'backward', 'loss_update', 'delete', 'total']
    # iterative traning loop
    for epoch in range(max_epochs):
        model, counter, epoch_loss = train_epoch(model, epoch, max_epochs, train_loader, device, optimizer, loss_func,
                                                 time_process, batch_print_freq, time_names, batch_size)
        val_loss = validate_epoch(model, val_loader, device, loss_func)

        losses.append(epoch_loss)
        val_losses.append(val_loss)
        session.report({'val_loss': val_loss, 'train_loss': epoch_loss})#, checkpoint=checkpoint)

    # save function for final model, needs to be done anyway
    save_path_model = os.path.join(save_dir_model, model_file_name)
    torch.save(model.state_dict(), save_path_model)
    # here is the solution, have the model as several modules; then different modules can be saved seperately
    save_path_enocder = os.path.join(save_dir_model, "encoder_" + model_file_name)
    torch.save(model.encoder.state_dict(), save_path_enocder)
