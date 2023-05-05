import ast
import copy
import math
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as fn
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm
import pickle as pkl
import torchplot as plt
from pathlib import Path

import DECCaTNet_model.contrastive_framework as cf
import preprocessing.preprocess as pre
import DECCaTNet_model.fine_tuning as fn

from DECCaTNet_model import DECCaTNet_model as DECCaTNet
from DECCaTNet_model.custom_dataset import PathDataset, ConcatPathDataset, FineTunePathDataset
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
    ray.init(num_gpus=2)
    if hyper_prams['PRE_TRAINING']:
        mode = 'min'
        metric = 'val_loss'
        scheduler = ASHAScheduler(
            metric=metric,
            mode=mode,
            max_t=hyper_prams['max_t'],
            grace_period=hyper_prams['grace_period'],
            reduction_factor=hyper_prams['reduction_factor'])
        reporter = CLIReporter(
            # ``parameter_columns=["l1", "l2", "lr", "batch_size"]``,
            metric_columns=["val_loss", "train_loss", "training_iteration"])
    else:
        mode = 'max'
        metric = 'val_acc'
        scheduler = ASHAScheduler(
            metric=metric,
            mode=mode,
            max_t=hyper_prams['max_t'],
            grace_period=hyper_prams['grace_period'],
            reduction_factor=hyper_prams['reduction_factor'])
        reporter = CLIReporter(
            # ``parameter_columns=["l1", "l2", "lr", "batch_size"]``,
            metric_columns=["val_loss", "train_loss", 'val_acc', "training_iteration"],
            max_report_frequency=hyper_prams['max_report_frequency'])
    if global_params['n_cpu']>0:
        trainable = tune.with_resources(hyper_search_train,resources={'gpu':1, 'cpu':math.floor(global_params['n_jobs']/global_params['n_cpu'])})
    else:
        trainable = tune.with_resources(hyper_search_train, resources={'gpu': 1, 'cpu': math.floor(global_params['n_jobs']/5)})
    result = tune.run(
        partial(trainable, hyper_params=hyper_prams, all_params=copy.deepcopy(all_params),
                global_params=copy.deepcopy(global_params)),
        config=configs,
        num_samples=hyper_prams['num_samples'],
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir='../tune_results'
    )
    best_trial = result.get_best_trial(metric=metric,mode=mode)
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["val_loss"]))
    if mode == 'max':
        print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["val_acc"]))


def make_correct_config(hyper_params, all_params, global_params):
    config = {}
    for key in hyper_params['config']:
        config[key] = eval(hyper_params['config'][key])
    return config

def hyper_search_train(config, hyper_params=None, all_params=None, global_params=None):
    # check for global params in config
    for key in global_params:
        if key in config:
            global_params[key] = config[key]
    if hyper_params['PERFORM_PREPROCESS']:
        pre.run_preprocess(all_params, global_params)

    if hyper_params['FINE_AND_PRE']:
        for key in all_params['pre_training']:
            if key in config:
                all_params['pre_training'][key] = config[key]
        cf.pre_train_model(copy.deepcopy(all_params), global_params)
        fine_tuning_hypersearch(copy.deepcopy(all_params), global_params)
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
    print('=================== START FINE-TUNING IN HYPER-SEARCH ====================')

    if params['REDO_PREPROCESS']:
        print('It is not allowed to REDO_PREPROCESS when running HYPERSEARCH, skipping')

    idx = []
    # we need, idx, paths and dataset_params
    path = all_params['fine_tuning']['ds_path']
    # get splits path by first getting windows path and then removing last object
    preproc_path = os.path.join(*Path(path).parts[:-2], 'first_preproc')
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

    # im thinking load one window
    ds_channel_order = dataset.__getitem__(0, window_order=True)

    for i in range(len(idx)):
        window_order = dataset.__getitem__(i, window_order=True)
        # if not window_order == ds_channel_order:
        #     changes = [ds_channel_order.index(ch_n) if ds_channel_order[i] != ch_n else i for i, ch_n in
        #                enumerate(window_order)]
            # print(ds_channel_order,'\n',windows_ds.windows.ch_names,'\n',changes)
        #assert window_order == ds_channel_order, f'{window_order} \n {ds_channel_order}' # TODO fix assertion

    channel_groups = _make_adjacent_groups(ds_channel_order, global_params['n_channels'])

    dataset, _ = dataset.get_splits(all_params['hyper_search']['fine_tune_split'])

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
        model = FineTuneNet(channel_groups, ds_channel_order, all_params, global_params)
    except:
        assert all_params['hyper_search']['FINE_TUNING'] == True, (
            'assertion failed as this should only be accsessible when only finetuning')
        all_params['fine_tuning'][
            'encoder_path'] = 'C:/Users/Styrk/OneDrive-NTNU/Documents/Skole/Master/master_code/master-eeg-trans/DECCaTNet/' + \
                              all_params['fine_tuning']['encoder_path']
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
                            'val_acc': correct_eval_preds / num_eval_preds})  # , checkpoint=checkpoint)


def pre_train_hypersearch(all_params=None, global_params=None):
    params = all_params['pre_training']
    preprocess_params = all_params['preprocess']
    pretrain_datasets = params['datasets']
    datasets_dict = {}
    for dataset in pretrain_datasets:
        preprocess_root = preprocess_params[dataset]['preprocess_root']
        with open(os.path.join(preprocess_root, 'pickles', f'{global_params["channel_select_function"]}_{global_params["n_channels"]}'+'split_idx_list.pkl'), 'rb') as fid:
            pre_train_ids = pickle.load(fid)
        datasets_dict[dataset] = (os.path.join(preprocess_root, f'split_{global_params["channel_select_function"]}_{global_params["n_channels"]}'), pre_train_ids)

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

    dataset,_ = dataset.get_splits(all_params['hyper_search']['pre_train_split'])
    # load dataset
    train_set, val_set = dataset.get_splits(train_split)

    # create data_loaders, here batch size is decided
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle,
                                               num_workers=num_workers, drop_last=True)
    # maybe alos num_workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

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
        session.report({'val_loss': val_loss, 'train_loss': epoch_loss})  # , checkpoint=checkpoint)

    # save function for final model, needs to be done anyway
    save_path_model = os.path.join(save_dir_model, model_file_name)
    torch.save(model.state_dict(), save_path_model)
    # here is the solution, have the model as several modules; then different modules can be saved seperately
    save_path_enocder = os.path.join(save_dir_model, "encoder_" + model_file_name)
    torch.save(model.encoder.state_dict(), save_path_enocder)
