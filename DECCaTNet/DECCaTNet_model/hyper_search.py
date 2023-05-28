import ast
import copy
import math
import os
import pickle
import time
from ray.experimental.tqdm_ray import tqdm
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as fn
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
import pickle as pkl
import torchplot as plt
from pathlib import Path

import DECCaTNet_model.contrastive_framework as cf
import preprocessing.preprocess as pre
import DECCaTNet_model.fine_tuning as fn
from ray.air.checkpoint import Checkpoint

from DECCaTNet_model import DECCaTNet_model as DECCaTNet
from DECCaTNet_model.custom_dataset import PathDataset, ConcatPathDataset, FineTunePathDataset
from ray import tune
import ray
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.schedulers import FIFOScheduler
from functools import partial
from ray.tune.search.bohb import TuneBOHB
from ray.air import session

from DECCaTNet_model.contrastive_framework import ContrastiveLoss, train_epoch, validate_epoch
from preprocessing.preprocess import _make_adjacent_groups, check_windows, run_preprocess
from DECCaTNet_model.fine_tuning import train_epoch as train_epoch_fine
from DECCaTNet_model.fine_tuning import validate_epoch as validate_epoch_fine
from DECCaTNet_model.fine_tuning import FineTuneNet
from ray.util import inspect_serializability


# things we wnt to be able to hypersearch:n_channels, channels_selection, augmentation_selection, all losses, architectures, basically a lot of shit

def hyper_search(all_params, global_params):
    hyper_prams = all_params['hyper_search']
    configs = make_correct_config(hyper_prams, all_params, global_params)

    # ray.init(num_cpus=1)
    # ray.init(num_gpus=2)
    if hyper_prams['PRE_TRAINING']:
        mode = 'min'
        metric = 'val_loss'
        search_alg = TuneBOHB(metric=metric, mode=mode)
        scheduler = HyperBandForBOHB(
            metric=metric,
            mode=mode,
            max_t=hyper_prams['max_t'],
            reduction_factor=hyper_prams['reduction_factor'])
        reporter = CLIReporter(
            # ``parameter_columns=["l1", "l2", "lr", "batch_size"]``,
            metric_columns=["val_loss", "train_loss", "training_iteration"],
            max_report_frequency=hyper_prams['max_report_frequency'])
    elif hyper_prams['PERFORM_PREPROCESS']:
        scheduler = FIFOScheduler()
        metric = 'val_acc'
        mode = 'max'
        search_alg = None
        scheduler = ASHAScheduler(
            metric=metric,
            mode=mode,
            max_t=hyper_prams['max_t'],
            reduction_factor=hyper_prams['reduction_factor'])

        reporter = CLIReporter(
            # ``parameter_columns=["l1", "l2", "lr", "batch_size"]``,
            max_report_frequency=hyper_prams['max_report_frequency'])

    elif hyper_prams['GRID_SEARCH']:
        mode = 'max'
        metric = 'val_acc'
        search_alg = None
        scheduler = ASHAScheduler(
            metric=metric,
            mode=mode,
            max_t=hyper_prams['max_t'],
            reduction_factor=hyper_prams['reduction_factor'])
        reporter = CLIReporter(
            # ``parameter_columns=["l1", "l2", "lr", "batch_size"]``,
            metric_columns=["val_loss", "train_loss", 'val_acc', "training_iteration"],
            max_report_frequency=hyper_prams['max_report_frequency'])
    else:
        mode = 'max'
        metric = 'val_acc'
        search_alg = TuneBOHB(metric=metric, mode=mode)
        scheduler = HyperBandForBOHB(
            metric=metric,
            mode=mode,
            max_t=hyper_prams['max_t'],
            reduction_factor=hyper_prams['reduction_factor'])
        reporter = CLIReporter(
            # ``parameter_columns=["l1", "l2", "lr", "batch_size"]``,
            metric_columns=["val_loss", "train_loss", 'val_acc', "training_iteration"],
            max_report_frequency=hyper_prams['max_report_frequency'])
    if global_params['n_gpu'] > 0:
        trainable = tune.with_resources(hyper_search_train, resources={'gpu': 0.9, 'cpu': math.floor(
            global_params['n_jobs'] / (global_params['n_gpu'])), "accelerator_type:V100": 0.45})
    else:
        n_jobs = math.floor(global_params['n_jobs'] / 5)
        if n_jobs == 0:
            n_jobs = 1
        trainable = tune.with_resources(hyper_search_train, resources={'cpu': n_jobs})

    # Try t fix noisy logging
    # ray.init(configure_logging=True, logging_level=logging.ERROR)
    # configs['RAY_DEDUP_LOGS'] = 0
    # configs['log_level'] = 'ERROR'
    # os.environ['RAY_DEDUP_LOGS'] = '0'

    result = tune.run(
        tune.with_parameters(trainable, hyper_params=hyper_prams, all_params=copy.deepcopy(all_params),
                             global_params=copy.deepcopy(global_params)),
        config=configs,
        num_samples=hyper_prams['num_samples'],
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir='../tune_results',
        name=global_params['experiment_name'],
        verbose=2,
        search_alg=search_alg,
        # reuse_actors=False
    )

    best_trial = result.get_best_trial(metric=metric, mode=mode)
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

    if hyper_params['FINE_AND_PRE']: # testing only pre_training variables here.
        for key in all_params['pre_training']:
            if key in config:
                all_params['pre_training'][key] = config[key]
        for key in all_params['downstream_params']:
            if key in config:
                all_params['downstream_params'][key] = config[key]
        for key in all_params['encoder_params']:
            if key in config:
                all_params['encoder_params'][key] = config[key]
        to_join = str(os.getcwd())
        model_path = to_join + '/' + all_params['fine_tuning']['encoder_path']
        if not os.path.isfile(model_path):
            print('====== NO PRETRAINED MODEL, STARTING PRETRAINING =====')
            print(f'Tried to find the following path {model_path}')
            cf.pre_train_model(copy.deepcopy(all_params), global_params)
        else:
            print('====== ALREADY PERFORMED PRETRAINING, GO TO FINETUNING =========')
        fine_tuning_hypersearch(copy.deepcopy(all_params), global_params)
    elif hyper_params['PRE_TRAINING']:
        for key in all_params['pre_training']:
            if key in config:
                all_params['pre_training'][key] = config[key]
        for key in all_params['encoder_params']:
            if key in config:
                all_params['encoder_params'][key] = config[key]
        pre_train_hypersearch(all_params, global_params)
    elif hyper_params['FINE_TUNING']:
        for key in all_params['fine_tuning']:
            if key in config:
                all_params['fine_tuning'][key] = config[key]
        for key in all_params['downstream_params']:
            if key in config:
                all_params['downstream_params'][key] = config[key]
        for key in all_params['encoder_params']:
            if key in config:
                all_params['encoder_params'][key] = config[key]
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
        print("======================== Moved model to CUDA FOR FINE-TUNING ==========================")

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)  # TODO: check out betas for Adam and if Adam is the best choice

    if perform_k_fold:
        k_fold_training(epochs, model, dataset,
                        params["batch_size"],
                        test_loader, device,
                        loss_func,
                        optimizer, validate_test,
                        n_folds, disable=global_params['TQDM'])
    else:
        train_model(epochs, model, train_loader,
                    val_loader, test_loader,
                    device,
                    loss_func, optimizer,
                    validate_test, disable=global_params['TQDM'])


def train_model(epochs, model, train_loader, val_loader, test_loader, device, loss_func, optimizer, validate_test,
                early_stop=None, disable=False):
    for epoch in range(epochs):
        print(f'============== HYPER SEARCH FINE-TUNING EPOCH: {epoch} of {epochs}==========================')
        model, train_loss, correct_train_preds, num_train_preds = train_epoch_fine(model, train_loader, device,
                                                                                   loss_func,
                                                                                   optimizer, disable=disable)

        val_loss, correct_eval_preds, num_eval_preds = validate_epoch_fine(model, val_loader, device, loss_func,
                                                                           disable=disable)

        # checkpoint = Checkpoint.from_dict({"epoch": epoch})

        session.report({'val_loss': val_loss / num_eval_preds, 'train_loss': train_loss / num_train_preds,
                        'val_acc': correct_eval_preds / num_eval_preds,
                        'train_acc': correct_train_preds / num_train_preds})  # ,checkpoint=checkpoint)


def k_fold_training(epochs, model, dataset, batch_size, test_loader, device, loss_func, optimizer, validate_test,
                    n_folds, early_stop=None, random_state=422, disable=False):
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    print('========================== STARTED K-FOLD-TRAINING ====================================')
    for fold, (train_idx, val_idx) in enumerate(folds.split(np.arange(len(dataset)))):
        print(f'=====Fold {fold + 1}=========')

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

        for epoch in range(epochs):
            print('epoch number in k_fold training: ', epoch, 'of: ', epochs)
            model, train_loss, correct_train_preds, num_train_preds = train_epoch_fine(model, train_loader, device,
                                                                                       loss_func,
                                                                                       optimizer, disable=disable)
            val_loss, correct_eval_preds, num_eval_preds = validate_epoch_fine(model, val_loader, device, loss_func,
                                                                               disable=disable)

            # checkpoint = Checkpoint.from_dict({"epoch": epoch})
            session.report({'val_loss': val_loss / len(val_loader), 'train_loss': train_loss / len(train_loader),
                            'val_acc': correct_eval_preds / num_eval_preds})  # , checkpoint=checkpoint)


def pre_train_hypersearch(all_params=None, global_params=None):
    params = all_params['pre_training']
    preprocess_params = all_params['preprocess']
    pretrain_datasets = params['datasets']
    datasets_dict = {}
    for dataset in pretrain_datasets:
        preprocess_root = preprocess_params[dataset]['preprocess_root']
        with open(os.path.join(preprocess_root, 'pickles',
                               f'{global_params["channel_select_function"]}_{global_params["n_channels"]}' + 'split_idx_list.pkl'),
                  'rb') as fid:
            pre_train_ids = pickle.load(fid)
        datasets_dict[dataset] = (os.path.join(preprocess_root,
                                               f'split_{global_params["channel_select_function"]}_{global_params["n_channels"]}'),
                                  pre_train_ids)

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

    dataset, _ = dataset.get_splits(all_params['hyper_search']['pre_train_split'])
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
    print('========================== START PRE-TRAINING in HYPER-SEARCH =================================')
    for epoch in range(max_epochs):
        print(f'========== Epoch nr {epoch} of {max_epochs} =======================')
        model, counter, epoch_loss = train_epoch(model, epoch, max_epochs, train_loader, device, optimizer, loss_func,
                                                 time_process, batch_print_freq, time_names, batch_size,
                                                 disable=global_params['TQDM'])
        val_loss = validate_epoch(model, val_loader, device, loss_func, disable=global_params['TQDM'])

        losses.append(epoch_loss)
        val_losses.append(val_loss)
        session.report({'val_loss': val_loss, 'train_loss': epoch_loss})  # , checkpoint=checkpoint)

    # save function for final model, needs to be done anyway
    save_path_model = os.path.join(save_dir_model, model_file_name)
    torch.save(model.state_dict(), save_path_model)
    # here is the solution, have the model as several modules; then different modules can be saved seperately
    save_path_enocder = os.path.join(save_dir_model, "encoder_" + model_file_name)
    torch.save(model.encoder.state_dict(), save_path_enocder)
