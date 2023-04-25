import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
from tqdm import tqdm
import pickle as pkl
import torchplot as plt
import DECCaTNet_model.contrastive_framework as cf
import preprocessing.preprocess as pre
import DECCaTNet_model.fine_tuning as fn

from DECCaTNet_model import DECCaTNet_model as DECCaTNet
from DECCaTNet_model.custom_dataset import PathDataset, ConcatPathDataset
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
from DECCaTNet_model.contrastive_framework import ContrastiveLoss, train_epoch, validate_epoch
from preprocessing.preprocess import _make_adjacent_groups, check_windows, run_preprocess
from DECCaTNet_model.fine_tuning import k_fold_training, train_model, FineTuneNet


# things we wnt to be able to hypersearch:n_channels, channels_selection, augmentation_selection, all losses, architectures, basically a lot of shit

def hyper_search(all_params, global_params):
    hyper_prams = all_params['hyper_search']
    configs, all_params, global_params = make_correct_config(hyper_prams, all_params, global_params)


def hyper_search_train(config, hyper_params=None, all_params=None, global_params=None, checkpoint_dir=None):
    # check for global params in config
    for key in global_params:
        if key in config:
            global_params[key] = config[key]
    if hyper_params['PREPROCESSING']:
        pre.run_preprocess(all_params, global_params)
    if hyper_params['FINE_AND_PRE']:
        if hyper_params['PRE_TRAINING']:
            for key in all_params['pre_training']:
                if key in config:
                    all_params['pre_training'][key] = config[key]
        elif hyper_params['FINE_TUNING']:
            for key in all_params['pre_training']:
                if key in config:
                    all_params['pre_training'][key] = config[key]
        cf.pre_train_model(all_params,global_params)
        fine_tuning_hypersearch(all_params,global_params,checkpoint_dir)
    elif hyper_params['PRE_TRAINING']:
        for key in all_params['pre_training']:
            if key in config:
                all_params['pre_training'][key] = config[key]
        pre_train_hypersearch(all_params, global_params, checkpoint_dir)
    elif hyper_params['FINE_TUNING']:
        for key in all_params['pre_training']:
            if key in config:
                all_params['pre_training'][key] = config[key]
        fine_tuning_hypersearch(all_params, global_params, checkpoint_dir)

def fine_tuning_hypersearch(all_params=None,global_params=None,checkpoint_dir=None,test_set=None):
    params = all_params['fine_tuning']

    if params['REDO_PREPROCESS']: # this should allways be true, will take up time
        all_params['preprocess'] = params['fine_tuning_preprocess']
        dataset = run_preprocess(all_params, global_params, fine_tuning=True)[0]

    epochs = params["max_epochs"]
    learning_rate = params['lr_rate']
    weight_decay = params['weight_decay']
    num_workers = global_params['n_jobs']
    perform_k_fold = params['PERFORM_KFOLD']
    n_folds = params['n_folds']

    ds_channel_order = dataset.datasets[0].windows.ch_names

    for i, windows_ds in enumerate(dataset.datasets):
        if not windows_ds.windows.ch_names == ds_channel_order:
            changes = [ds_channel_order.index(ch_n) if ds_channel_order[i] != ch_n else i for i, ch_n in
                       enumerate(windows_ds.windows.ch_names)]
            # print(ds_channel_order,'\n',windows_ds.windows.ch_names,'\n',changes)
        # assert windows_ds.windows.ch_names == ds_channel_order, f'{windows_ds.windows.ch_names} \n {ds_channel_order}' # TODO remove comment, i cant pass this assertion. Might have something to do with paralellization
    print("All recordings have the correct channel order")

    channel_groups = _make_adjacent_groups(ds_channel_order, global_params['n_channels'])
    model = FineTuneNet(channel_groups, ds_channel_order, all_params, global_params)

    train, test = torch.utils.data.random_split(dataset, [params['train_split'], 1 - params['train_split']])

    train_loader = torch.utils.data.DataLoader(train, batch_size=params["batch_size"], shuffle=params["SHUFFLE"],
                                               num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(test, batch_size=params['batch_size'], shuffle=params["SHUFFLE"],
                                             num_workers=num_workers)
    if test_set is not None:
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=params['batch_size'],
                                                  shuffle=params["SHUFFLE"],
                                                  num_workers=num_workers)
        validate_test = True
    else:
        validate_test = False
        test_loader = torch.utils.data.DataLoader(test, batch_size=params['batch_size'], shuffle=params["SHUFFLE"],
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
                                                                                         test_loader, device,
                                                                                         loss_func,
                                                                                         optimizer, validate_test,
                                                                                         n_folds, early_stopper)
    else:
        loss, train_acc, val_loss, val_acc, test_loss, test_acc, model = train_model(epochs, model, train_loader,
                                                                                     val_loader, test_loader,
                                                                                     device,
                                                                                     loss_func, optimizer,
                                                                                     validate_test, early_stopper)

    # TODO: implement savinf parts of models underway in training
    # save function for final model
    if not os.path.exists(params['save_dir_model']):
        os.makedirs(params['save_dir_model'])

    save_path_model = os.path.join(params['save_dir_model'], params['model_file_name'])
    torch.save(model.state_dict(), save_path_model)
    # here is the solution, have the model as several modules; then different modules can be saved seperately
    save_path_enocder = os.path.join(params['save_dir_model'], "encoder_" + params['model_file_name'])
    torch.save(model.encoder.state_dict(), save_path_enocder)



def pre_train_hypersearch(all_params=None, global_params=None, checkpoint_dir=None):
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
                                               num_workers=num_workers)
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

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

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

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(val_loss=val_loss, train_loss=epoch_loss)

    print('Traning Done!!')


def run_hypersearch(all_params, global_params):
    config = make_correct_config(all_params['config'])
    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=all_params['pre_training']['max_epochs'],
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # ``parameter_columns=["l1", "l2", "lr", "batch_size"]``,
        metric_columns=["val_loss", "epoch_loss", "training_iteration"])
    result = tune.run(
        partial(train_hypersearch, all_params=all_params, global_params=global_params),
        config=config,
        num_samples=10,
        scheduler=scheduler,
        progress_reporter=reporter
    )
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))


def make_correct_config(hyper_params, all_params, global_params):
    config_helper = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16])
    }
    return config, all_params, global_params
