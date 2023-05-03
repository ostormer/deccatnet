import DECCaTNet_model.contrastive_framework as cf
import preprocessing.preprocess as pre
import DECCaTNet_model.fine_tuning as fn
import DECCaTNet_model.hyper_search as hs

import pickle
import os
import yaml

import pickle

def run(config_path_name):
    config_path = os.path.join('DECCaTNet_model/configs',config_path_name+'.yaml')
    with open(config_path, 'r') as fid:
        params = yaml.safe_load(fid)
    global_params = params['global']
    if global_params['HYPER_SEARCH']:
        hs.hyper_search(params,global_params)
    if global_params['PREPROCESSING']:
        pre.run_preprocess(params,global_params)
    if global_params['PRE_TRAINING']:
        cf.pre_train_model(params,global_params)
    if global_params['FINE_TUNING']:
        fn.run_fine_tuning(params,global_params)

# TODO: how to report correct performance when we are ready to do so.

if __name__ == '__main__':
    run('config_anakin')

