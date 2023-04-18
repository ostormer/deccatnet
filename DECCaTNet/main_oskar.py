import pickle

import yaml

from DECCaTNet.DECCaTNet_model.contrastive_framework import pre_train_model
from DECCaTNet.DECCaTNet_model.custom_dataset import PathDataset
from DECCaTNet.DECCaTNet_model.fine_tuning import run_fine_tuning
from DECCaTNet.preprocessing.preprocess import run_preprocess

if __name__ == '__main__':
    ds = run_preprocess('preprocessing/preprocessing_oskar.yaml')