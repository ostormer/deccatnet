from preprocessing.run_preprocess import run_preprocess
# from SeqCLR.custom_dataset import PathDataset
import pickle
import torch
if __name__ == '__main__':

    run_preprocess('preprocessing/preprocessing_oskar.yaml', stop_idx=50)