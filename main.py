import DECCaNet.contrastive_framework as cf
from DECCaNet.custom_dataset import PathDataset
from DECCaNet.DECCaNet_model import DECCaNet
#from SeqCLR.contrastive_framework import pre_train_model

import pickle
if __name__ == '__main__':
    with open('datasets/TUH/TUH-pickles/windowed_ids.pkl','rb') as f:
        ids_to_load = pickle.load(f)

    path = 'datasets/TUH/preprocessed/step_2'
    dataset = PathDataset(ids_to_load=ids_to_load,path=path, preload=False)

    cf.pre_train_model(dataset, 10,0.7, 10,True,None, 1,0.01,0.01,2,10,5,'models', 'test', None)

