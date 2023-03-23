from DECCaNet.contrastive_framework import pre_train_model
from DECCaNet.custom_dataset import PathDataset
#from SeqCLR.contrastive_framework import pre_train_model
import pickle
if __name__ == '__main__':
    with open('datasets/TUH/TUH-pickles/windowed_ids.pkl','rb') as f:
        ids_to_load = pickle.load(f)

    path = 'datasets/TUH/preprocessed/step_2'
    dataset = PathDataset(ids_to_load=ids_to_load,path=path, preload=False)

    pre_train_model()