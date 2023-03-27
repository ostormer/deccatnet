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

    cf.pre_train_model(dataset=dataset, batch_size=128,train_split=0.7,save_freq=10,shuffle=True,
                       trained_model_path=None,temperature=1,learning_rate=0.01,weight_decay=0.01,
                       num_workers=2,max_epochs=10,batch_print_freq=5,save_dir_model='models', model_file_name='test',
                       model_params=None)

