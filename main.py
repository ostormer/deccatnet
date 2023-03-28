import DECCaTNet.contrastive_framework as cf
from DECCaTNet.custom_dataset import PathDataset
from DECCaTNet.DECCaTNet_model import DECCaTNet
#from SeqCLR.contrastive_framework import pre_train_model

import pickle
if __name__ == '__main__':
    with open('datasets/TUH/TUH-pickles/windowed_ids.pkl','rb') as f:
        ids_to_load = pickle.load(f)

    path = 'datasets/TUH/preprocessed/step_2'
    dataset = PathDataset(ids_to_load=ids_to_load,path=path, preload=False)

    cf.pre_train_model(dataset=dataset, batch_size=256,train_split=0.7,save_freq=10,shuffle=True,
                       trained_model_path=None,temperature=1,learning_rate=0.01,weight_decay=0.01,
                       num_workers=2,max_epochs=10,batch_print_freq=2,save_dir_model='models', model_file_name='test',
                       model_params=None, time_process=True)

"""
batch_size = 2
Average time used on batch : 0.0008
Average time used on encoding : 0.0732
Average time used on loss_calculation : 0.0026
Average time used on backward : 0.1165
Average time used on loss_update : 0.0366
Average time used on delete : 0.0007

batch_size = 8
Average time used on batch : 0.0002
Average time used on encoding : 0.0605
Average time used on loss_calculation : 0.0096
Average time used on backward : 0.1196
Average time used on loss_update : 0.0091
Average time used on delete : 0.0008


batch_size = 32
Average time used on batch : 0.0005
Average time used on encoding : 0.0624
Average time used on loss_calculation : 0.0425
Average time used on backward : 0.1767
Average time used on loss_update : 0.003
Average time used on delete : 0.0026

batch_size = 128
Average time used on batch : 0.0001
Average time used on to_device : 0.0
Average time used on encoding : 0.1012
Average time used on loss_calculation : 0.131
Average time used on loss_update : 0.4354
Average time used on delete : 0.0207


found out:
    - """


