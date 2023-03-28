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

    cf.pre_train_model(dataset=dataset, batch_size=8,train_split=0.7,save_freq=10,shuffle=True,
                       trained_model_path=None,temperature=1,learning_rate=0.01,weight_decay=0.01,
                       num_workers=2,max_epochs=10,batch_print_freq=5,save_dir_model='models', model_file_name='test',
                       model_params=None, time_process=True)

"""With batch_size = 8 on Styrks computer over an entire epoch: 
Average time used on batch : 0.0002845612582781457
Average time used on to_device : 0.0
Average time used on encoding : 0.06299151490066225
Average time used on loss_calculation : 0.010257139900662252
Average time used on loss_update : 0.1312991514900662
Average time used on delete : 0.0006984685430463576

batch_size = 32
Average time used on batch : 0.007056451612903226
Average time used on to_device : 0.0
Average time used on encoding : 2.004032258064516
Average time used on loss_calculation : 1.387600806451613
Average time used on loss_update : 5.786290322580645
Average time used on delete : 0.08770161290322581

batch_size = 128


found out:
    - data is laoded once it is sent to device,not before that
    - The average time used on different processes increases through the entire epoch """


