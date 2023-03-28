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

    cf.pre_train_model(dataset=dataset, batch_size=32,train_split=0.7,save_freq=10,shuffle=True,
                       trained_model_path=None,temperature=1,learning_rate=0.01,weight_decay=0.01,
                       num_workers=2,max_epochs=10,batch_print_freq=5,save_dir_model='models', model_file_name='test',
                       model_params=None, time_process=True)

"""With batch_size = 8 on Styrks computer over an entire epoch: 
Average time used on batch : 0.0012807377049180327
Average time used on to_device : 0.0
Average time used on encoding : 0.5258709016393442
Average time used on loss_calculation : 0.0932377049180328
Average time used on loss_update : 1.1303790983606556
Average time used on delete : 0.007684426229508197

batch_size = 32


found out:
    - data is laoded once it is sent to device,not before that
    - The average time used on different processes increases through the entire epoch """


