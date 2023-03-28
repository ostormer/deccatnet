from preprocessing.preprocess import run_preprocess
import DECCaTNet.contrastive_framework as cf
from DECCaTNet.custom_dataset import PathDataset
import pickle
if __name__ == '__main__':

    # idx_list = run_preprocess('preprocessing/preprocessing_oskar.yaml')
    # print(idx_list[:100])

    with open('datasets/TUH/pickles/split_idx_list.pkl','rb') as f:
        ids_to_load = pickle.load(f)

    path = 'datasets/TUH/preprocessed/step_2'
    dataset = PathDataset(ids_to_load=ids_to_load,path=path, preload=False)

    cf.pre_train_model(dataset=dataset, batch_size=16,train_split=0.7,save_freq=1,shuffle=True,
                       trained_model_path=None,temperature=1,learning_rate=0.01,weight_decay=0.01,
                       num_workers=8,max_epochs=4,batch_print_freq=5,save_dir_model='models', model_file_name='test',
                       model_params=None, time_process=True)


    # tuh_eeg description columns: ['year', 'month', 'day', 'path', 'version', 'subject', 'session', 'segment', 'age', 'gender']
    # tuh_eeg description columns: ['year', 'month', 'day', 'path', 'version', 'subject', 'session', 'segment', 'age', 'gender', 'train', 'pathological']