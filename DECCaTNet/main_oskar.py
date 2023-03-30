import pandas
import yaml
import pickle

from DECCaTNet.preprocessing.preprocess import run_preprocess
from DECCaTNet.DECCaTNet_model.fine_tuning import run_fine_tuning

if __name__ == '__main__':

    # windowed_ds = run_preprocess('preprocessing/preprocessing_abnormal.yaml')
    with open('../datasets/TUH/pickles_abnormal/windowed_ds.pkl', 'rb') as f:
        windowed_ds = pickle.load(f)

    with open('DECCaTNet_model/configs/fine_tune_test.yaml', 'r') as fid:
        params = yaml.safe_load(fid)

    pandas.set_option('display.max_columns', 20)
    windowed_ds.description.head()
    run_fine_tuning(windowed_ds, params)
    # path = 'datasets/TUH/preprocessed/step_2'
    # dataset = PathDataset(ids_to_load=ids_to_load,path=path, preload=False)
    #
    # cf.pre_train_model(dataset=dataset, batch_size=16,train_split=0.7,save_freq=1,shuffle=True,
    #                    trained_model_path=None,temperature=1,learning_rate=0.01,weight_decay=0.01,
    #                    num_workers=8,max_epochs=4,batch_print_freq=5,save_dir_model='models', model_file_name='test',
    #                    model_params=None, time_process=True)

