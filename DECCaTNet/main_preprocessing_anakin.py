import pickle

import yaml

# from DECCaTNet.DECCaTNet_model.contrastive_framework import pre_train_model
# from DECCaTNet.DECCaTNet_model.custom_dataset import PathDataset
# from DECCaTNet.DECCaTNet_model.fine_tuning import run_fine_tuning
from preprocessing.preprocess import run_preprocess

if __name__ == '__main__':
    ds = run_preprocess('DECCaTNet/preprocessing/preprocessing_anakin.yaml')


    # with open('DECCaTNet_model/configs/fine_tune_test.yaml', 'r') as fid:
    #     params = yaml.safe_load(fid)

    # pre_train_path = '../datasets/TUH/preprocessed/step_2'
    # with open('../datasets/TUH/pickles/split_idx_list.pkl', 'rb') as fid:
    #     pre_train_ids = pickle.load(fid)

    # dataset = PathDataset(ids_to_load=pre_train_ids, path=pre_train_path, preload=False)
    # pre_train_model(dataset=dataset, batch_size=16, train_split=0.7, save_freq=1, shuffle=True,
    #                 trained_model_path=None, temperature=1, learning_rate=0.01, weight_decay=0.01,
    #                 num_workers=8, max_epochs=2, batch_print_freq=4, save_dir_model='models', model_file_name='test',
    #                 model_params=None, time_process=True)

    # pandas.set_option('display.max_columns', 20)
    # windowed_ds.description.head()
    # run_fine_tuning(fine_tune_ds, params)
    #
