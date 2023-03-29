import DECCaTNet.DECCaTNet_model.contrastive_framework as cf
from DECCaTNet.DECCaTNet_model import PathDataset
import pickle
if __name__ == '__main__':

    # idx_list = run_preprocess('preprocessing/preprocessing_anakin.yaml')
    # print(idx_list[:100])

    with open('../datasets/TUH/pickles/split_idx_list.pkl', 'rb') as f:
        idx_list = pickle.load(f)

    path = '../datasets/TUH/preprocessed/step_2'
    dataset = PathDataset(ids_to_load=idx_list,path=path, preload=False)

    cf.pre_train_model(dataset=dataset, batch_size=16, train_split=0.7, save_freq=1, shuffle=True,
                       trained_model_path=None, temperature=1, learning_rate=0.01, weight_decay=0.01,
                       num_workers=14, max_epochs=5, batch_print_freq=5, save_dir_model='models', model_file_name='test',
                       model_params=None, time_process=True)


    # tuh_eeg description columns: ['year', 'month', 'day', 'path', 'version', 'subject', 'session', 'segment', 'age', 'gender']
    # tuh_eeg description columns: ['year', 'month', 'day', 'path', 'version', 'subject', 'session', 'segment', 'age', 'gender', 'train', 'pathological']

    # Batch size    64, GPU
    # Average time used on batch :  0.002944
    # Average time used on to_device :  0.000106
    # Average time used on encoding :  0.000662
    # Average time used on loss_calculation :  0.039110
    # Average time used on loss_update :  0.007310
    # Average time used on delete :  0.001943

    # Batch size 16, CPU
    # Average time used on batch :  0.000413
    # Average time used on to_device :  0.000001
    # Average time used on encoding :  0.011763
    # Average time used on loss_calculation :  0.004522
    # Average time used on loss_update :  0.022966
    # Average time used on delete :  0.001652

    # Batch size 16, GPU
    # Average time used on batch :  0.001025
    # Average time used on to_device :  0.000151
    # Average time used on encoding :  0.001575
    # Average time used on loss_calculation :  0.010637
    # Average time used on loss_update :  0.003455
    # Average time used on delete :  0.000808