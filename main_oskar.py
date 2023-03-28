from preprocessing.preprocess import run_preprocess
import DECCaTNet.contrastive_framework as cf
from DECCaTNet.custom_dataset import PathDataset
import pickle
if __name__ == '__main__':

    windowed_ds = run_preprocess('preprocessing/preprocessing_abnormal.yaml')
    print(windowed_ds.datasets[:100])

    # with open('datasets/TUH/pickles/split_idx_list.pkl','rb') as f:
    #     ids_to_load = pickle.load(f)

    # path = 'datasets/TUH/preprocessed/step_2'
    # dataset = PathDataset(ids_to_load=ids_to_load,path=path, preload=False)
    #
    # cf.pre_train_model(dataset=dataset, batch_size=16,train_split=0.7,save_freq=1,shuffle=True,
    #                    trained_model_path=None,temperature=1,learning_rate=0.01,weight_decay=0.01,
    #                    num_workers=8,max_epochs=4,batch_print_freq=5,save_dir_model='models', model_file_name='test',
    #                    model_params=None, time_process=True)


    # batch size 16, GPU
    # Average time used on batch :  0.000026
    # Average time used on to_device :  0.000018
    # Average time used on encoding :  0.000528
    # Average time used on loss_calculation :  0.017235
    # Average time used on loss_update :  0.000308
    # Average time used on delete :  0.000625