from preprocessing.preprocess import run_preprocess
# from SeqCLR.custom_dataset import PathDataset
import pickle
import torch
if __name__ == '__main__':

    idx_list = run_preprocess('preprocessing/preprocessing_oskar.yaml', stop_idx=50)
    print(idx_list[:50])

    # ds = run_preprocess('preprocessing/preprocessing_abnormal.yaml', is_downstream_ds=True)
    # with open('datasets/TUH/pickles_abnormal/preproc1.pkl', 'rb') as fid:
    #     ds = pickle.load(fid)
    # print(ds.description)
    # print(list(ds.description.columns.values))

    # tuh_eeg description columns: ['year', 'month', 'day', 'path', 'version', 'subject', 'session', 'segment', 'age', 'gender']
    # tuh_eeg description columns: ['year', 'month', 'day', 'path', 'version', 'subject', 'session', 'segment', 'age', 'gender', 'train', 'pathological']