from preprocessing.run_preprocess import run_preprocess
from SeqCLR.custom_dataset import PathDataset
import pickle
import torch
if __name__ == '__main__':

    with open('datasets/TUH/TUH-pickles/windowed_ids.pkl','rb') as f:
        ids_to_load = pickle.load(f)
    path = 'datasets/TUH/preprocessed/step_2'
    dataset = PathDataset(ids_to_load=ids_to_load,path=path, preload=True)

    loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)

    for aug_1, aug_2, sample in loader:
        # transfer to GPU or CUDA
        print(aug_1.shape, aug_2.shape, sample.shape)