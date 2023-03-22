from preprocessing.run_preprocess import run_preprocess
import pickle
if __name__ == '__main__':

    with open('datasets/TUH/TUH-pickles/windowed_ids.pkl','rb') as f:
        ids_to_load = pickle.load(f)

    print(ids_to_load)