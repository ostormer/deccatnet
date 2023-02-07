from EdfReader import EDF_reader
import mne
import os
import pandas as pd

def fix_columns(row):
    return list(row.name[0].split()[1:])




pd.set_option('display.max_rows', 500)
path = '../datasets/UCI-ML-repository/SMNI_CMI_TRAIN/co2a0000364'
dir_list = os.listdir(path)
print(dir_list)
for file in dir_list:
    print(file)
    file_path = path + '/' + file
    df = pd.read_csv(file_path)
    n_trials = int(df.iloc[0].name[0].split()[1])
    n_chans = int(df.iloc[0].name[1].split()[0])

    n_samples = df.iloc[0][0].split()[0]
    n_post_stim_samples = df.iloc[0][0].split()[2]

    freq = df.iloc[1].name[0][1:]
    trial = int(df.iloc[2].name[1][-1:])
    df = pd.DataFrame(df.iloc[4:, :].apply(lambda row: fix_columns(row), axis=1).to_list(),
                      columns=['channel', 'trial_number', 'value'])

    df_channels = df[~df.trial_number.map(lambda x: x.isdigit())]
    df_channels = df_channels.set_index('value')['channel']
    channels_dict = df_channels.to_dict()

    df = df[df.trial_number.map(lambda x: x.isdigit())]




#raw = mne.io.RawArray(data,)

