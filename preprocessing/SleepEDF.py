from EdfReader import EDF_reader
import os
path = '../datasets/SleepEDF-subset'
dir_list = os.listdir(path)
print(dir_list)
for file in dir_list:
    file_path = path + '/' + file
    test = EDF_reader(file_path)
    try:
        test.plot_file()
    except:
        print(test.get_header())