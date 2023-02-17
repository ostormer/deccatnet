from EdfReader import EDF_reader
import os
path = '../datasets/TUH/normal/01_tcp_ar'
dir_list = os.listdir(path)
print(dir_list)
for file in dir_list:
    file_path = path + '/' + file
    test = EDF_reader(file_path)
    test.plot_file()
