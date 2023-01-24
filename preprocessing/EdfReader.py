import mne
from datetime import datetime


class EDF_reader:

    def __init__(self, filename):
        self.edf_file = mne.io.read_raw_edf(filename)


    def raw_edf(self):
        return self.edf_file.get_data()

    def get_header(self):
        return self.edf_file.info

    def get_time(self):
        return self.edf_file.info['meas_date']

#test = EDF_reader('../datasets/TUH/normal/01_tcp_ar/aaaaaaav_s004_t000.edf')
#print(test.get_header())
