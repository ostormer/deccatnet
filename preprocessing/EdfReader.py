import mne

class EDF_reader:

    def __init__(self, filename):
        self.edf_file = mne.io.read_raw_edf(filename)


    def raw_edf(self):
        return self.edf_file.get_data()

    def get_header(self):
        return self.edf_file.info

    def get_time(self):
        return self.edf_file.info['meas_date']

    def plot_file(self):
        self.edf_file.plot(block=True)


