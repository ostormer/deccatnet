from mne.io import read_raw_edf

class EDF_reader:

    def __init__(self, filename):
        self.edf_file = read_raw_edf(filename)
        # TODO: MNE handles channels with different sample rates poorly,
        # blockwise resampling to the highest sample rate
        # Maybe handle this by detecting and dropping channels
        # with weird sample rates

    def raw_edf(self):
        return self.edf_file.get_data()

    def get_header(self):
        return self.edf_file.info

    def get_time(self):
        return self.edf_file.info['meas_date']

    def plot_file(self):
        self.edf_file.plot(block=True)


