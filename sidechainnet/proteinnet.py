from glob import glob
import os

class ProteinNet(object):
    """
    Defines a wrapper for interacting with a ProteinNet dataset.
    """

    def __init__(self, raw_dir, training_set):
        self.raw_dir = raw_dir
        self.training_set = training_set

    def parse_raw_data(self):
        input_files = glob(os.path.join(self.raw_dir, "raw/*[!.ids]"))


