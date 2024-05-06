from configparser import ConfigParser
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset
from datasets import DnaDataset

config = ConfigParser()
config.read('params.ini')

class Pool():
    def __init__(self, dataset_name) -> None:
        self.dataset_config = config[dataset_name.upper()]
        self.default_config = config['DEFAULT']
        # setting the dataset
        if dataset_name == 'dna':
            self.dataset = DnaDataset()
        # setting the indecies
        self.idx_abs = np.arange(len(self.dataset)) # absolute indecies
        