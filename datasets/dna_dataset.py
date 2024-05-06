from torch.utils.data import Dataset
from configparser import ConfigParser
import numpy as np

# Read in the configurations from the params.ini file
config = ConfigParser()
config.read('params.ini')
dna_config = config['DNA']

class DnaDataset(Dataset):
    def __init__(self):
        super().__init__()
        with np.load(dna_config['data'], allow_pickle=True) as file:
            self.x = file["x"].astype(np.float32)
            self.y = file["y"].astype(np.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]