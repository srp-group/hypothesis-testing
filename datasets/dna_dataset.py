from torch.utils.data import Dataset
from configparser import ConfigParser

# Read in the configurations from the params.ini file
config = ConfigParser()
config.read('params.ini')
dna_config = config['DNA']

class DnaDataset(Dataset):
    def ensure_data(self) -> None:
        pass

    def __init__(self, split_name):
        super().__init__()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]