from configparser import ConfigParser
from torch.utils.data import DataLoader, Subset, ConcatDataset

config = ConfigParser()
config.read('params.ini')

class Pool():
    def __init__(self, dataset_name) -> None:
        self.dataset_config = config[dataset_name]
        