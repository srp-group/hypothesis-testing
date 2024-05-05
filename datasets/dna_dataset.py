from torch.utils.data import Dataset
from configparser import ConfigParser
import os
import requests
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Read in the configurations from the params.ini file
config = ConfigParser()
config.read('params.ini')
dna_config = config['DNA']

class DnaDataset(Dataset):

    def get_and_preprocess_data(self):
        self.file_path = f"data/{dna_config[self.split_name]}"
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w') as f:
                r = requests.get(dna_config[f"{self.split_name}_url"])
                f.writelines(r.content.decode("utf-8"))
        x, y  = load_svmlight_file(self.file_path, n_features=int(dna_config['n_features']))
        x = np.asarray(x.todense(), dtype=np.float32)
        y = y.reshape(-1, 1).astype(np.float32)
        # pre-process the labels
        encoder = OneHotEncoder(sparse_output=False)
        y = encoder.fit_transform(y)
        self.x, self.y = x, y

    def __init__(self, split_name):
        super().__init__()
        self.split_name = split_name
        self.get_and_preprocess_data()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]