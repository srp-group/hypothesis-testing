from torch.utils.data import Dataset
import numpy as np
from configparser import SectionProxy
import os
import pandas as pd
class TwoMoonsDataset(Dataset):
    def __init__(self, dataset_config: SectionProxy) -> None:
        super().__init__()
        current_file_path = os.path.abspath(__file__)
        data_path = os.path.join(os.path.dirname(current_file_path), '..', dataset_config['data'])
        data_path = os.path.normpath(data_path)
        df = pd.read_excel(data_path)
        df_Y = df['Label']
        df_X = df.drop('Label', axis=1)
        self.x = df_X.values
        self.y = df_Y.values

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple:
        return self.x[idx], self.y[idx]