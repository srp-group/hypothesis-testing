from torch.utils.data import Dataset
import numpy as np
from configparser import SectionProxy
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
class TwoMoonsDataset(Dataset):
    def __init__(self, dataset_config: SectionProxy) -> None:
        super().__init__()
        current_file_path = os.path.abspath(__file__)
        data_path = os.path.join(os.path.dirname(current_file_path), '..', dataset_config['data'])
        data_path = os.path.normpath(data_path)
        df = pd.read_excel(data_path)
        df_Y = df['Label']
        df_X = df.drop('Label', axis=1)
        self.x = df_X.values.astype(np.float32)
        self.x = MinMaxScaler().fit_transform(self.x)
        self.y = df_Y.values.astype(np.int32).reshape(-1, 1)
        self.y = OneHotEncoder(sparse_output=False).fit_transform(self.y)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple:
        return self.x[idx], self.y[idx]