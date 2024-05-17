from torch.utils.data import Dataset
import numpy as np
from configparser import SectionProxy
import os
class ProteinDataset(Dataset):
    def __init__(self, dataset_config: SectionProxy) -> None:
        super().__init__()
        current_file_path = os.path.abspath(__file__)
        data_path = os.path.join(os.path.dirname(current_file_path), '..', dataset_config['data'])
        data_path = os.path.normpath(data_path)
        with np.load(data_path, allow_pickle=True) as file:
            self.x = file["x"].astype(np.float32)
            self.y = file["y"].astype(np.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple:
        return self.x[idx], self.y[idx]