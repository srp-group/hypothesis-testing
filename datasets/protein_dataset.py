from torch.utils.data import Dataset
import numpy as np
from configparser import SectionProxy

class ProteinDataset(Dataset):
    def __init__(self, dataset_config: SectionProxy) -> None:
        super().__init__()
        with np.load(dataset_config['data'], allow_pickle=True) as file:
            self.x = file["x"].astype(np.float32)
            self.y = file["y"].astype(np.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple:
        return self.x[idx], self.y[idx]