from datasets import DnaDataset
from torch.utils.data import DataLoader

class ActiveLearning:
    def __init__(self, dataset_name) -> None:
        dataset = DnaDataset()