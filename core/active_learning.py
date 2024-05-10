from datasets import DnaDataset
from torch.utils.data import DataLoader
import core
class ActiveLearning:
    def __init__(self, dataset_name, random_seed) -> None:
        self.pool = core.Pool(dataset_name=dataset_name, random_seed=random_seed)
        