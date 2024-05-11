from datasets import DnaDataset
from torch.utils.data import DataLoader
import core
import acquisitions
import models

class ActiveLearning:
    def __init__(self, dataset_name: str, random_seed: int) -> None:
        self.pool = core.Pool(dataset_name=dataset_name, random_seed=random_seed)
        self.acquisition_function = acquisitions.Random(self.pool)
        self.clf = core.Classifier(pool=self.pool)

    def run(self) -> None:
        best_dropout_rate, best_l2_reg, best_val_loss = self.clf.tune()
        test_loss, test_metrics = self.clf.test(best_l2_reg, best_dropout_rate)
        print(test_loss)
