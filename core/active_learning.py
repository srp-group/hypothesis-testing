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
        self.visualizer = core.Visualization()

    def run(self) -> None:

        test_loss_list = []
        best_dropout_rate_list = []
        best_l2_reg_list = []

        for i in range(int(self.pool.dataset_config['budget'])):
            print(f"============ iteration: {i+1} ============")
            print(f"current number of labeled data: {len(self.pool.idx_label)}")
            
            best_dropout_rate, best_l2_reg, best_val_loss = self.clf.tune()
            test_loss, test_metrics = self.clf.test(best_l2_reg, best_dropout_rate)
            
            test_loss_list.append(test_loss)
            best_dropout_rate_list.append(best_dropout_rate)  
            best_l2_reg_list.append(best_l2_reg)

            self.pool.add_labeled_data(self.acquisition_function.query())

        print(f"============ Final Results ============")
        
        print(f"test loss: {test_loss_list}")
        print(f"best dropout rate: {best_dropout_rate_list}")
        print(f"best l2 reg: {best_l2_reg_list}")
        
        self.visualizer.plot_primary_results(test_loss_list, best_dropout_rate_list, best_l2_reg_list)