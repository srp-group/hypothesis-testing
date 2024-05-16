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

        iteration = []
        test_loss_list = []
        best_dropout_rate_list = []
        best_l2_reg_list = []

        for i in range(754):
            print(f"============ iteration: {i+1} ============")
            print(f"current number of labeled data: {len(self.pool.idx_label)}")
            
            best_dropout_rate, best_l2_reg, best_val_loss = self.clf.tune()
            test_loss, test_metrics = self.clf.test(best_l2_reg, best_dropout_rate)
            
            iteration.append(i)
            test_loss_list.append(test_loss)
            best_dropout_rate_list.append(best_dropout_rate)  
            best_l2_reg_list.append(best_l2_reg)

            self.pool.add_labeled_data(self.acquisition_function.query())

        print(f"============ Final Results ============")
        
        print(f"test loss: {test_loss_list}")
        print(f"best dropout rate: {best_dropout_rate_list}")
        print(f"best l2 reg: {best_l2_reg_list}")

        import matplotlib.pyplot as plt

        # Assuming test_loss_list is defined and accessible
        iterations = range(len(test_loss_list))

        plt.figure(figsize=(15, 5))

        # Plot for test loss
        plt.subplot(1, 3, 1)  # 1 row, 3 columns, 1st subplot
        plt.plot(iterations, test_loss_list, label='Test Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Test Loss')
        plt.title('Test Loss per Iteration')
        plt.legend()

        # Plot for best dropout rate
        plt.subplot(1, 3, 2)  # 1 row, 3 columns, 2nd subplot
        plt.plot(iterations, best_dropout_rate_list, label='Best Dropout Rate', linestyle='--')
        plt.xlabel('Iteration')
        plt.ylabel('Dropout Rate')
        plt.title('Dropout Rate per Iteration')
        plt.legend()

        # Plot for best l2 reg
        plt.subplot(1, 3, 3)  # 1 row, 3 columns, 3rd subplot
        plt.plot(iterations, best_l2_reg_list, label='Best L2 Reg', linestyle='-.')
        plt.xlabel('Iteration')
        plt.ylabel('L2 Regularization')
        plt.title('L2 Regularization per Iteration')
        plt.legend()

        plt.tight_layout()
        plt.show()
            