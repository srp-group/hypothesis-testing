from datasets import DnaDataset
from torch.utils.data import DataLoader
import core
import acquisitions
import os
from configparser import ConfigParser
import datetime

class ActiveLearning:
    def __init__(self, dataset_name: str, random_seed: int) -> None:
        self.dataset_name = dataset_name
        # Load the configuration
        config = ConfigParser()
        current_file_path = os.path.abspath(__file__)
        params_path = os.path.join(os.path.dirname(current_file_path), '..', 'params.ini')
        params_path = os.path.normpath(params_path)
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"The configuration file {params_path} does not exist.")
        config.read(params_path)
        self.database_config = config[self.dataset_name.upper()]
        self.default_config = config['DEFAULT']
        # Initialize the core components
        self.pool = core.Pool(dataset_name=dataset_name, random_seed=random_seed,
                database_config=self.database_config, default_config=self.default_config)
        self.acquisition_function = acquisitions.Random(self.pool)
        self.clf = core.Classifier(pool=self.pool)
        # initialize the visualization and logging components
        current_time = datetime.datetime.now()
        date_path = current_time.strftime("%Y-%m-%d_%H-%M-%S") 
        self.visualizer = core.Visualization(dataset_name=self.dataset_name, date_path=date_path, should_show_the_plot=bool(int(self.default_config['should_show_the_plot'])))
        self.data_logger = core.Logger(dataset_name=self.dataset_name, date_path=date_path)

    def run(self) -> None:
        current_file_path = os.path.abspath(__file__)
        root_dir = os.path.join(os.path.dirname(current_file_path), '..')
        root_dir = os.path.normpath(root_dir)
        print(root_dir)
        return
        test_loss_list = []
        best_dropout_rate_list = []
        best_l2_reg_list = []
        test_accuracy_list = []

        for i in range(int(self.pool.max_budget)):
            print(f"============ iteration: {i+1} ============")
            print(f"current number of labeled data: {len(self.pool.idx_label)}")
            
            best_dropout_rate, best_l2_reg, best_val_loss = self.clf.tune()
            test_loss, test_metrics = self.clf.test(best_l2_reg, best_dropout_rate)
            
            test_loss_list.append(test_loss)
            best_dropout_rate_list.append(best_dropout_rate)  
            best_l2_reg_list.append(best_l2_reg)
            test_accuracy_list.append(test_metrics.item())

            self.pool.add_labeled_data(self.acquisition_function.query())

        print(f"============ Final Results ============")
        
        print(f"test loss: {test_loss_list}")
        print(f"best dropout rate: {best_dropout_rate_list}")
        print(f"best l2 reg: {best_l2_reg_list}")
        
        
        file_path = self.data_logger.log_primary_results(test_loss_list, best_dropout_rate_list, best_l2_reg_list, test_accuracy_list)
        self.visualizer.plot_results(file_path)
        self.visualizer.plot_primary_results(test_loss_list, best_dropout_rate_list, best_l2_reg_list, test_accuracy_list)