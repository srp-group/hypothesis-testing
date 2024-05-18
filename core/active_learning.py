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
        self.get_config()
        # Initialize the core components
        self.pool = core.Pool(dataset_name=dataset_name, random_seed=random_seed,
                database_config=self.database_config, default_config=self.default_config)
        self.acquisition_function = acquisitions.Random(self.pool)
        self.clf = core.Classifier(pool=self.pool)
        # initialize the visualization and logging components
        self.set_logging_dir()
        self.visualizer = core.Visualization(dataset_name=self.dataset_name, should_show_the_plot=bool(int(self.default_config['should_show_the_plot'])), logging_dir=self.logging_dir)
        self.data_logger = core.Logger(dataset_name=self.dataset_name, logging_dir=self.logging_dir)


    def get_config(self) -> None:
        root_dir = os.getcwd()
        params_path = os.path.join(root_dir, 'params.ini')
        params_path = os.path.normpath(params_path)
        config = ConfigParser()
        config.read(params_path)
        self.database_config = config[self.dataset_name.upper()]
        self.default_config = config['DEFAULT']
    
    def set_logging_dir(self) -> None:
        current_time = datetime.datetime.now()
        date_path = current_time.strftime("%Y-%m-%d_%H-%M-%S") 
        root_dir = os.getcwd()
        self.logging_dir = os.path.join(root_dir, 'logs', date_path)
        self.logging_dir = os.path.normpath(self.logging_dir)
        # Check if the folder exists    
        if not os.path.exists(self.logging_dir):
            # Create the folder
            os.makedirs(self.logging_dir)

    def run(self) -> None:
        
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