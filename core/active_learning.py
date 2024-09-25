from datasets import DnaDataset
from torch.utils.data import DataLoader
import core
import acquisitions
import os
from configparser import ConfigParser
import datetime
import time
from datetime import timedelta
class ActiveLearning:
    def __init__(self, random_seed, dataset_name, acq_func, model_name) -> None:
        self.model_name = model_name
        self.random_seed = random_seed
        self.dataset_name = dataset_name
        self.acq_func = acq_func
        # Load the configuration
        self.get_config()       
        # Initialize the core components
        self.pool = core.Pool(dataset_name=self.dataset_name, random_seed=self.random_seed,
                database_config=self.database_config, default_config=self.default_config)
        self.clf = core.Classifier(pool=self.pool, model_name=self.model_name)
        if self.acq_func.lower() == 'random':
            self.acquisition_function = acquisitions.Random(self.pool)
        elif self.acq_func.lower() == 'entropy': # currently only works for MLP
            self.acquisition_function = acquisitions.Entropy(self.pool, self.clf)
        # initialize the visualization and logging components
        self.set_logging_dir()
        self.visualizer = core.Visualization(dataset_name=self.dataset_name, should_show_the_plot=bool(int(self.default_config['should_show_the_plot'])), logging_dir=self.logging_dir, model_name=self.model_name)
        self.data_logger = core.Logger(dataset_name=self.dataset_name, logging_dir=self.logging_dir)


    def get_config(self) -> None:
        current_file_path = os.path.abspath(__file__)
        params_path = os.path.join(os.path.dirname(current_file_path), "..", 'params.ini')
        params_path = os.path.normpath(params_path)
        config = ConfigParser()
        config.read(params_path)
        self.default_config = config['DEFAULT']
        self.database_config = config[self.dataset_name.upper()]

    def set_logging_dir(self) -> None:
        current_time = datetime.datetime.now()
        date_path = current_time.strftime("%Y-%m-%d_%H-%M-%S") 
        current_file_path = os.path.abspath(__file__)
        self.logging_dir = os.path.join(os.path.dirname(current_file_path), "..", 'logs', date_path)
        self.logging_dir = os.path.normpath(self.logging_dir)
        # Check if the folder exists    
        if not os.path.exists(self.logging_dir):
            # Create the folder
            os.makedirs(self.logging_dir)

    def run(self) -> None:
        allgemein_start_time = time.time()
        self.test_loss_list = []
        self.best_dropout_rate_list = []
        self.best_l2_reg_list = []
        self.test_accuracy_list = []

        al_iterations = int(self.pool.max_budget) // int(self.default_config['random_batch_size'])
        for i in range(al_iterations):
            start_time = time.time()

            print(f"============ iteration: {i+1} ============")
            print(f"current number of newly labeled data: {len(self.pool.idx_newly_labeled)}")

            best_dropout_rate, best_l2_reg, best_val_loss = self.clf.tune()
            test_loss, test_metrics, best_model = self.clf.test(best_l2_reg, best_dropout_rate)
            
            self.test_loss_list.append(test_loss)
            self.best_dropout_rate_list.append(best_dropout_rate)  
            self.best_l2_reg_list.append(best_l2_reg)
            self.test_accuracy_list.append(test_metrics.item())

            for j in range(int(self.default_config['random_batch_size'])):
                self.pool.add_newly_labeled_data(self.acquisition_function.query(best_model))
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            formatted_time = str(timedelta(seconds=int(elapsed_time)))
            print(f"iteration time: {formatted_time}")
            print(f"iteration test_loss: {test_loss}")
            print(f"iteration test_accuracy: {test_metrics.item()}")
            print(f"iteration best dropout rate: {best_dropout_rate}")
            print(f"iteration best l2 reg: {best_l2_reg}")
            self.visualizer.plot_primary_results(self.test_loss_list, self.best_dropout_rate_list, self.best_l2_reg_list, self.test_accuracy_list)

        print(f"============ Final Results ============")
        
        allgemein_end_time = time.time()
        allgemein_elapsed_time = allgemein_end_time - allgemein_start_time
        self.allgemein_formatted_time = str(timedelta(seconds=int(allgemein_elapsed_time)))
        print(f"time spent: {self.allgemein_formatted_time}")

        
        print(f"test loss: {self.test_loss_list}")
        print(f"best dropout rate: {self.best_dropout_rate_list}")
        print(f"best l2 reg: {self.best_l2_reg_list}")
        
        
        file_path = self.data_logger.log_primary_results(self.test_loss_list, self.best_dropout_rate_list, self.best_l2_reg_list, self.test_accuracy_list)
        self.log_params()
        self.visualizer.plot_results(file_path)
    
    def log_params(self) -> None:
        results_dict = dict(self.database_config)
        results_dict['dataset_name'] = self.dataset_name
        results_dict['acquisition_function'] = self.acq_func
        results_dict['model_name'] = self.model_name
        if self.allgemein_formatted_time:
            results_dict['time_spent'] = self.allgemein_formatted_time
        results_dict['random_seed'] = self.pool.random_seed
        if self.best_dropout_rate_list:
            results_dict['last_dropout_rate'] = self.best_dropout_rate_list[-1]
        if self.best_l2_reg_list:
            results_dict['last_l2_reg'] = self.best_l2_reg_list[-1]
        results_dict['random_batch_size'] = self.default_config['random_batch_size']
        self.data_logger.log_params(results_dict)
