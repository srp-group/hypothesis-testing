import csv
import os

class Logger:
    def __init__(self, dataset_name:str, logging_dir : str) -> None:
        self.dataset_name = dataset_name
        self.logging_dir = logging_dir                
    
    def log_primary_results(self, test_loss_list: list, best_dropout_rate_list: list, best_l2_reg_list:list, test_accuracy_list:list) -> str:
        iterations = range(len(test_loss_list))

        data = list(zip(iterations,best_l2_reg_list,best_dropout_rate_list,test_loss_list,test_accuracy_list))
        
        filename = os.path.join(self.logging_dir, f'results_{self.dataset_name}.csv')
        filename = os.path.normpath(filename)
        # Write the data to a CSV file
        with open(filename, 'w', newline='') as csvfile:
            # Create a CSV writer object
            csvwriter = csv.writer(csvfile)
            
            # Write the header
            csvwriter.writerow(['iterations', 'L2_value', 'drop_value', 'loss', 'accuracy'])
            
            # Write the data
            csvwriter.writerows(data)
        
        return filename