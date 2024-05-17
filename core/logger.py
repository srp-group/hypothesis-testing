import csv
import os

class Logger:
    def __init__(self, dataset_name:str, date_path: str) -> None:
        self.dataset_name = dataset_name
        self.date_path = date_path
    
    def log_primary_results(self, test_loss_list: list, best_dropout_rate_list: list, best_l2_reg_list:list, test_accuracy_list:list) -> str:
        iterations = range(len(test_loss_list))

        data = list(zip(iterations,best_l2_reg_list,best_dropout_rate_list,test_loss_list,test_accuracy_list))

        
        
        root_dir = os.path.abspath(__file__).replace("core\\logger.py", "") 
        # Check if the folder exists
        if not os.path.exists(f"{root_dir}\\logs\\{self.date_path}\\"):
            # Create the folder
            os.makedirs(f"{root_dir}\\logs\\{self.date_path}\\")
        
        # Specify the file name
        
        
        filename = f"{root_dir}\\logs\\{self.date_path}\\results_" + self.dataset_name + ".csv"

        # Write the data to a CSV file
        with open(filename, 'w', newline='') as csvfile:
            # Create a CSV writer object
            csvwriter = csv.writer(csvfile)
            
            # Write the header
            csvwriter.writerow(['iterations', 'L2_value', 'drop_value', 'loss', 'accuracy'])
            
            # Write the data
            csvwriter.writerows(data)
        
        return filename