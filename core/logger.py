import csv
import datetime

class Logger:
    def __init__(self) -> None:
        pass
    
    def log_primary_results(self, test_loss_list: list, best_dropout_rate_list: list, best_l2_reg_list:list, test_accuracy_list:list, dataset_name:str = '') -> None:
        iterations = range(len(test_loss_list))

        data = list(zip(iterations,best_l2_reg_list,best_dropout_rate_list,test_loss_list,test_accuracy_list))

        current_time = datetime.datetime.now()

        # Specify the file name
        filename = "logs/results_" + dataset_name + "_" + current_time.strftime("%Y-%m-%d_%H-%M-%S") + ".csv"

        # Write the data to a CSV file
        with open(filename, 'w', newline='') as csvfile:
            # Create a CSV writer object
            csvwriter = csv.writer(csvfile)
            
            # Write the header
            csvwriter.writerow(['iterations', 'L2_value', 'drop_value', 'loss', 'accuracy'])
            
            # Write the data
            csvwriter.writerows(data)