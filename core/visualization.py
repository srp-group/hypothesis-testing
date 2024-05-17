import matplotlib.pyplot as plt
import csv
import datetime
import main

class Visualization:
    def __init__(self) -> None:
        pass
    
    def plot_primary_results(self, test_loss_list: list, best_dropout_rate_list: list, best_l2_reg_list:list, test_accuracy_list:list, dataset_name:str = '') -> None:
        
        iterations = range(len(test_loss_list))
        data = list(zip(iterations,best_l2_reg_list,best_dropout_rate_list,test_loss_list,test_accuracy_list))

        current_time = datetime.datetime.now()

        # Specify the file name
        filename = "results_" + dataset_name + "_" + current_time.strftime("%Y-%m-%d_%H-%M-%S") + ".csv"

        # Write the data to a CSV file
        with open(filename, 'w', newline='') as csvfile:
            # Create a CSV writer object
            csvwriter = csv.writer(csvfile)
            
            # Write the header
            csvwriter.writerow(['iterations', 'L2_value', 'drop_value', 'loss', 'accuracy'])
            
            # Write the data
            csvwriter.writerows(data)

        plt.figure(figsize=(5, 20))  # Adjusted figure size to better fit 4 plots

        # Plot for test loss
        plt.subplot(4, 1, 1)  # Now 4 rows, 1 column, 1st subplot
        plt.plot(iterations, test_loss_list, label='Test Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Test Loss')
        plt.title('Test Loss per Iteration')
        plt.legend()

        # Plot for test accuracy
        plt.subplot(4, 1, 4)  # Added 4th subplot for test accuracy
        plt.plot(iterations, test_accuracy_list, label='Test Accuracy')
        plt.xlabel('Iteration')
        plt.ylabel('Test Accuracy')
        plt.title('Test Accuracy per Iteration')
        plt.legend()

        # Plot for best dropout rate
        plt.subplot(4, 1, 2)  # Now 4 rows, 1 column, 2nd subplot
        plt.plot(iterations, best_dropout_rate_list, label='Best Dropout Rate')
        plt.xlabel('Iteration')
        plt.ylabel('Dropout Rate')
        plt.title('Dropout Rate per Iteration')
        plt.legend()

        # Plot for best l2 reg
        plt.subplot(4, 1, 3)  # Now 4 rows, 1 column, 3rd subplot
        plt.plot(iterations, best_l2_reg_list, label='Best L2 Reg')
        plt.xlabel('Iteration')
        plt.ylabel('L2 Regularization')
        plt.title('L2 Regularization per Iteration')
        plt.legend()

        plt.tight_layout()
        plt.show()