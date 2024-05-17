import matplotlib.pyplot as plt
import os

class Visualization:
    def __init__(self, date_path: str, should_show_the_plot: bool, dataset_name: str) -> None:
        self.date_path = date_path
        self.should_show_the_plot = should_show_the_plot
        self.dataset_name = dataset_name
    
    
    def plot_primary_results(self, test_loss_list: list, best_dropout_rate_list: list, best_l2_reg_list:list, test_accuracy_list:list) -> None:
        
        iterations = range(len(test_loss_list))

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
        
        if self.should_show_the_plot:
            plt.show()
        
        
        root_dir = os.path.abspath(__file__).replace("core\\visualization.py", "") 
        
        # Check if the folder exists
        if not os.path.exists(f"{root_dir}\\logs\\{self.date_path}\\"):
            # Create the folder
            os.makedirs(f"{root_dir}\\logs\\{self.date_path}\\")
        # Specify the file name
        filename = f"{root_dir}\\logs\\{self.date_path}\\results_" + self.dataset_name + ".png"
        
        # Save the plot as an image
        plt.savefig(filename)
        
        
    def plot_results(self, file_path) -> None:
        pass
