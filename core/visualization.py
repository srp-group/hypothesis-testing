import matplotlib.pyplot as plt
import datetime


class Visualization:
    def __init__(self) -> None:
        pass
    
    def plot_primary_results(self, test_loss_list: list, best_dropout_rate_list: list, best_l2_reg_list:list, test_accuracy_list:list, dataset_name: str, should_show_the_plot: bool) -> None:
        
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
        
        if should_show_the_plot:
            plt.show()
        
        current_time = datetime.datetime.now()

        # Specify the file name
        filename = "logs/results_" + dataset_name + "_" + current_time.strftime("%Y-%m-%d_%H-%M-%S") + ".png"
        
        # Save the plot as an image
        plt.savefig(filename)
