import matplotlib.pyplot as plt

class Visualization:
    def __init__(self) -> None:
        pass
    
    def plot_primary_results(self, test_loss_list: list, best_dropout_rate_list: list, best_l2_reg_list:list) -> None:
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