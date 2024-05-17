import matplotlib.pyplot as plt
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

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
        
        
        # Save the plot to an HTML file
        current_file_path = os.path.abspath(__file__)
        root_dir = os.path.join(os.path.dirname(current_file_path), '..')
        root_dir = os.path.normpath(root_dir)
        
        # Check if the folder exists
        if not os.path.exists(f"{root_dir}\\logs\\{self.date_path}\\"):
            # Create the folder
            os.makedirs(os.path.normpath(f"{root_dir}\\logs\\{self.date_path}\\"))
        # Specify the file name
        filename = f"{root_dir}\\logs\\{self.date_path}\\results_" + self.dataset_name + ".png"
        filename = os.path.normpath(filename)
        # Save the plot as an image
        plt.savefig(filename)
        
    
    
    def plot_3d_scatter_surface(self, df: pd.DataFrame, x_variable: str, y_variables: list, output_variable: str, surface: bool = True, highlight: bool = False) -> None:
        name = ['L2', 'Dropout']
        
        for i, y_var in enumerate(y_variables):
            print(f'{name[i]}: Min value = {df[y_var].min()}, Max value = {df[y_var].max()}, Range = {df[y_var].max() - df[y_var].min()}')
            print(f'Min Loss = {df[output_variable].min()}')
            # Find the index of the lowest output variable value if highlight is True
            lowest_index = df[output_variable].idxmin() if highlight else None
            
            # Create a scatter plot
            scatter = go.Scatter3d(
                x=df[x_variable],
                y=df[y_var],
                z=df[output_variable],
                mode='markers',
                marker=dict(
                    size=[15 if i == lowest_index else 5 for i in range(len(df))],  # Highlighted point is bigger,
                    color=['red' if i == lowest_index else 'grey' for i in range(len(df))] if highlight else 'grey',  # Highlight the lowest value if highlight is True
                    colorscale='Viridis',  # Choose a color scale
                    opacity=0.8,
                ),
                name=output_variable
            )

            # Create surface data
            x_data = df[x_variable]
            y_data = df[y_var]
            z_data = df[output_variable]

            x_range = np.linspace(min(x_data), max(x_data), 100)
            y_range = np.linspace(min(y_data), max(y_data), 100)
            X, Y = np.meshgrid(x_range, y_range)
            Z = griddata((x_data, y_data), z_data, (X, Y), method='cubic')

            if surface:
                # Create a 3D surface
                surface = go.Surface(
                    x=X,
                    y=Y,
                    z=Z,
                    colorscale='Jet',  # Choose a color scale
                    opacity=0.6
                )

                # Create the figure
                fig = go.Figure(data=[scatter, surface])

            else:
                fig = go.Figure(data=[scatter])

            # Update layout
            fig.update_layout(
                scene=dict(
                    xaxis_title=x_variable,
                    yaxis_title=y_var,
                    zaxis_title=output_variable
                ),
                width=800,  # adjust width
                height=600,  # adjust height
                title=f'Mean Loss for different values of {name[i]} regularization'
            )

            # Show the plot
            if self.should_show_the_plot:
                fig.show()

            # Save the plot to an HTML file
            current_file_path = os.path.abspath(__file__)
            root_dir = os.path.join(os.path.dirname(current_file_path), '..')
            root_dir = os.path.normpath(root_dir)
        
            # Check if the folder exists    
            if not os.path.exists(f"{root_dir}\\logs\\{self.date_path}\\"):
                # Create the folder
                os.makedirs(os.path.normpath(f"{root_dir}\\logs\\{self.date_path}\\"))
            file_name = f'{root_dir}\\logs\\{self.date_path}\\3D_plot_{output_variable}_vs_{y_var}_plot.html'
            filename = os.path.normpath(filename)
            pio.write_html(fig, file_name)
    
    def plot_2d_scatter(self, df: pd.DataFrame, x_column: str, output_column: str, highlight: bool = False):

        # Find the index of the lowest output variable value if highlight is True
        lowest_index = df[output_column].idxmin() if highlight else None
        
        # Create a scatter plot
        scatter = go.Scatter(
            x=df[x_column],
            y=df[output_column],
            mode='lines+markers',
        
            marker=dict(
                size=[15 if i == lowest_index else 5 for i in range(len(df))],  # Highlighted point is bigger
                color=['red' if i == lowest_index else 'blue' for i in range(len(df))] if highlight else 'blue',  # Highlight the lowest value if highlight is True
                opacity=0.8,
            ),
            name=output_column
        )

        # Create the figure
        fig = go.Figure(data=[scatter])

        # Update layout
        fig.update_layout(
            xaxis_title=x_column,
            yaxis_title=output_column,
            # yaxis=dict(tickformat='e'),
            width=800,  # adjust width
            height=600,  # adjust height
            title=f'{output_column} as a function of {x_column}'
        )

        # Show the plot
        if self.should_show_the_plot:
            fig.show()
        # Save the plot to an HTML file
        current_file_path = os.path.abspath(__file__)
        root_dir = os.path.join(os.path.dirname(current_file_path), '..')
        root_dir = os.path.normpath(root_dir)
        # Check if the folder exists    
        if not os.path.exists(f"{root_dir}\\logs\\{self.date_path}\\"):
            # Create the folder
            os.makedirs(os.path.normpath(f"{root_dir}\\logs\\{self.date_path}\\"))
        # Save the plot to an HTML file
        file_name = f'{root_dir}\\logs\\{self.date_path}\\{output_column}_vs_{x_column}_plot.html'
        filename = os.path.normpath(filename)
        pio.write_html(fig, file_name)
    
    
    def plot_results(self, file_path) -> None:
        df = pd.read_csv(file_path)
        # df = pd.read_csv('AUBC_test.csv')

        df.columns.values[0] = 'iterations'
        
        # Calculate the interval between each instance
        df['interval'] = df['iterations'].diff().fillna(0)

        # Calculate the trapezoidal area
        df.loc[:, 'trapezoidal_area'] = 0.5 * (df['loss'] + df['loss'].shift(1, fill_value=0)) * df['interval']

        df['cumulative_area'] = df['trapezoidal_area'].cumsum()

        df['AUBC'] = df['cumulative_area'] / (df['cumulative_area'].max() * df.shape[0])

        df['AUBC_totals'] = df['AUBC'].cumsum()
        
        self.plot_3d_scatter_surface(df, 'iterations', ['L2_value', 'drop_value'], 'loss', surface=False, highlight=True)
        self.plot_2d_scatter(df, 'iterations', 'loss', highlight=True)
        self.plot_2d_scatter(df, 'iterations', 'L2_value', highlight=True)
        self.plot_2d_scatter(df, 'iterations', 'drop_value', highlight=True)
        print(f"Area under the curve (AUC): {df['AUBC_totals'].iloc[-1]:.3f}")
        self.plot_2d_scatter(df, 'iterations', 'AUBC', highlight=False)

