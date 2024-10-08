import matplotlib.pyplot as plt
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import base64
import io
from typing import List

class Visualization:
    def __init__(self, should_show_the_plot: bool, dataset_name: str, logging_dir : str, model_name : str) -> None:
        self.should_show_the_plot = should_show_the_plot
        self.dataset_name = dataset_name
        self.logging_dir = logging_dir
        self.model_name = model_name              

    def plot_primary_results(self, test_loss_list: List[float], best_dropout_rate_list: List[float], best_l2_reg_list: List[float], test_accuracy_list: List[float]) -> str:
        """
        Creates a Matplotlib figure with four subplots arranged in a 2x2 grid and returns it as a Base64-encoded PNG image.
        """
        iterations = range(len(test_loss_list))

        # Adjusted figure size for 2x2 subplots
        plt.figure(figsize=(15, 10))  # Width=15 inches, Height=10 inches

        # Plot for test loss
        plt.subplot(2, 2, 1)  # 2 rows, 2 columns, 1st subplot
        plt.plot(iterations, test_loss_list, label='Test Loss', color='blue')
        plt.xlabel('Iteration')
        plt.ylabel('Test Loss')
        plt.title('Test Loss per Iteration')
        plt.legend()

        # Plot for best dropout rate
        plt.subplot(2, 2, 2)  # 2 rows, 2 columns, 2nd subplot
        plt.plot(iterations, best_dropout_rate_list, label='Best Dropout Rate', color='green')
        plt.xlabel('Iteration')
        plt.ylabel('Dropout Rate')
        plt.title('Dropout Rate per Iteration')
        plt.legend()

        # Plot for best L2 regularization
        plt.subplot(2, 2, 3)  # 2 rows, 2 columns, 3rd subplot
        plt.plot(iterations, best_l2_reg_list, label='Best L2 Reg', color='purple')
        plt.xlabel('Iteration')
        plt.ylabel('L2 Regularization')
        plt.title('L2 Regularization per Iteration')
        plt.legend()

        # Plot for test accuracy
        plt.subplot(2, 2, 4)  # 2 rows, 2 columns, 4th subplot
        plt.plot(iterations, test_accuracy_list, label='Test Accuracy', color='orange')
        plt.xlabel('Iteration')
        plt.ylabel('Test Accuracy')
        plt.title('Test Accuracy per Iteration')
        plt.legend()

        plt.tight_layout()

        if self.should_show_the_plot:
            plt.show()

        # Save the plot to a BytesIO buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')

        return image_base64
    
    
    def plot_3d_scatter_surface(self, df: pd.DataFrame, x_variable: str, y_variables: List[str], output_variable: str, surface: bool = True, highlight: bool = False) -> List[str]:
        """
        Creates 3D scatter (and optionally surface) plots and returns their HTML representations.
        """
        name = ['L2', 'Dropout']
        html_snippets = []
        
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
                    size=[15 if idx == lowest_index else 5 for idx in df.index],  # Highlighted point is bigger
                    color=['red' if idx == lowest_index else 'grey' for idx in df.index] if highlight else 'grey',  # Highlight the lowest value if highlight is True
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
                surface_plot = go.Surface(
                    x=X,
                    y=Y,
                    z=Z,
                    colorscale='Jet',  # Choose a color scale
                    opacity=0.6,
                    name='Surface'
                )
                # Create the figure
                fig = go.Figure(data=[surface_plot, scatter])
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

            if self.should_show_the_plot:
                fig.show()

            # Get HTML representation of the figure
            fig_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
            html_snippets.append(fig_html)
        
        return html_snippets
    
    
    def plot_2d_scatter(self, df: pd.DataFrame, x_column: str, output_column: str, highlight: bool = False) -> str:
        """
        Creates a 2D scatter plot and returns its HTML representation.
        """
        # Find the index of the lowest output variable value if highlight is True
        lowest_index = df[output_column].idxmin() if highlight else None
        
        # Create a scatter plot
        scatter = go.Scatter(
            x=df[x_column],
            y=df[output_column],
            mode='lines+markers',
        
            marker=dict(
                size=[15 if idx == lowest_index else 5 for idx in df.index],  # Highlighted point is bigger
                color=['red' if idx == lowest_index else 'blue' for idx in df.index] if highlight else 'blue',  # Highlight the lowest value if highlight is True
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
            width=800,  # adjust width
            height=600,  # adjust height
            title=f'{output_column} as a function of {x_column}'
        )

        if self.should_show_the_plot:
            fig.show()

        # Get HTML representation of the figure
        fig_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        return fig_html
    
    
    def plot_results(self, file_path: str) -> None:
        df = pd.read_csv(file_path)
        # df = pd.read_csv('AUBC_test.csv')

        df.columns.values[0] = 'iterations'
        
        # Calculate the interval between each instance
        df['interval'] = df['iterations'].diff().fillna(0)

        # Calculate the trapezoidal area
        df['trapezoidal_area'] = 0.5 * (df['loss'] + df['loss'].shift(1, fill_value=0)) * df['interval']

        df['cumulative_area'] = df['trapezoidal_area'].cumsum()

        df['AUBC'] = df['cumulative_area'] / (df['cumulative_area'].max() * df.shape[0])

        df['AUBC_totals'] = df['AUBC'].cumsum()
        
        # Collect all plot HTML snippets
        all_plots_html = []

        # # Plot 3D scatter and surface
        # if self.model_name == 'MLP':
        #     plots_3d = self.plot_3d_scatter_surface(df, 'iterations', ['L2_value', 'drop_value'], 'loss', surface=True, highlight=True)
        #     all_plots_html.extend(plots_3d)
        # elif self.model_name in ['MLR', 'SVM']:
        #     plots_3d = self.plot_3d_scatter_surface(df, 'iterations', ['L2_value'], 'loss', surface=False, highlight=True)
        #     all_plots_html.extend(plots_3d)

        # Plot 2D scatter plots
        scatter_loss = self.plot_2d_scatter(df, 'iterations', 'loss', highlight=True)
        scatter_l2 = self.plot_2d_scatter(df, 'iterations', 'L2_value', highlight=True)
        scatter_drop = self.plot_2d_scatter(df, 'iterations', 'drop_value', highlight=True)
        scatter_aubc = self.plot_2d_scatter(df, 'iterations', 'AUBC', highlight=False)

        all_plots_html.extend([scatter_loss, scatter_l2, scatter_drop, scatter_aubc])

        # Plot primary results (Matplotlib)
        primary_plot_base64 = self.plot_primary_results(
            test_loss_list=df['loss'].tolist(),
            best_dropout_rate_list=df['drop_value'].tolist(),
            best_l2_reg_list=df['L2_value'].tolist(),
            test_accuracy_list=df.get('accuracy', [0]*len(df)).tolist()  # Assuming there's an 'accuracy' column
        )

        # Create HTML content
        '''
        <h2>3D Scatter and Surface Plots</h2>
        {"".join(plots_3d) if self.model_name in ['MLP', 'MLR', 'SVM'] else ''}
        '''
        html_content = f"""
        <html>
        <head>
            <title>All Plots for {self.dataset_name}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>Visualization Results for {self.dataset_name}</h1>
            
            <h2>Primary Results</h2>
            <img src="data:image/png;base64,{primary_plot_base64}" alt="Primary Results Plot" style="width:100%; height:auto;"/>

            <h2>2D Scatter Plots</h2>
            {scatter_loss}
            {scatter_l2}
            {scatter_drop}
            {scatter_aubc}

            <h2>Area under the curve (AUC)</h2>
            <p>{df['AUBC_totals'].iloc[-1]:.3f}</p>
        </body>
        </html>
        """

        # Specify the output HTML file name
        output_html = os.path.join(self.logging_dir, f'all_results_{self.dataset_name}.html')
        output_html = os.path.normpath(output_html)

        # Write the HTML content to the file
        with open(output_html, 'w') as f:
            f.write(html_content)

        print(f"All plots have been saved to {output_html}")
