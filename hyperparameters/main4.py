# %%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torcheval import metrics
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # For progress bars
import os
import json
from typing import Dict, Tuple, List
from collections import OrderedDict
import random
import argparse
import optuna
from optuna.trial import Trial
optuna.logging.set_verbosity(optuna.logging.ERROR)

# %%
class GeneralizedDataset(Dataset):
    def __init__(self, data_path: str) -> None:
        super().__init__()
        self._load_npz(data_path)
    def _load_npz(self, data_path: str) -> None:
        with np.load(data_path, allow_pickle=True) as file:
            self.x = file["x"].astype(np.float32)
            self.y = file["y"]
            if self.y.ndim > 1 and self.y.shape[1] > 1:
                self.y = np.argmax(self.y, axis=1).astype(np.int64)
            else:
                self.y = self.y.astype(np.int64)
    def __len__(self) -> int:
        return len(self.x)
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        return self.x[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, n_features: int,
                n_classes: int,
                no_layers: int,
                small_hidden_size: int,
                big_hidden_size: int) -> None:
        """
        Initializes the MLP model.

        Parameters:
        - n_features (int): Number of input features.
        - n_classes (int): Number of output classes.
        """
        super().__init__()
        if no_layers % 2 != 0:
            raise ValueError("Number of layers must be even.")
        # Define the layers
        layers = []
        for i in range(no_layers):
            if i == 0:
                layers.append(('dense_{}'.format(i), nn.Linear(n_features, big_hidden_size)))
            elif i % 2 == 1:
                layers.append(('dense_{}'.format(i), nn.Linear(big_hidden_size, small_hidden_size)))
            else:
                layers.append(('dense_{}'.format(i), nn.Linear(small_hidden_size, big_hidden_size)))
            layers.append(('activation_{}'.format(i), nn.ReLU()))
        # Output layer without Softmax
        layers.append(('dense_output', nn.Linear(small_hidden_size, n_classes)))
        self.layers = nn.Sequential(OrderedDict(layers))
        self.criterion = nn.CrossEntropyLoss()
        self.metric = metrics.MulticlassAccuracy(num_classes=n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def calculate_accuracy(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        self.metric.update(y_pred, y_true)
        accuracy = self.metric.compute().item()
        self.metric.reset()
        return accuracy

class ModelWrapper:
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        device: str,
        no_layers: int,
        small_hidden_size: int,
        big_hidden_size: int
        ) -> None:
        self.device = device
        self.model = MLP(
            n_features=n_features,
            n_classes=n_classes,
            no_layers=no_layers,
            small_hidden_size=small_hidden_size,
            big_hidden_size=big_hidden_size
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def eval(self, loader: DataLoader) -> tuple:
        '''Equivalent to the test loop. Used by both test and validation.'''
        self.model.eval()
        total_loss = 0
        total_acc = 0
        with torch.no_grad():
            for inputs, targets in loader:
                targets = targets.to(self.device)
                inputs = inputs.to(self.device)
                predictions = self.model(inputs)
                batch_loss = self.model.criterion(predictions, targets)
                total_loss += batch_loss.item()
                total_acc += self.model.calculate_accuracy(predictions, targets)
        return total_loss/len(loader), total_acc/len(loader)
    
    def train(self, train_loader: DataLoader) -> tuple:
        '''Equivalent to the training loop'''
        self.model.train()
        for inputs, targets in train_loader:
            targets = targets.to(self.device)
            inputs = inputs.to(self.device)
            predictions = self.model(inputs)
            loss = self.model.criterion(predictions, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, paitient_factor: int = 10) -> tuple:
        '''An umbrella method on top of the training and validation loops'''
        best_val_loss = float('inf')
        patience_counter = 0
        for epoch_num in range(epochs):
            self.train(train_loader)
            val_loss, val_acc = self.eval(val_loader)
            # Check for improvement
            if val_loss < best_val_loss:  # Lower is better 
                best_val_loss = val_loss
                patience_counter = 0  # Reset patience counter on improvement
            else:
                patience_counter += 1
            # Early stopping
            if patience_counter >= paitient_factor:
                break
        return best_val_loss, val_acc

class HyperparameterTuner:
    def __init__(self, X_train, Y_train, X_test, y_test, n_features: int,
        n_classes: int,
        dataset_name: str,
        device: str) -> None:
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.y_test = y_test
        self.device = device
        self.n_classes = n_classes
        self.n_features = n_features
        self.dataset_name = dataset_name

    def objective(self, trial: optuna.trial) -> float:
        # CREATE MODEL      
        no_layers = trial.suggest_categorical("no_layers", [2, 4, 6, 8, 10])
        small_hidden_size = trial.suggest_categorical("small_hidden_size", [8, 16, 32, 64, 128, 256])
        big_hidden_size = trial.suggest_categorical("big_hidden_size", [128, 256, 512, 1024, 2048, 4096])
        model = ModelWrapper(
            n_classes=self.n_classes,
            n_features=self.n_features,
            device=self.device,
            no_layers=no_layers,
            small_hidden_size=small_hidden_size,
            big_hidden_size=big_hidden_size
        )
        train_dataset = TensorDataset(
            torch.tensor(self.X_train, dtype=torch.float32),
            torch.tensor(self.Y_train, dtype=torch.long)
        )
        batch_size = 64
        epochs = 100
        if self.dataset_name == 'protein':
            batch_size = 128
            epochs = 50
        if len(self.X_train) <= batch_size:
            train_loader = DataLoader(train_dataset, batch_size=len(self.X_train), shuffle=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        validation_dataset = TensorDataset(
            torch.tensor(self.X_test, dtype=torch.float32),
            torch.tensor(self.y_test, dtype=torch.long)
        )
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
        loss, val_acc = model.fit(train_loader, validation_loader, epochs=epochs)
        return loss

    def tune(self) -> None:
        search_space = {
            "no_layers": [2, 4, 6, 8, 10],
            "small_hidden_size": [8, 16, 32, 64, 128, 256],
            "big_hidden_size": [128, 256, 512, 1024, 2048, 4096]
        }
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.GridSampler(search_space=search_space, seed=42))
        study.optimize(self.objective, n_trials=180)
        best_no_layers = study.best_params['no_layers']
        best_big_hidden_size = study.best_params['big_hidden_size']
        best_small_hidden_size = study.best_params['small_hidden_size']
        trials_df = study.trials_dataframe()
        return best_no_layers, best_small_hidden_size, best_big_hidden_size, trials_df

# %%
# List of dataset file paths
dataset_paths = [
    'data/splice/splice.npz',
    'data/protein/protein.npz',
    'data/dna/dna.npz',
    'data/twomoons/twomoons.npz',
    'data/electricalFault/detect.npz',
    'data/pokerdataset/poker.npz'
]

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run experiments with different datasets and regularization types.')
parser.add_argument('--d', type=int, required=False, help='An integer argument for demonstration purposes.')
args = parser.parse_args()
# Specify the device ('cpu' or 'cuda' if GPU is available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"Using device: {device}")
def main(dataset_path):
    # Extract dataset name from the file path
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    print(f"Running experiments on '{dataset_name}' dataset.")
    # Load the current dataset
    DS = GeneralizedDataset(data_path=dataset_path)
    X = np.asarray(DS.x, dtype=np.float32)
    y = np.asarray(DS.y).astype(np.int64)
    # Determine number of classes
    n_classes = len(np.unique(y))
    n_features = X.shape[1]
    # Split the dataset into training and testing sets
    train_idx, test_idx = train_test_split(
        np.arange(len(X)),
        train_size=0.6,
        random_state=42,
        stratify=y if n_classes > 2 else None  # Stratify for classification balance
    )
    X_train_full, y_train_full = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    # Define the CSV filename based on dataset and regularization type
    csv_filename = f'{dataset_name}_ARCH_HPO_results.csv'
    best_no_layers, best_small_hidden_size, best_big_hidden_size, trials_df = HyperparameterTuner(
        n_classes=n_classes,
        n_features=n_features,
        X_train=X_train_full,
        Y_train=y_train_full,
        X_test=X_test,
        y_test=y_test,
        dataset_name=dataset_name,
        device=device
    ).tune()
    # Save the results to the CSV file
    trials_df.to_csv(csv_filename, index=False)
    print(f'Best number of layers: {best_no_layers}')
    print(f'Best small hidden size: {best_small_hidden_size}')
    print(f'Best big hidden size: {best_big_hidden_size}')
    print(f"Results saved to '{csv_filename}'.")


if __name__ == '__main__':
    DS_2_RUN : str = dataset_paths[args.d]
    main(DS_2_RUN)