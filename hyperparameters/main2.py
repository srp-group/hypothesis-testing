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
    def __init__(self, n_features: int, n_classes: int, dropout_rate: float = 0.5, use_dropout: bool = False) -> None:
        """
        Initializes the MLP model.

        Parameters:
        - n_features (int): Number of input features.
        - n_classes (int): Number of output classes.
        - dropout_rate (float): Dropout rate.
        - use_dropout (bool): Whether to include dropout layers.
        """
        super().__init__()

        layer_sizes = [
            (n_features, 512),
            (512, 128),
            (128, 512),
            (512, 128),
            (128, 512),
            (512, 128)
        ]

        # layer_sizes = [
        #     (n_features, 2048),
        #     (2048, 512),
        #     (512, 2048),
        #     (2048, 512),
        #     (512, 2048),
        #     (2048, 512),
        #     (512, 2048),
        #     (2048, 512),
        #     (512, 2048),
        #     (2048, 512)
        # ]

        layers = []
        for i, (in_size, out_size) in enumerate(layer_sizes):
            layers.append(('dense_{}'.format(i), nn.Linear(in_size, out_size)))
            layers.append(('activation_{}'.format(i), nn.ReLU()))
            if use_dropout:
                layers.append(('dropout_{}'.format(i), nn.Dropout(dropout_rate)))

        # Output layer without Softmax
        layers.append(('dense_output', nn.Linear(layer_sizes[-1][1], n_classes)))

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
        reg_type: str,
        reg_val: np.float32,
        n_features: int,
        n_classes: int,
        dropout_rate: float = 0.5,
        device: str = 'cpu',
        lr: float = 0.001
    ):
        self.device = device
        reg_type = reg_type.lower()
        if reg_type == 'dropout':
            use_dropout = True
            weight_decay = 0.0  # Disable L2 regularization when using dropout
            dropout_rate = reg_val  # Use reg_val as the dropout rate
        elif reg_type == 'l2':
            use_dropout = False
            weight_decay = reg_val

        self.model = MLP(
            n_features=n_features,
            n_classes=n_classes,
            dropout_rate=dropout_rate,
            use_dropout=use_dropout
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), weight_decay=weight_decay, lr=lr)
        self.reg_type = reg_type
        self.reg_val = reg_val

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
        total_loss = 0
        total_acc = 0
        for inputs, targets in train_loader:
            targets = targets.to(self.device)
            inputs = inputs.to(self.device)
            predictions = self.model(inputs)
            loss = self.model.criterion(predictions, targets)
            total_loss += loss.item()
            self.optimizer.zero_grad()
            total_acc += self.model.calculate_accuracy(predictions, targets)
            loss.backward()
            self.optimizer.step()
        return total_loss/len(train_loader), total_acc/len(train_loader)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, paitient_factor: int = 10) -> tuple:
        '''An umbrella method on top of the training and validation loops'''
        best_val_loss = float('inf')
        patience_counter = 0
        for epoch_num in range(epochs):
            train_loss, train_acc = self.train(train_loader)
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
        return train_loss, train_acc, best_val_loss, val_acc

def get_lambda_list(reg_type: str, dataset_name: str) -> np.ndarray:
    # Define lambda_list based on regularization type
    if dataset_name == 'splice':
        if reg_type.lower() == 'l2':
            lambda_list: np.ndarray = np.unique(np.concatenate([np.logspace(-6, -2, num=5) ,np.linspace(0.01, 0.1, num=10)])).astype(np.float32)
        elif reg_type.lower() == 'dropout':
            # For dropout, reg_val represents the dropout rate; typically between 0.1 and 0.9
            lambda_list: np.ndarray = np.array([0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    elif dataset_name == 'protein':
        if reg_type.lower() == 'l2':
            lambda_list: np.ndarray = np.unique(np.concatenate([np.logspace(-6, -3, num=4) ,np.linspace(0.001, 0.01, num=10), np.logspace(-2, -1, num=2)])).astype(np.float32)
        elif reg_type.lower() == 'dropout':
            # For dropout, reg_val represents the dropout rate; typically between 0.1 and 0.9
            lambda_list: np.ndarray = np.array([0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    elif dataset_name == 'dna':
        if reg_type.lower() == 'l2':
            lambda_list: np.ndarray = np.unique(np.concatenate([np.logspace(-6, -2, num=5) ,np.linspace(0.01, 0.1, num=10)])).astype(np.float32)
        elif reg_type.lower() == 'dropout':
            # For dropout, reg_val represents the dropout rate; typically between 0.1 and 0.9
            lambda_list: np.ndarray = np.array([0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    else:
        if reg_type.lower() == 'l2':
            lambda_list: np.ndarray = np.logspace(-6, 0, 7).astype(np.float32)
        elif reg_type.lower() == 'dropout':
            # For dropout, reg_val represents the dropout rate; typically between 0.1 and 0.9
            lambda_list: np.ndarray = np.array([0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    return lambda_list

def run_experiment(
    data: Tuple[np.ndarray, np.ndarray],
    dataset_name: str,
    reg_type: str,
    device: str,
    batch_size: int,
    epochs: int
) -> pd.DataFrame:
    # Define experiment parameters
    seeds: List[int] = list(range(31, 61))  # Seeds 1 to 30
    dataset_sizes_pct: np.ndarray = np.unique(np.concatenate([np.linspace(1, 10, num=10), np.linspace(10, 100, num=10)])).astype(np.int32) # Dataset sizes in percentages
    # Define the list of regularization parameters
    if reg_type.lower() == 'l2':
        points = np.array([0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
        midpoints = (points[:-1] + points[1:]) / 2
        result = np.sort(np.concatenate((points, midpoints)))
        lambda_list: np.ndarray = result.astype(np.float32)
    else:
        lambda_list: np.ndarray = np.array([0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)

    # Initialize a list to store all results
    results: List[Dict[str, any]] = []

    # Iterate over each dataset
    print(f"Processing dataset: {dataset_name}")

    # Ensure X and y are NumPy arrays
    X = np.asarray(data[0], dtype=np.float32)
    y = np.asarray(data[1]).astype(np.int64)

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


    # Iterate over each seed for reproducibility
    for seed in tqdm(seeds, desc=f"Seeds for {dataset_name}", leave=False):
        # setting the seed for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Iterate over each dataset size percentage
        for d in dataset_sizes_pct:
            # Calculate the number of samples for the current dataset size
            n_samples = max(1, int(len(X_train_full) * (d / 100.0)))

            # Randomly sample without replacement
            sampled_indices = np.random.choice(
                len(X_train_full),
                size=n_samples,
                replace=False
            )
            X_train_d, y_train_d = X_train_full[sampled_indices], y_train_full[sampled_indices]
            # Iterate over each regularization parameter
            for reg_val in lambda_list:
                for lr in np.logspace(-5, -1, 5):
                    # Initialize the model with the current regularization parameters
                    model = ModelWrapper(
                        reg_type=reg_type,
                        reg_val=reg_val,
                        n_features=n_features,
                        n_classes=n_classes,
                        dropout_rate=0.5,  # Default dropout rate; overridden if reg_type is 'dropout'
                        device=device,
                        lr=lr
                    )
                    train_dataset = TensorDataset(
                        torch.tensor(X_train_d, dtype=torch.float32),
                        torch.tensor(y_train_d, dtype=torch.long)
                    )
                    
                    if len(X_train_d) <= batch_size:
                        train_loader = DataLoader(train_dataset, batch_size=len(X_train_d), shuffle=True)
                    else:
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
                    
                    validation_dataset = TensorDataset(
                        torch.tensor(X_test, dtype=torch.float32),
                        torch.tensor(y_test, dtype=torch.long)
                    )
                    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
                    
                    train_loss, train_acc, loss, val_acc = model.fit(train_loader, validation_loader, epochs=epochs)
                    
                    # Store the results
                    results.append({
                        'dataset_name': dataset_name,
                        'reg_type': reg_type,
                        'reg_val': reg_val,
                        'loss': loss,
                        'seed': seed,
                        'data_size_pct': d,
                        'val_acc': val_acc,
                        "lr": lr,
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                    })

    print(f"Completed processing for dataset: {dataset_name}\n")

    # Convert the results list to a pandas DataFrame
    results_df = pd.DataFrame(results)

    return results_df

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
reg_types = [
    'l2',
    'dropout'
]
# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run experiments with different datasets and regularization types.')
parser.add_argument('--d', type=int, required=False, help='An integer argument for demonstration purposes.')
parser.add_argument('--r', type=int, required=False, help='An integer argument for demonstration purposes.')

args = parser.parse_args()



# Specify the device ('cpu' or 'cuda' if GPU is available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"Using device: {device}")

def main(dataset_path, reg_type):
        # Extract dataset name from the file path
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    
    # Load the current dataset
    DS = GeneralizedDataset(data_path=dataset_path)
    dataset_x, dataset_y = DS.x, DS.y
    # Iterate over each regularization type
    reg_type = reg_type
    print(f"Applying regularization type: {reg_type}")
    
    # Run the experiment for the current dataset and regularization type
    batch_size = 64
    epochs = 100
    if dataset_name == 'protein':
        batch_size = 128
        epochs = 50
    results_df = run_experiment(data=(dataset_x, dataset_y), dataset_name=dataset_name, reg_type=reg_type, device=device, batch_size=batch_size, epochs=epochs)
    
    # Define the CSV filename based on dataset and regularization type
    csv_filename = f'{dataset_name}_{reg_type}_results.csv'
    
    # Save the results to the CSV file
    results_df.to_csv(csv_filename, index=False)
    print(f"Results saved to '{csv_filename}'.")


if __name__ == '__main__':
    if args.d is None:
        for reg_type in reg_types:
            for dataset_path in dataset_paths[:-2]:
                main(dataset_path, reg_type)
    else:
        DS_2_RUN : str = dataset_paths[args.d]
        if args.r is None:
            for reg_type in reg_types:
                main(DS_2_RUN, reg_type)
        else:
            REG_2_RUN : str = reg_types[args.r]
            main(DS_2_RUN, REG_2_RUN)