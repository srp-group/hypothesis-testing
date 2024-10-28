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

# %%
class GeneralizedDataset(Dataset):
    def __init__(self, data_path: str, data_format: str = 'npz') -> None:
        """
        Initializes the dataset by loading data from the specified path and format.

        Parameters:
        - data_path (str): Path to the dataset file.
        - data_format (str): Format of the dataset ('npz', 'csv', 'json').
        """
        super().__init__()
        self.data_format = data_format.lower()

        # Load data based on format
        load_method = {
            'npz': self._load_npz,
            'csv': self._load_csv,
            'json': self._load_json
        }.get(self.data_format)

        if load_method:
            load_method(data_path)
        else:
            raise ValueError(f"Unsupported data format: {self.data_format}")

    def _load_npz(self, data_path: str) -> None:
        with np.load(data_path, allow_pickle=True) as file:
            self.x = file["x"].astype(np.float32)
            self.y = file["y"]
            if self.y.ndim > 1 and self.y.shape[1] > 1:
                self.y = np.argmax(self.y, axis=1).astype(np.int64)
            else:
                self.y = self.y.astype(np.int64)

    def _load_csv(self, data_path: str) -> None:
        df = pd.read_csv(data_path)
        # Assuming the last column is the target
        self.x = df.iloc[:, :-1].values.astype(np.float32)
        self.y = df.iloc[:, -1].values
        if self.y.dtype == object or self.y.ndim > 1 or len(np.unique(self.y)) > 1:
            self.y = pd.get_dummies(self.y).values
            self.y = np.argmax(self.y, axis=1).astype(np.int64)
        else:
            self.y = self.y.astype(np.int64)

    def _load_json(self, data_path: str) -> None:
        with open(data_path, 'r') as f:
            data = json.load(f)
        self.x = np.array(data['x'], dtype=np.float32)
        self.y = np.array(data['y'])
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
            (n_features, 1024),
            (1024, 256),
            (256, 1024),
            (1024, 256),
            (256, 1024),
            (1024, 256)
        ]

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

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

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
        reg_val: float,
        n_features: int,
        n_classes: int,
        dropout_rate: float = 0.5,
        device: str = 'cpu'
    ):
        """
        Initializes the ModelWrapper with the specified regularization.

        Parameters:
        - reg_type (str): Type of regularization ('L2', 'dropout', 'none').
        - reg_val (float): Regularization strength (λ). For dropout, this represents the dropout rate.
        - n_features (int): Number of input features.
        - n_classes (int): Number of output classes.
        - dropout_rate (float): Dropout rate (used only if reg_type is 'dropout').
        - device (str): 'cpu' or 'cuda' for GPU.
        """
        self.device = device
        reg_type = reg_type.lower()

        if reg_type == 'dropout':
            use_dropout = True
            weight_decay = 0.0  # Disable L2 regularization when using dropout
            dropout_rate = reg_val  # Use reg_val as the dropout rate
        elif reg_type == 'l2':
            use_dropout = False
            weight_decay = reg_val
        elif reg_type == 'none':
            use_dropout = False
            weight_decay = 0.0
        else:
            raise ValueError(f"Unsupported regularization type: {reg_type}")

        self.model = MLP(
            n_features=n_features,
            n_classes=n_classes,
            dropout_rate=dropout_rate,
            use_dropout=use_dropout
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), weight_decay=weight_decay)
        self.reg_type = reg_type
        self.reg_val = reg_val

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32
    ) -> None:
        """
        Trains the model on the provided training data.

        Parameters:
        - X_train (np.ndarray): Training features.
        - y_train (np.ndarray): Training targets.
        - epochs (int): Number of training epochs.
        - batch_size (int): Batch size for training.
        """
        self.model.train()

        # Create DataLoader
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(1, epochs + 1):
            epoch_losses = []
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                # Forward pass
                outputs = self.model(batch_X)
                loss = self.model.criterion(outputs, batch_y)
                epoch_losses.append(loss.item())

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # avg_loss = np.mean(epoch_losses)
            # Optionally, log the loss
            # print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Evaluates the model on the test data and returns the loss.

        Parameters:
        - X_test (np.ndarray): Test features.
        - y_test (np.ndarray): Test targets.

        Returns:
        - loss (float): The loss value on the test set.
        """
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(self.device)

            outputs = self.model(X_test_tensor)
            loss = self.model.criterion(outputs, y_test_tensor).item()

        return loss

def run_experiment(
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    reg_type: str,
    device: str = 'cpu'
) -> pd.DataFrame:
    """
    Runs the experiment as per the specified algorithm.

    Parameters:
    - datasets (dict): Mapping from dataset names to (X, y) tuples.
                        Example: {'dataset1': (X1, y1), 'dataset2': (X2, y2)}
    - reg_type (str): The regularization type to test (e.g., 'L1', 'L2', 'dropout', 'none').
    - device (str): 'cpu' or 'cuda' for GPU usage.

    Returns:
    - results_df (pd.DataFrame): DataFrame containing the results with columns
        ['dataset_name', 'reg_type', 'reg_val', 'loss', 'seed', 'data_size_pct']
    """
    # Define experiment parameters
    seeds: List[int] = list(range(1, 31))  # Seeds 1 to 10
    dataset_sizes_pct: List[int] = [2, 3, 4, 6, 8, 10]  # Dataset sizes in percentages

    # Define lambda_list based on regularization type
    if reg_type.lower() == 'l2':
        lambda_list: List[float] = [10**i for i in range(0, -6, -1)]  # 10^0 to 10^-5
    elif reg_type.lower() == 'dropout':
        # For dropout, reg_val represents the dropout rate; typically between 0.1 and 0.9
        lambda_list: List[float] = [0.0, 0.1, 0.3, 0.5, 0.8]
    elif reg_type.lower() == 'none':
        lambda_list: List[float] = [0.0]
    else:
        raise ValueError(f"Unsupported regularization type for experiment: {reg_type}")

    # Initialize a list to store all results
    results: List[Dict[str, any]] = []

    # Iterate over each dataset
    for dataset_name, (X, y) in datasets.items():
        print(f"Processing dataset: {dataset_name}")

        # Ensure X and y are NumPy arrays
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.int64)

        # Determine number of classes
        n_classes = len(np.unique(y))
        n_features = X.shape[1]

        # Split the dataset into training and testing sets
        train_idx, test_idx = train_test_split(
            np.arange(len(X)),
            train_size=0.6,
            random_state=42,
            shuffle=True,
            stratify=y if n_classes > 2 else None  # Stratify for classification balance
        )
        X_train_full, y_train_full = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Iterate over each seed for reproducibility

        for seed in tqdm(seeds, desc=f"Seeds for {dataset_name}", leave=False):
            # print(seed)
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
                    # Initialize the model with the current regularization parameters
                    model = ModelWrapper(
                        reg_type=reg_type,
                        reg_val=reg_val,
                        n_features=n_features,
                        n_classes=n_classes,
                        dropout_rate=0.5,  # Default dropout rate; overridden if reg_type is 'dropout'
                        device=device
                    )

                    # Train the model on the sampled training data
                    model.train(X_train_d, y_train_d, epochs=50, batch_size=32)  # Adjust epochs and batch_size as needed

                    # Evaluate the model on the test set to obtain the loss
                    loss = model.evaluate(X_test, y_test)

                    # Store the results
                    results.append({
                        'dataset_name': dataset_name,
                        'reg_type': reg_type,
                        'reg_val': reg_val,
                        'loss': loss,
                        'seed': seed,
                        'data_size_pct': d
                    })

        print(f"Completed processing for dataset: {dataset_name}\n")

    # Convert the results list to a pandas DataFrame
    results_df = pd.DataFrame(results)

    return results_df

def load_datasets(dataset_paths: List[str]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Loads datasets from the provided file paths.

    Parameters:
    - dataset_paths (List[str]): List of dataset file paths.

    Returns:
    - datasets (Dict[str, Tuple[np.ndarray, np.ndarray]]): Mapping from dataset names to (X, y) tuples.
    """
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for file_path in dataset_paths:
        # Determine the format based on file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        if ext == '.csv':
            data_format = 'csv'
        elif ext == '.npz':
            data_format = 'npz'
        elif ext == '.json':
            data_format = 'json'
        else:
            raise ValueError(f"Unsupported file extension: {ext} for file {file_path}")

        # Load the dataset using GeneralizedDataset
        dataset = GeneralizedDataset(data_path=file_path, data_format=data_format)

        # Extract X and y
        X, y = dataset.x, dataset.y

        # Assign a dataset name based on the file name without extension
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]

        # Add to the datasets dictionary
        datasets[dataset_name] = (X, y)

    return datasets

# %%
# List of dataset file paths
dataset_paths = [
    'data/splice/splice.npz',
    'data/protein/protein.npz',
    'data/dna/dna.npz'
    # Add more dataset paths as needed
]

# List of regularization types
reg_types = [
    'l2',
    'dropout'
]

# Specify the device ('cpu' or 'cuda' if GPU is available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"Using device: {device}")

# Iterate over each dataset
for dataset_path in dataset_paths:
    # Extract dataset name from the file path
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    # print(f"\nProcessing dataset: {dataset_name}")
    
    # Load the current dataset
    dataset = load_datasets([dataset_path])  # Assuming load_datasets can handle a single path
    # If load_datasets expects multiple datasets and returns a list, you might need to adjust:
    # dataset = load_datasets([dataset_path])[0]
    
    # Iterate over each regularization type
    for reg_type in reg_types:
        print(f"Applying regularization type: {reg_type}")
        
        # Run the experiment for the current dataset and regularization type
        results_df = run_experiment(dataset, reg_type, device=device)
        
        # Define the CSV filename based on dataset and regularization type
        csv_filename = f'{dataset_name}_{reg_type}_results.csv'
        
        # Save the results to the CSV file
        results_df.to_csv(csv_filename, index=False)
        print(f"Results saved to '{csv_filename}'.")



