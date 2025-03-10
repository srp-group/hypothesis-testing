import argparse
import datetime
import os
import random
import time  # For time tracking
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torcheval import metrics
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

def format_time(seconds):
    """Formats time in seconds to HH:MM:SS."""
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


class GeneralizedDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self._load_npz(data_path)

    def _load_npz(self, data_path):
        with np.load(data_path, allow_pickle=True) as file:
            self.x = file["x"].astype(np.float32)
            self.y = file["y"]
            if self.y.ndim > 1 and self.y.shape[1] > 1:
                self.y = np.argmax(self.y, axis=1).astype(np.int64)
            else:
                self.y = self.y.astype(np.int64)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, n_features, n_classes, dropout_rate):
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
            layers.append(('dropout_{}'.format(i), nn.Dropout(dropout_rate)))


        # Output layer without Softmax
        layers.append(('dense_output', nn.Linear(layer_sizes[-1][1], n_classes)))

        self.layers = nn.Sequential(OrderedDict(layers))
        self.criterion = nn.CrossEntropyLoss()
        self.metric = metrics.MulticlassAccuracy(num_classes=n_classes)

    def forward(self, x):
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
        dropout_rate: float,
        device: str = 'cpu',
        lr: float = 0.001
    ):
        self.device = device
        
        self.model = MLP(
            n_features=n_features,
            n_classes=n_classes,
            dropout_rate=dropout_rate
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

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

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, paitient_factor: int) -> tuple:
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

def run_experiment(data, dataset_name, device, params, start_time):
    # Extract parameters
    batch_size = params["batch_size"]
    epochs = params["epochs"]
    patience = params["patience"]
    lr_rates = params["lr_rates"]
    lambda_list_dropout = params["lambda_list_dropout"]
    seeds = list(range(1, 31))  # Seeds 1 to 30
    results = []

    X = np.asarray(data[0], dtype=np.float32)
    y = np.asarray(data[1]).astype(np.int64)

    n_classes = len(np.unique(y))
    n_features = X.shape[1]

    # Split the dataset into training and testing sets
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, train_size=0.6, random_state=42, stratify=y)



    print(f"Train set size: {len(X_train_full)}")
    print(f"Test set size: {len(X_test)}")

    n = 100
    max_samples = 1200
    num_iterations = 12  # For linear increments
    total_iterations = (len(seeds) * (num_iterations) * len(lambda_list_dropout) * len(lr_rates))

    pbar = tqdm(total=total_iterations, desc="Total Progress", leave=True)

    # Iterate over each seed for reproducibility
    for seed in seeds:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        instances_used = []
        current_samples = n

        while current_samples <= max_samples:
            instances_used.append(current_samples)
            if current_samples != max_samples:
                # Stratified sampling for 'current_samples'
                sss2 = StratifiedShuffleSplit(
                    n_splits=1,
                    train_size=current_samples
                )
                indices_train, _ = next(sss2.split(X_train_full, y_train_full))
                X_train_d = X_train_full[indices_train]
                y_train_d = y_train_full[indices_train]
            else:
                X_train_d = X_train_full
                y_train_d = y_train_full
            actual_samples_used = len(X_train_d)       
            for reg_val_dropout in lambda_list_dropout:
                for lr in lr_rates:
                    # tqdm.write("")
                    pbar.set_description(
                        f"Seed: {seed} | Instances: {actual_samples_used} "
                        f"| Dropout: {reg_val_dropout} | LR: {lr}"
                    )

                    # Create model + optimizer
                    model = ModelWrapper(
                        n_features=n_features,
                        n_classes=n_classes,
                        dropout_rate=reg_val_dropout,  
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
                    

                    # Train
                    train_loss, train_acc, val_loss, val_acc = model.fit(train_loader, validation_loader, epochs=epochs, paitient_factor=patience)


                    # Store results
                    results.append({
                        "dataset_name": dataset_name,
                        "train_size": actual_samples_used,  # track how many we used this iteration
                        "reg_dropout": reg_val_dropout,
                        "seed": seed,
                        "lr": lr,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "test_loss": val_loss,
                        "test_acc": val_acc
                    })

                    pbar.update(1)

            # Move to the next iteration
            current_samples += n

    pbar.close()
    print(f"Completed processing for dataset: {dataset_name}\n")
    return pd.DataFrame(results), instances_used, len(X_train_full), len(X_test)


def main():
    parser = argparse.ArgumentParser(description="Run experiments with different datasets and regularization types.")
    parser.add_argument("--d", type=int, required=True, help="Dataset index to run.")
    args = parser.parse_args()

    # List of dataset file paths
    dataset_paths = [
        'data/splice/splice.npz',
        'data/protein/protein.npz',
        'data/dna/dna.npz',
        'data/twomoons/twomoons.npz',
    ]

    if args.d < 0 or args.d >= len(dataset_paths):
        tqdm.write(f"Error: Dataset index {args.d} is out of range.")
        exit(1)

    DS_2_RUN = dataset_paths[args.d]
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"logs/{timestamp_str}/"
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    params = {
        "batch_size": 64,
        "epochs": 100,
        "lambda_list_dropout": np.array([0, 0.2, 0.4, 0.6, 0.8], dtype=np.float64),
        "lr_rates": np.array([0.1, 0.01, 0.001, 0.0001, 0.00001]),
        "patience": 10  # Early stopping patience
    }

    tqdm.write("Parameters Configuration:")
    for key, value in params.items():
        tqdm.write(f"{key}: {value}")

    dataset_name = os.path.splitext(os.path.basename(DS_2_RUN))[0]

    try:
        DS = GeneralizedDataset(data_path=DS_2_RUN)
    except FileNotFoundError as e:
        tqdm.write(f"Error: {e}")
        exit(1)
    # Stratified sampling down to 2000 instances
    sss = StratifiedShuffleSplit(n_splits=1, train_size=2000, random_state=42)
    subsample_indices, _ = next(sss.split(DS.x, DS.y))
    dataset_x = DS.x[subsample_indices]
    dataset_y = DS.y[subsample_indices]

    
    n_classes = len(np.unique(dataset_y))
    features = dataset_x.shape[1]

    tqdm.write(f"Applying regularization types: L2 and Dropout")

    start_time = time.time()
    results_df, instances_used, train_set_size, test_set_size = run_experiment(
        data=(dataset_x, dataset_y),
        dataset_name=dataset_name,
        device=device,
        params=params,
        start_time=start_time,
    )

    # Now store instances_used in params
    params["instances_used"] = instances_used

    csv_filename = os.path.join(output_dir, f"{dataset_name}.csv")
    results_df.to_csv(csv_filename, index=False)
    tqdm.write(f"Results saved to '{csv_filename}'.\n")


    best_result = results_df.loc[results_df["test_acc"].idxmax()]
    now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
    total_time = time.time() - start_time
    formatted_total_time = format_time(total_time)
    tqdm.write(f"Total time elapsed: {formatted_total_time}")


    summary_text = f"""
Date and Time: {now_formatted}
Dataset Name: {dataset_name}
Regularization Types: L2 and Dropout
Total Run Time: {formatted_total_time}

### Experiment Parameters:
    Random Seeds: 1-30
    Number of Instances: {params['instances_used']}
    Dropout Values: {params['lambda_list_dropout']}
    Learning Rates: {params['lr_rates']}
    Patience: {params['patience']}

### Training Details:
    Batch Size: {params['batch_size']}
    Epochs: {params['epochs']}

### Performance:
    Best Hyperparameters:
        Data Size Used: {best_result['train_size']}
        Dropout Value: {best_result['reg_dropout']}
        Learning Rate: {best_result['lr']}
    Best Results:
        Train Loss: {best_result['train_loss']:.4f}
        Train Accuracy: {best_result['train_acc'] * 100:.2f}%
        Test Loss: {best_result['test_loss']:.4f}
        Test Accuracy: {best_result['test_acc'] * 100:.2f}%

### Dataset Details:
    Total Samples Available: {len(DS.x)}
    Total Samples Used: 2000
    Train Set Size: 1200
    Test Set Size: 800
    Classes: {n_classes}
    Features: {features}

"""
    summary_filename = os.path.join(output_dir, f"{dataset_name}_summary_{timestamp_str}.txt")
    with open(summary_filename, "w") as f:
        f.write(summary_text)

    tqdm.write(f"Summary saved to '{summary_filename}'.") 


if __name__ == "__main__":
    main()