import argparse
import datetime
import os
import random
import time  # For time tracking
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm


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
    def __init__(self, n_features, n_classes, dropout_rate=0.5, use_dropout=False):
        """
        Initializes the MLP model.

        Parameters:
        - n_features (int): Number of input features.
        - n_classes (int): Number of output classes.
        - dropout_rate (float): Dropout rate.
        - use_dropout (bool): Whether to include dropout layers.
        """
        super().__init__()

        self.model_type = "MLP"
        self.loss_function = "CrossEntropyLoss"

        ######################################################################### MODEL ARCHITECTURE 

        layer_sizes = [
                    (n_features, 1024),
                    (1024, 256),
                    (256, 1024),
                    (1024, 256),
                    (256, 1024),
                    (1024, 256)
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
        #     (2048, 512),
        # ]

        layers = []
        for in_size, out_size in layer_sizes:
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
            if use_dropout:
                layers.append(nn.Dropout(dropout_rate))

        # Output layer without Softmax
        layers.append(nn.Linear(layer_sizes[-1][1], n_classes))

        self.layers = nn.Sequential(*layers)
        self.criterion = nn.CrossEntropyLoss()

        # Store architecture details for summary
        self.architecture = []
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                self.architecture.append(f"Linear({layer.in_features} -> {layer.out_features})")
            elif isinstance(layer, nn.ReLU):
                self.architecture.append("ReLU")
            elif isinstance(layer, nn.Dropout):
                self.architecture.append(f"Dropout(p={layer.p})")

    def forward(self, x):
        return self.layers(x)

    def calculate_correct(self, y_pred, y_true):
        predicted_classes = y_pred.argmax(dim=1)
        correct = (predicted_classes == y_true).sum().item()
        return correct


class ModelWrapper:
    def __init__(
        self,
        reg_type,
        reg_val,
        n_features,
        n_classes,
        dropout_rate=0.5,
        device="cpu",
        lr=0.001,
    ):
        self.device = device
        reg_type = reg_type.lower()
        if reg_type == "dropout":
            use_dropout = True
            weight_decay = 0.0  # Disable L2 regularization when using dropout
            dropout_rate = reg_val  # Use reg_val as the dropout rate
            optimizer_type = "Adam"  # Assuming Adam is used
        elif reg_type == "l2":
            use_dropout = False
            weight_decay = reg_val
            optimizer_type = "Adam"  # Assuming Adam is used
        else:
            raise ValueError(f"Unsupported regularization type: {reg_type}")

        self.optimizer_type = optimizer_type  # Store optimizer type
        self.loss_function = "CrossEntropyLoss"  # Store loss function

        self.model = MLP(
            n_features=n_features,
            n_classes=n_classes,
            dropout_rate=dropout_rate,
            use_dropout=use_dropout,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), weight_decay=weight_decay, lr=lr
        )
        self.reg_type = reg_type
        self.reg_val = reg_val

    def eval(self, loader):
        """Equivalent to the test loop. Used by both test and validation."""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, targets in loader:
                targets = targets.to(self.device)
                inputs = inputs.to(self.device)
                predictions = self.model(inputs)
                batch_loss = self.model.criterion(predictions, targets)
                total_loss += batch_loss.item() * targets.size(0)
                total_correct += self.model.calculate_correct(predictions, targets)
                total_samples += targets.size(0)
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return avg_loss, avg_acc

    def train(self, train_loader):
        """Equivalent to the training loop"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for inputs, targets in train_loader:
            targets = targets.to(self.device)
            inputs = inputs.to(self.device)
            predictions = self.model(inputs)
            loss = self.model.criterion(predictions, targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * targets.size(0)
            total_correct += self.model.calculate_correct(predictions, targets)
            total_samples += targets.size(0)
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return avg_loss, avg_acc

    def fit(self, train_loader, test_loader, epochs):
        """An umbrella method on top of the training and validation loops"""
        for _ in tqdm(range(epochs), desc="Training Epochs", leave=False):
            self.train(train_loader)
        # Evaluate on training data
        train_loss, train_acc = self.eval(train_loader)
        # Evaluate on test data
        test_loss, test_acc = self.eval(test_loader)
        return train_loss, train_acc, test_loss, test_acc


def format_time(seconds):
    """Formats time in seconds to HH:MM:SS."""
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def run_experiment(data, dataset_name, reg_type, device, params, start_time):
    # Extract parameters
    batch_size = params["batch_size"]
    epochs = params["epochs"]
    seeds = params["seeds"]
    dataset_sizes_pct = params["dataset_sizes_pct"]
    l_rates = params["l_rates"]

    # Define the list of regularization parameters
    if reg_type.lower() == "l2":
        lambda_list = params["lambda_list_l2"]
    else:
        lambda_list = params["lambda_list_dropout"]
        tqdm.write(f"Dropout values: {lambda_list}")

    # Initialize a list to store all results
    results = []

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
        stratify=y if n_classes > 2 else None,  # Stratify for classification balance
    )
    X_train_full, y_train_full = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Calculate total number of iterations for the progress bar
    total_iterations = (
        len(seeds) * len(dataset_sizes_pct) * len(lambda_list) * len(l_rates)
    )
    pbar = tqdm(total=total_iterations, desc="Total Progress")

    # Iterate over each seed for reproducibility
    for seed in seeds:
        # Setting the seed for reproducibility
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
                len(X_train_full), size=n_samples, replace=False
            )
            X_train_d, y_train_d = X_train_full[sampled_indices], y_train_full[
                sampled_indices
            ]
            # Iterate over each regularization parameter
            for reg_val in lambda_list:
                for lr in l_rates:
                    tqdm.write(
                        f"Seed: {seed}, Size: {d}%, Reg Val: {reg_val}, LR: {lr}"
                    )
                    # Initialize the model with the current regularization parameters
                    model = ModelWrapper(
                        reg_type=reg_type,
                        reg_val=reg_val,
                        n_features=n_features,
                        n_classes=n_classes,
                        dropout_rate=0.5,  # Default dropout rate; overridden if reg_type is 'dropout'
                        device=device,
                        lr=lr,
                    )
                    train_dataset = TensorDataset(
                        torch.tensor(X_train_d, dtype=torch.float32),
                        torch.tensor(y_train_d, dtype=torch.long),
                    )

                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=min(batch_size, len(train_dataset)),
                        shuffle=True,
                    )

                    test_dataset = TensorDataset(
                        torch.tensor(X_test, dtype=torch.float32),
                        torch.tensor(y_test, dtype=torch.long),
                    )
                    test_loader = DataLoader(
                        test_dataset,
                        batch_size=min(batch_size, len(test_dataset)),
                        shuffle=False,
                    )

                    try:
                        train_loss, train_acc, test_loss, test_acc = model.fit(
                            train_loader, test_loader, epochs=epochs
                        )
                    except Exception as e:
                        tqdm.write(f"Error during training: {e}")
                        pbar.update(1)
                        continue  # Skip to the next combination

                    # Store the results
                    results.append(
                        {
                            "dataset_name": dataset_name,
                            "reg_type": reg_type,
                            "reg_val": reg_val,
                            "seed": seed,
                            "data_size_pct": d,
                            "lr": lr,
                            "train_loss": train_loss,
                            "train_acc": train_acc,
                            "test_loss": test_loss,
                            "test_acc": test_acc,
                        }
                    )

                    # Calculate and print elapsed time after each combination
                    elapsed_time = time.time() - start_time
                    formatted_elapsed_time = format_time(elapsed_time)
                    tqdm.write(f"Time elapsed: {formatted_elapsed_time}\n")

                    pbar.update(1)

    pbar.close()
    tqdm.write(f"Completed processing for dataset: {dataset_name}\n")

    # Convert the results list to a pandas DataFrame
    results_df = pd.DataFrame(results)

    return results_df


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run experiments with different datasets and regularization types."
    )
    parser.add_argument("--d", type=int, required=True, help="Dataset index to run.")
    parser.add_argument(
        "--r", type=int, required=True, help="Regularization type index to run."
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save the results."
    )

    args = parser.parse_args()

    # List of dataset file paths ########################################################################  DATASETS AND REGULARIZATION
    dataset_paths = [
        # r"C:\Users\canel\OneDrive\Desktop\SRP\hypothesis-testing3\hypothesis-testing\data\splice\splice.npz"
        r"C:\Users\canel\OneDrive\Desktop\SRP\hypothesis-testing3\hypothesis-testing\data\protein\protein.npz",
        # r"C:\Users\canel\OneDrive\Desktop\SRP\hypothesis-testing3\hypothesis-testing\data\dna\dna.npz",
        # "data/twomoons/twomoons.npz",
        # "data/electricalFault/detect.npz",
        # "data/pokerdataset/poker.npz",
    ]

    reg_types = [
        "l2",
        "dropout",
    ]

    # Validate dataset index
    if args.d < 0 or args.d >= len(dataset_paths):
        tqdm.write(f"Error: Dataset index {args.d} is out of range.")
        exit(1)

    # Validate regularization type index
    if args.r < 0 or args.r >= len(reg_types):
        tqdm.write(f"Error: Regularization type index {args.r} is out of range.")
        exit(1)

    DS_2_RUN = dataset_paths[args.d]
    REG_2_RUN = reg_types[args.r]
    output_dir = args.output_dir

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Specify the device ('cpu' or 'cuda' if GPU is available)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get current date and time for filenames
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")

    # Define the parameters ################################################################## TRAINING PARAMETERS
    params = {
        "batch_size": 128,
        "epochs": 25,
        "seeds": list(range(1, 15)),
        "dataset_sizes_pct": [100],
        "lambda_list_l2": np.array([0, 0.1, 0.01]),
        "lambda_list_dropout": np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9], dtype=np.float32),
        "l_rates": np.array([0.001]),
    }

    tqdm.write("Parameters Configuration:")
    for key, value in params.items():
        tqdm.write(f"{key}: {value}")

    # Extract dataset name from the file path
    dataset_name = os.path.splitext(os.path.basename(DS_2_RUN))[0]

    # Load the current dataset with error handling
    try:
        DS = GeneralizedDataset(data_path=DS_2_RUN)
    except FileNotFoundError as e:
        tqdm.write(f"Error: {e}")
        exit(1)
    dataset_x, dataset_y = DS.x, DS.y

    reg_type = REG_2_RUN
    tqdm.write(f"Applying regularization type: {reg_type}\n")

    # Record the start time
    start_time = time.time()

    # Run the experiment for the current dataset and regularization type
    results_df = run_experiment(
        data=(dataset_x, dataset_y),
        dataset_name=dataset_name,
        reg_type=reg_type,
        device=device,
        params=params,
        start_time=start_time,  # Pass start_time to track elapsed time
    )

    # Define the CSV filename based on dataset, regularization type, and timestamp
    csv_filename = os.path.join(
        output_dir, f"{dataset_name}_{reg_type}_results_{timestamp_str}.csv"
    )

    # Save the results to the CSV file
    results_df.to_csv(csv_filename, index=False)
    tqdm.write(f"Results saved to '{csv_filename}'.\n")

    # Find the best result based on highest test accuracy
    if results_df.empty:
        tqdm.write("No results to summarize.")
        exit(1)
    best_result = results_df.loc[results_df["test_acc"].idxmax()]

    # Get current date and time
    now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")

    # Calculate total time taken
    total_time = time.time() - start_time
    formatted_total_time = format_time(total_time)
    tqdm.write(f"Total time elapsed: {formatted_total_time}")

    # Prepare the summary text with enhanced details
    # Extract architecture details dynamically from the model
    # Since `run_experiment` creates multiple models, we'll instantiate one to extract architecture
    sample_model = ModelWrapper(
        reg_type=reg_type,
        reg_val=best_result["reg_val"],
        n_features=dataset_x.shape[1],
        n_classes=len(np.unique(dataset_y)),
        dropout_rate=0.5,
        device=device,
        lr=best_result["lr"],
    )
    architecture = sample_model.model.architecture

    # Calculate Grid Search Combinations
    if reg_type == "l2":
        grid_combinations = (
            len(params["seeds"])
            * len(params["dataset_sizes_pct"])
            * len(params["lambda_list_l2"])
            * len(params["l_rates"])
        )
    else:
        grid_combinations = (
            len(params["seeds"])
            * len(params["dataset_sizes_pct"])
            * len(params["lambda_list_dropout"])
            * len(params["l_rates"])
        )

    summary_text = f"""
Date and Time: {now_formatted}
Dataset Name: {dataset_name}
Regularization Type: {reg_type}
Total Run Time: {formatted_total_time}

### Experiment Parameters:
    Random Seeds: {params['seeds']}
    Dataset Sizes: {params['dataset_sizes_pct']}%
    Regularization Values: {params['lambda_list_l2'].tolist() if reg_type == 'l2' else params['lambda_list_dropout'].tolist()}
    Learning Rates: {params['l_rates'].tolist()}
    Total Combinations: {grid_combinations}

### Training Details:
    Batch Size: {params['batch_size']}
    Epochs: {params['epochs']}

### Performance:
    Best Hyperparameters:
        Seed: {best_result['seed']}
        Data Size Percentage: {best_result['data_size_pct']}%
        Regularization Value: {best_result['reg_val']}
        Learning Rate: {best_result['lr']}
    Best Results:
        Train Loss: {best_result['train_loss']:.4f}
        Train Accuracy: {best_result['train_acc'] * 100:.2f}%
        Test Loss: {best_result['test_loss']:.4f}
        Test Accuracy: {best_result['test_acc'] * 100:.2f}%

### Dataset Details:
    Total Samples: {len(dataset_x)}
    Training Split: 60%
    Testing Split: 40%
        
### Notes:
    Write any comments about the experiment here.

    
### Model Details:
    Model Type: {sample_model.model.model_type}
    Model Architecture:
"""
    for layer in architecture:
        summary_text += f"    - {layer}\n"
    summary_text += f"    Optimizer: {sample_model.optimizer_type}\n"
    summary_text += f"    Loss Function: {sample_model.model.loss_function}\n\n"
    summary_text += f"""

"""

    # Define the summary filename with timestamp
    summary_filename = os.path.join(
        output_dir, f"{dataset_name}_{reg_type}_summary_{timestamp_str}.txt"
    )

    # Save the summary to the text file
    with open(summary_filename, "w") as f:
        f.write(summary_text)

    tqdm.write(f"Summary saved to '{summary_filename}'.") 


if __name__ == "__main__":
    main()
