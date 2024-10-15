from torch.utils.data import Dataset, Subset
import numpy as np
import torch
import torch.nn as nn
from torcheval import metrics
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from torch.utils.data import DataLoader
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
epochs = 50
test_ratio = 0.4
batch_size = 32
random_seed = 42
number_of_log_labmdas = 1#20
number_of_linear_labmdas = 1#10
number_of_random_seeds = 1#10

class TwoMoonsDataset(Dataset):
    def __init__(self) -> None:
        super()
        current_file_path = os.path.abspath(__file__)
        data_path = os.path.join(os.path.dirname(current_file_path), "..", "data/twomoons/TwoMoonsDF.xlsx")
        data_path = os.path.normpath(data_path)
        df = pd.read_excel(data_path)
        df_Y = df['Label']
        df_X = df.drop('Label', axis=1)
        x = df_X.values.astype(np.float32)
        x = MinMaxScaler().fit_transform(x)
        y = df_Y.values.astype(np.int32).reshape(-1, 1)
        y = OneHotEncoder(sparse_output=False).fit_transform(y)
        self.X = torch.tensor(x)
        self.Y = torch.tensor(y)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], self.Y[idx]

def get_l2_reg_grid() -> np.ndarray:
    l2_values_log_round_values = set(np.logspace(-6, -1, num=6))
    l2_values_log = set(np.logspace(-6, 0, num=number_of_log_labmdas))
    l2_values_lin = set(np.linspace(1, 10, num=number_of_linear_labmdas))
    l2_values = l2_values_log | l2_values_log_round_values | l2_values_lin
    l2_values = np.array(sorted(l2_values))
    return l2_values

def set_seed(seed: int = random_seed) -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# Define the neural network architecture
class MLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int, l2_reg: float, dropout_rate: float, should_tune_dropout: bool) -> None:
        super().__init__()
        narrow_factor = 256
        wide_factor = 1024
        self.layers = nn.Sequential()
        # input layer and first hidden layer
        self.layers.add_module(f"dense_0", nn.Linear(n_features, wide_factor))
        self.layers.add_module(f"activation_0", nn.ReLU())
        if should_tune_dropout:
            self.layers.add_module(f"dropout_0", nn.Dropout(dropout_rate))
        # second hidden layer
        self.layers.add_module(f"dense_1", nn.Linear(wide_factor, narrow_factor))
        self.layers.add_module(f"activation_1", nn.ReLU())
        if should_tune_dropout:
            self.layers.add_module(f"dropout_1", nn.Dropout(dropout_rate))
        # third hidden layer
        self.layers.add_module(f"dense_2", nn.Linear(narrow_factor, wide_factor))
        self.layers.add_module(f"activation_2", nn.ReLU())
        if should_tune_dropout:
            self.layers.add_module(f"dropout_2", nn.Dropout(dropout_rate))
        # fourth hidden layer
        self.layers.add_module(f"dense_3", nn.Linear(wide_factor, narrow_factor))
        self.layers.add_module(f"activation_3", nn.ReLU())
        if should_tune_dropout:
            self.layers.add_module(f"dropout_3", nn.Dropout(dropout_rate))
        # fifth hidden layer
        self.layers.add_module(f"dense_4", nn.Linear(narrow_factor, wide_factor))
        self.layers.add_module(f"activation_4", nn.ReLU())
        if should_tune_dropout:
            self.layers.add_module(f"dropout_4", nn.Dropout(dropout_rate))
        # sixth hidden layer
        self.layers.add_module(f"dense_5", nn.Linear(wide_factor, narrow_factor))
        self.layers.add_module(f"activation_5", nn.ReLU())
        if should_tune_dropout:
            self.layers.add_module(f"dropout_5", nn.Dropout(dropout_rate))
        # output layer
        self.layers.add_module(f"dense_6", nn.Linear(narrow_factor, n_classes))
        self.layers.add_module(f"activation_6", nn.Softmax(dim=1))
        # loss function
        self.criterion = nn.CrossEntropyLoss()
        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=l2_reg)
        # metric
        self.metric = metrics.MulticlassAccuracy(num_classes=n_classes)

    def forward(self, x):
        return self.layers(x)

    def calculate_accuracy(self, y_pred, y_true):
        self.metric.update(y_pred, torch.argmax(y_true, dim=1))
        # compute the metric
        accuracy = self.metric.compute()
        return accuracy

def eval(loader: DataLoader, model: MLP) -> tuple:
    '''Equivalent to the test loop. Used by both test and validation.'''
    model.eval()
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for inputs, targets in loader:
            targets = targets.to(device)
            inputs = inputs.to(device)
            predictions = model(inputs)
            batch_loss = model.criterion(predictions, torch.argmax(targets, dim=1))
            total_loss += batch_loss.item()
            total_acc += model.calculate_accuracy(predictions, targets)
    return total_loss/len(loader), total_acc/len(loader)

def train(train_loader: DataLoader, model: MLP) -> tuple:
    '''Equivalent to the training loop'''
    model.train()
    for inputs, targets in train_loader:
        targets = targets.to(device)
        inputs = inputs.to(device)
        predictions = model(inputs)
        loss = model.criterion(predictions, torch.argmax(targets, dim=1))
        loss.backward()
        model.optimizer.step()
        model.zero_grad()

def fit(train_loader: DataLoader, val_loader: DataLoader, model: MLP) -> tuple:
    '''An umbrella method on top of the training and validation loops'''
    consecutive_high_acc = 0  # Counter for consecutive accuracies above 80%
    target_acc = 0.80         # Threshold accuracy to check against
    max_high_acc = 3          # Number of times to allow accuracy > 80% before stopping
    for epoch_num in range(epochs):
        train(train_loader, model)
        val_loss, val_metrics = eval(val_loader, model)
        # Check if validation accuracy exceeds the threshold
        if val_metrics > target_acc:
            consecutive_high_acc += 1
        else:
            consecutive_high_acc = 0  # Reset if accuracy drops below threshold
        # If validation accuracy exceeds the threshold 3 times, stop early
        if consecutive_high_acc >= max_high_acc:
            break
    return val_loss, val_metrics
    
def main():
    # handling data
    set_seed()
    dataset = TwoMoonsDataset()
    idx_train, idx_val = train_test_split(np.arange(len(dataset)), test_size=test_ratio)
    val_loader = DataLoader(Subset(dataset, idx_val), batch_size=batch_size, shuffle=False)

    # values
    d_values = []
    l2_lambda_values = []
    r_s_values = []
    val_loss_values = []
    val_metrics_values = []
    # algorithm start
    set_seed()
    with tqdm(total=number_of_random_seeds, desc="Algorithm Progress") as pbar:
        for r_s in np.random.randint(0, 100, number_of_random_seeds):
            set_seed(r_s.item())
            for d in [1, 2, 5, 10, 20, 50, 100]:
                chunck_size = np.int32(np.floor(len(idx_train) * (d/100)))
                chunck_idx = np.random.choice(idx_train, chunck_size, replace=False)
                train_loader = DataLoader(Subset(dataset, chunck_idx), batch_size=batch_size, shuffle=True)
                # L2 regularization grid search
                for l2_lambda in get_l2_reg_grid():
                    # handling model
                    model = MLP(n_features=dataset.X.shape[1], 
                                n_classes=dataset.Y.shape[1],
                                l2_reg=l2_lambda,
                                dropout_rate=0,
                                should_tune_dropout=False)    
                    val_loss, val_metrics = fit(train_loader, val_loader, model)
                    # saving values
                    d_values.append(d)
                    l2_lambda_values.append(l2_lambda)
                    r_s_values.append(r_s)
                    val_loss_values.append(val_loss)
                    val_metrics_values.append(val_metrics)
                    pbar.set_postfix({
                        "d": d,
                        "l2_lambda": l2_lambda
                    })
        pbar.update(1)
    # saving results
    results = pd.DataFrame({
        "d": d_values,
        "l2_lambda": l2_lambda_values,
        "r_s": r_s_values,
        "val_loss": val_loss_values,
        "val_metrics": val_metrics_values
    })
    results.to_csv("hyperparameters_2moons_results.csv", index=False, header=True)


if __name__ == "__main__":
    main()   
