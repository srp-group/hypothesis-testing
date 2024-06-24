
import torch
import torch.nn as nn
from torcheval import metrics


# Define the neural network architecture
class MLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int, l2_reg: float, dropout_rate: float) -> None:
        super().__init__()
        self.layers = nn.Sequential()
        self.layers.add_module(f"dense_0", nn.Linear(n_features, 64))
        self.layers.add_module(f"activation_0", nn.ReLU())
        self.layers.add_module(f"dropout_0", nn.Dropout(dropout_rate))
        self.layers.add_module(f"dense_1", nn.Linear(64, 64))
        self.layers.add_module(f"activation_1", nn.ReLU())
        self.layers.add_module(f"dense_4", nn.Linear(64, n_classes))
        self.layers.add_module(f"activation_4", nn.Softmax(dim=1))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=l2_reg)
        self.metric = metrics.MulticlassAccuracy(num_classes=n_classes)

    def forward(self, x):
        return self.layers(x)

    def calculate_accuracy(self, y_pred, y_true):
        self.metric.update(y_pred, torch.argmax(y_true, dim=1))
        # compute the metric
        accuracy = self.metric.compute()
        return accuracy


