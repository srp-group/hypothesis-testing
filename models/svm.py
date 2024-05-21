# Source: https://github.com/nathanlem1/SVM_PyTorch/blob/master/SVM_PyTorch_train.py#L16
import torch
import torch.nn as nn
from torcheval import metrics

class SVM(nn.Module):
    def __init__(self, n_features: int, n_classes: int, l2_reg: float) -> None:
        super().__init__()
        # Define a single linear layer
        self.linear = nn.Linear(n_features, n_classes)
        self.criterion = nn.MultiMarginLoss() # Multi-class classification hinge loss (margin-based loss); SVM
        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=l2_reg)
        self.metric = metrics.MulticlassAccuracy(num_classes=n_classes)

    def forward(self, x):
        # Apply the linear layer
        x = self.linear(x)
        return x

    def calculate_accuracy(self, y_pred, y_true):
        self.metric.update(y_pred, torch.argmax(y_true, dim=1))
        # Compute the metric
        accuracy = self.metric.compute()
        return accuracy


