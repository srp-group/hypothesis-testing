
import torch
import torch.nn as nn
from torcheval import metrics


# Define the neural network architecture
class MLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int, l2_reg: float, dropout_rate: float) -> None:
        super().__init__()
        self.layers = nn.Sequential()
        # input layer
        self.layers.add_module(f"dense_0", nn.Linear(n_features, 1024))
        self.layers.add_module(f"activation_0", nn.ReLU())
        self.layers.add_module(f"dropout_0", nn.Dropout(dropout_rate))
        # first hidden layer
        self.layers.add_module(f"dense_1", nn.Linear(1024, 1024))
        self.layers.add_module(f"activation_1", nn.ReLU())
        self.layers.add_module(f"dropout_1", nn.Dropout(dropout_rate))
        # second hidden layer
        self.layers.add_module(f"dense_2", nn.Linear(1024, 1024))
        self.layers.add_module(f"activation_2", nn.ReLU())
        self.layers.add_module(f"dropout_2", nn.Dropout(dropout_rate))
        # output layer
        self.layers.add_module(f"dense_4", nn.Linear(1024, n_classes))
        self.layers.add_module(f"activation_4", nn.Softmax(dim=1))
        # loss function
        self.criterion = nn.CrossEntropyLoss()
        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=l2_reg)
        # metric
        self.metric = metrics.MulticlassAccuracy(num_classes=n_classes)
        # initialize the weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight) 

    def set_dropout_rate(self, dropout_rate: float):
        for layer in self.layers:
            if isinstance(layer, nn.Dropout):
                layer.p = dropout_rate

    def set_l2_reg(self, l2_reg: float):
        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=l2_reg)

    def forward(self, x):
        return self.layers(x)

    def calculate_accuracy(self, y_pred, y_true):
        self.metric.update(y_pred, torch.argmax(y_true, dim=1))
        # compute the metric
        accuracy = self.metric.compute()
        return accuracy


