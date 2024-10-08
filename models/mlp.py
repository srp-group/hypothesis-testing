
import torch
import torch.nn as nn
from torcheval import metrics


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


