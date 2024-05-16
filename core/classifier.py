import torch
from torch.utils.data import  DataLoader
from models import MLP
import optuna
import core
from typing import Optional
from optuna.trial import Trial
import logging

# Set logging level to ERROR - For a more clean output terminal
optuna.logging.set_verbosity(optuna.logging.ERROR)

class Classifier():
    def __init__(self, pool: core.Pool) -> None:
        self.pool = pool
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = int(self.pool.dataset_config['epochs'])

    def eval(self, loader: DataLoader, model: MLP) -> tuple:
        model.eval()
        total_loss = 0
        total_acc = 0
        with torch.no_grad():
            for inputs, targets in loader:
                targets = targets.to(self.device)
                inputs = inputs.to(self.device)
                predictions = model(inputs)
                batch_loss = model.criterion(predictions, torch.argmax(targets, dim=1))
                total_loss += batch_loss.item()
                total_acc += model.calculate_accuracy(predictions, targets)  
        return total_loss/len(loader), total_acc/len(loader)
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader, model: MLP, trial: Optional[Trial] = None) -> tuple:
        model.train()
        for epoch_num in range(self.epochs):
            for inputs, targets in train_loader:
                targets = targets.to(self.device)
                inputs = inputs.to(self.device)
                predictions = model(inputs)
                loss = model.criterion(predictions, torch.argmax(targets, dim=1))
                model.zero_grad()
                loss.backward()
                model.optimizer.step()
            train_loss, train_metrics = self.eval(train_loader, model)
            val_loss, val_metrics = self.eval(val_loader, model)
            if trial:
                trial.report(val_metrics, epoch_num)
                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
        return (train_loss, train_metrics),  (val_loss, val_metrics)
    
    def objective(self, trial: optuna.trial) -> float:
        dropout_rate = trial.suggest_float("dropout_rate", 1e-3, 0.5)
        l2_reg = trial.suggest_float("l2_reg", 1e-6, 1e-1, log=True)
        # CREATE MODEL
        if(self.pool.dataset_name.lower() == "dna"):
            model = MLP(int(self.pool.dataset_config['n_features']),
                            int(self.pool.dataset_config['n_classes']),
                            l2_reg,
                            dropout_rate).to(self.device)
        train_loader, val_loader, test_loader = self.pool.get_loaders()
        (train_loss, train_metrics),  (val_loss, val_metrics) = self.fit(train_loader, val_loader, model, trial)
        return val_loss

    def tune(self) -> None:
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=int(self.pool.dataset_config['n_trials']))
        best_dropout_rate = study.best_params['dropout_rate']
        best_l2_reg = study.best_params['l2_reg']
        best_val_loss = study.best_value
        return best_dropout_rate, best_l2_reg, best_val_loss
    
    def test(self, l2_reg: float, dropout_rate: float) -> tuple:
        # CREATE MODEL
        if(self.pool.dataset_name.lower() == "dna"):
            model = MLP(int(self.pool.dataset_config['n_features']),
                            int(self.pool.dataset_config['n_classes']),
                            l2_reg,
                            dropout_rate).to(self.device)
        model.eval()
        train_loader, test_loader = self.pool.get_test_loaders()
        (train_loss, train_metrics), (test_loss, test_metrics) = self.fit(
            train_loader, test_loader, model
        )
        return test_loss, test_metrics
