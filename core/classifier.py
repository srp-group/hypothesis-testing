import numpy as np
import torch
from torch.utils.data import  DataLoader
from models import MLP, MLR, SVM
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
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu") # my GPU is slower than CPU LOL
        self.epochs = int(self.pool.dataset_config['epochs'])
        self.model_name = str(self.pool.dataset_config['model_name'])
        self.l2_reg = None
        self.dropout_rate = None
        self.model_state = None

    def init_model(self) -> MLP | MLR | SVM: # no repetition of code
        if self.model_name == 'MLP':
            model = MLP(int(self.pool.dataset_config['n_features']),
                        int(self.pool.dataset_config['n_classes']),
                        self.l2_reg,
                        self.dropout_rate).to(self.device)
        elif self.model_name == 'MLR':
            model = MLR(int(self.pool.dataset_config['n_features']),
                        int(self.pool.dataset_config['n_classes']),
                        self.l2_reg).to(self.device)
        elif self.model_name == 'SVM':
            model = SVM(int(self.pool.dataset_config['n_features']),
                        int(self.pool.dataset_config['n_classes']),
                        self.l2_reg).to(self.device)
        return model

    def get_model(self) -> MLP | MLR | SVM:
        if not self.model_state: 
            model = self.init_model()
            self.model_state = model.state_dict()
        model = self.init_model()
        model.load_state_dict(self.model_state)
        if self.model_name == 'MLP':
            model.set_dropout_rate(self.dropout_rate)
        model.set_l2_reg(self.l2_reg)
        return model

    def eval(self, loader: DataLoader, model: MLP | MLR | SVM ) -> tuple:
        '''Equivalent to the test loop. Used by both test and validation.'''
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
    
    def train(self, train_loader: DataLoader, model: MLP | MLR | SVM) -> tuple:
        '''Equivalent to the training loop'''
        model.train()
        for inputs, targets in train_loader:
            targets = targets.to(self.device)
            inputs = inputs.to(self.device)
            predictions = model(inputs)
            loss = model.criterion(predictions, torch.argmax(targets, dim=1))
            loss.backward()
            model.optimizer.step()
            model.zero_grad()
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader, model: MLP | MLR | SVM) -> tuple:
        '''An umbrella method on top of the training and validation loops'''
        total_val_loss = 0
        total_val_metrics = 0
        for epoch_num in range(self.epochs):
            self.train(train_loader, model)
            val_loss, val_metrics = self.eval(val_loader, model)
            total_val_loss += val_loss
            total_val_metrics += val_metrics
        return total_val_loss/self.epochs, total_val_metrics/self.epochs

    def objective(self, trial: optuna.trial) -> float:
        # CREATE MODEL      
        self.l2_reg = trial.suggest_float("l2_reg", 1e-6, 1e-1, log=True)
        if self.model_name == 'MLP':
            self.dropout_rate = trial.suggest_float("dropout_rate", 1e-3, 0.5)
        elif self.model_name == 'MLR' or self.model_name == 'SVM':
            self.dropout_rate = 0
        validation_loss = []
        # iterate through each fold. train on fold and test on test set
        for fold_index, train_fold in enumerate(self.pool.get_folds()):
            train_loader, val_loader, test_loader = self.pool.get_loaders(train_fold)
            model = self.get_model()
            val_loss, val_metrics = self.fit(train_loader, val_loader, model)
            validation_loss.append(val_loss) # save validation loss for each fold
            avg_val_loss = np.mean(validation_loss)
            if trial:
                # Report the intermediate results to Optuna
                trial.report(val_loss, fold_index)
        return avg_val_loss

    def tune(self) -> None:
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=int(self.pool.dataset_config['n_trials']))
        if self.model_name == 'MLP':
            best_dropout_rate = study.best_params['dropout_rate']
        elif self.model_name == 'MLR' or self.model_name == 'SVM':
            best_dropout_rate = 0
        best_l2_reg = study.best_params['l2_reg']
        best_val_loss = study.best_value
        return best_dropout_rate, best_l2_reg, best_val_loss
    
    def test(self, l2_reg: float, dropout_rate: float) -> tuple:
        # setting the best hyperparameters
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        # getting the model
        model = self.get_model()
        # getting the loaders
        train_loader, test_loader = self.pool.get_test_loaders()
        # training the model and testing on the test set
        (test_loss, test_metrics) = self.fit(
            train_loader, test_loader, model
        )
        return test_loss, test_metrics
    
    def probability(self, x):   
        model = self.init_model()
        model.train()
        with torch.no_grad():
            a = model(x.to(self.device))
            return model(x.to(self.device))