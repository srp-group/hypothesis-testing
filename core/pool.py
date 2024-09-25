import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset
from datasets import DnaDataset, SpliceDataset, ProteinDataset, TwoMoonsDataset
import torch
import random
from sklearn.model_selection import train_test_split
import math
from typing import Optional
from configparser import SectionProxy
from typing import List, Tuple

class Pool():
    def __init__(self, dataset_name:str, random_seed: int, database_config: SectionProxy, default_config: SectionProxy) -> None:
        self.dataset_name = dataset_name
        self.dataset_config = database_config
        self.default_config = default_config
        self.random_seed = random_seed
        # setting the dataset
        if dataset_name.lower() == 'dna':
            self.dataset = DnaDataset(self.dataset_config)
        elif dataset_name.lower() == 'splice':
            self.dataset = SpliceDataset(self.dataset_config)
        elif dataset_name.lower() == 'protein':
            self.dataset = ProteinDataset(self.dataset_config)
        elif dataset_name.lower() == 'twomoons':
            self.dataset = TwoMoonsDataset(self.dataset_config)
        # setting the indecies
        self.idx_abs = np.arange(len(self.dataset)) # absolute indecies
        self.idx_newly_labeled = np.array([], dtype=int) # newly labeled indecies
        # static hold-out testing  
        test_ratio = float(self.dataset_config['test_share'])
        self.set_seed()
        self.idx_train, self.idx_test = train_test_split(self.idx_abs, test_size=test_ratio)
        # setting the labeled indecies
        initial_lb_size = int(self.dataset_config['labeled_share']) # initial labeled size is always number, not percentage
        # setting folding configuration
        self.initially_labeled_in_fold_size = math.floor(initial_lb_size*float(self.dataset_config['initially_labeled_in_fold_ratio']))
        self.no_folds = int(self.default_config['no_folds'])
        self.set_seed()
        self.idx_ini_label = np.random.choice(self.idx_train, size=math.floor(initial_lb_size), replace=False)  # original labeled indecies
        if(float(self.dataset_config['budget'])<1):
            self.max_budget = math.floor(float(self.dataset_config['budget']) * len(self.idx_train)) - initial_lb_size # if maximum budget is decimal, it is percentage
        else:
            self.max_budget = int(self.dataset_config['budget']) # if maximum budget is fixed number

        print(f"Dataset: {len(self.dataset)}")
        print(f"Initial labeled data: {len(self.idx_ini_label)}")
        print(f"Test: {len(self.idx_test)}")
        print(f"Budget: {self.max_budget}")



    def set_seed(self, seed: Optional[int] = None) -> None:
        if seed is None:
            seed = self.random_seed
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    def get_loaders(self, train_fold: np.ndarray, val_fold: np.ndarray) -> tuple:
        '''Returns the train, validation and test loaders respectively. The training fold should be passed as an argument.
        whuch is an array of indecies.'''
        train_loader = DataLoader(Subset(self.dataset, train_fold), batch_size=int(self.dataset_config['batch_size']), shuffle=True)
        val_loader = DataLoader(Subset(self.dataset, val_fold), batch_size=int(self.dataset_config['batch_size']), shuffle=False)
        test_loader = DataLoader(Subset(self.dataset, self.idx_test), batch_size=int(self.dataset_config['batch_size']), shuffle=False)
        return train_loader, val_loader, test_loader
    
    def get_test_loaders(self) -> tuple:
        '''Returns the train and test loaders respectively. train is consisted of the union train and validation sets.'''
        idx_train = np.concatenate((self.idx_ini_label, self.idx_newly_labeled))
        train_loader = DataLoader(Subset(self.dataset, idx_train), batch_size=int(self.dataset_config['batch_size']), shuffle=True)
        test_loader = DataLoader(Subset(self.dataset, self.idx_test), batch_size=int(self.dataset_config['batch_size']), shuffle=False)
        return train_loader, test_loader
    
    def add_newly_labeled_data(self, idx:int):
        self.idx_newly_labeled = np.concatenate((self.idx_newly_labeled, [idx]))

    def get_unlabeled_indecies(self) -> np.ndarray:
        '''Returns the unlabeled data.'''
        idx_labelled = np.concatenate((self.idx_ini_label, self.idx_newly_labeled))
        return np.setdiff1d(self.idx_train, idx_labelled)
    
    def get_folds(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        folds = []
        self.set_seed()
        for _ in range(self.no_folds):
            fold_indecies = np.random.choice(self.idx_ini_label, size=self.initially_labeled_in_fold_size, replace=False)
            validation_fold_indecies = np.setdiff1d(self.idx_ini_label, fold_indecies)
            fold_indecies = np.concatenate((fold_indecies, self.idx_newly_labeled))
            folds.append((fold_indecies, validation_fold_indecies))
        return folds
