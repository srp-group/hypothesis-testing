import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset
from datasets import DnaDataset, SpliceDataset, ProteinDataset
import torch
from sklearn.model_selection import train_test_split
import math
from typing import Optional
from configparser import SectionProxy
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
        # setting the indecies
        self.set_seed()
        self.idx_abs = np.arange(len(self.dataset)) # absolute indecies

        # static hold-out testing  
        # 1 setting the train, validation and test indecies
        validation_ratio = float(self.dataset_config['val_share'])
        test_ratio = float(self.dataset_config['test_share'])
        self.set_seed()
        self.idx_train, rest_data = train_test_split(self.idx_abs, test_size=validation_ratio+test_ratio)
        self.set_seed()
        self.idx_test, self.idx_val = train_test_split(rest_data, test_size=validation_ratio/(test_ratio+validation_ratio))


        # setting the labeled indecies
        initial_lb_size = int(self.dataset_config['labeled_share']) # initial labeled size is always number, not percentage
        self.set_seed()
        self.idx_label = np.random.choice(self.idx_train, size=math.floor(initial_lb_size), replace=False) # original labeled indecies
        if(float(self.dataset_config['budget'])<1):
            self.max_budget = math.floor(float(self.dataset_config['budget']) * len(self.idx_train)) - initial_lb_size # if maximum budget is decimal, it is percentage
        else:
            self.max_budget = int(self.dataset_config['budget']) # if maximum budget is fixed number

        self.get_unlabeled_indecies()

        print(f"Dataset: {len(self.dataset)}")
        print(f"Train: {len(self.idx_train)}")
        print(f"Validation: {len(self.idx_val)}")
        print(f"Test: {len(self.idx_test)}")
        print(f"Initial labeled data: {len(self.idx_label)}")
        print(f"Budget: {self.max_budget}")

        

    def set_seed(self, seed: Optional[int] = None) -> None:
        if seed is None:
            seed = self.random_seed
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    
    def get_loaders(self) -> tuple:
        '''Returns the train, validation and test loaders respectively.'''
        train_loader = DataLoader(Subset(self.dataset, self.idx_label), batch_size=int(self.dataset_config['batch_size']), shuffle=True)
        val_loader = DataLoader(Subset(self.dataset, self.idx_val), batch_size=int(self.dataset_config['batch_size']), shuffle=False)
        test_loader = DataLoader(Subset(self.dataset, self.idx_test), batch_size=int(self.dataset_config['batch_size']), shuffle=False)
        return train_loader, val_loader, test_loader
    
    def get_test_loaders(self) -> tuple:
        '''Returns the train and test loaders respectively. train is consisted of the union train and validation sets.'''
        train_loader = DataLoader(Subset(self.dataset, np.concatenate((
            self.idx_label, self.idx_val))), batch_size=int(self.dataset_config['batch_size']), shuffle=True)
        test_loader = DataLoader(Subset(self.dataset, self.idx_test), batch_size=int(self.dataset_config['batch_size']), shuffle=False)
        return train_loader, test_loader
    
    def add_labeled_data(self, idx: int) -> None:
        '''Add labeled data to the pool.'''
        self.idx_label = np.concatenate((self.idx_label, [idx]))

    def get_unlabeled_indecies(self) -> np.ndarray:
        '''Returns the unlabeled data.'''
        return np.setdiff1d(self.idx_train, self.idx_label)
