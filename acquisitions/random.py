import numpy as np
import torch
# The Acquisition function
class Random():
    def __init__(self, pool: 'core.Pool') -> None:
        self.pool = pool
    def query(self, best_model: torch.nn.Module) -> int:
        all_unlabelled_indecies = self.pool.get_unlabeled_indecies()
        print(f"current number of unlabeled data: {len(all_unlabelled_indecies)}") 
        all_scores = np.random.random(len(all_unlabelled_indecies))
        max_scores = np.argmax(all_scores)
        idx = np.random.choice(max_scores, 1)[0]
        return all_unlabelled_indecies[idx]