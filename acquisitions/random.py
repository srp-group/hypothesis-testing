import numpy as np
# The Acquisition function
class Random():
    def __init__(self, pool: 'core.Pool') -> None:
        self.pool = pool
    def query(self) -> int:
        all_unlabelled_indecies = self.pool.get_unlabeled_indecies()
        all_scores = np.random.random(len(all_unlabelled_indecies))
        max_scores = np.argwhere(np.isclose(all_scores, all_scores.max())).ravel()            
        self.pool.set_seed()
        idx = np.random.choice(max_scores, 1)[0]
        return all_unlabelled_indecies[idx]