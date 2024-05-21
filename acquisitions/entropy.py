import numpy as np
import torch
# The Acquisition function
class Entropy():
    def __init__(self, pool: 'core.Pool', clf: 'core.Classifier') -> None:
        self.pool = pool
        self.clf = clf
    
    def get_scores(self, values=None):
        values = self.pool.dataset.x[self.pool.get_unlabeled_indecies()]
        probs = self.clf.probability(torch.Tensor(values)).cpu()
        log_probs = torch.log(probs + torch.finfo(torch.float32).smallest_normal)
        U = -(probs*log_probs).sum(axis=1)
        return U

    def query(self) -> int:
        all_unlabelled_indecies = self.pool.get_unlabeled_indecies()
        print(f"current number of unlabeled data: {len(all_unlabelled_indecies)}")
        all_scores = self.get_scores(all_unlabelled_indecies)
        max_scores = np.argwhere(np.isclose(all_scores, all_scores.max())).ravel()
        self.pool.set_seed()
        idx = np.random.choice(max_scores, 1)[0]
        return all_unlabelled_indecies[idx]