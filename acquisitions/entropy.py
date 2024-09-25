import numpy as np
import torch
# The Acquisition function
class Entropy():
    def __init__(self, pool: 'core.Pool', clf: 'core.Classifier') -> None:
        self.pool = pool
        self.clf = clf
    
    def get_scores(self, all_unlabelled_indecies, best_model: torch.nn.Module) -> np.ndarray:
        values = self.pool.dataset.x[all_unlabelled_indecies]
        with torch.no_grad():
            probs = best_model(torch.Tensor(values).to(self.clf.device))
        log_probs = torch.log(probs + torch.finfo(torch.float32).smallest_normal)
        U = -(probs*log_probs).sum(axis=1)
        return U

    def query(self, best_model: torch.nn.Module) -> int:
        all_unlabelled_indecies = self.pool.get_unlabeled_indecies()
        all_scores = self.get_scores(all_unlabelled_indecies, best_model)
        max_scores = np.argwhere(np.isclose(all_scores, all_scores.max())).ravel()
        idx = np.random.choice(max_scores, 1)[0]
        return all_unlabelled_indecies[idx]