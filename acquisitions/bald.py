import numpy as np
import torch
# The Acquisition function
class BALD():
    def __init__(self, pool: 'core.Pool', clf: 'core.Classifier') -> None:
        self.pool = pool
        self.clf = clf
        self.forward_passes = 100
    
    def get_scores(self, all_unlabelled_indecies, best_model: torch.nn.Module) -> np.ndarray:
        values = self.pool.dataset.x[all_unlabelled_indecies]
        no_classes = int(self.pool.dataset_config['n_classes'])
        total_predictions = torch.empty((0, len(values), no_classes)).to(self.clf.device)
        best_model.train()
        for _ in range(self.forward_passes):
            with torch.no_grad():
                probs = best_model(torch.Tensor(values))
            total_predictions = torch.cat((total_predictions, torch.unsqueeze(probs, 0)))
        average_prob =  total_predictions.mean(dim=0)
        total_uncertainty = -(average_prob*torch.log(average_prob + torch.finfo(torch.float32).smallest_normal)).sum(dim=-1)
        data_uncertainty = (-(total_predictions*torch.log(total_predictions + torch.finfo(torch.float32).smallest_normal))).sum(dim=-1).mean(dim=0)
        knowledge_uncertainty = total_uncertainty - data_uncertainty
        return knowledge_uncertainty.cpu()

    def query(self, best_model: torch.nn.Module) -> int:
        all_unlabelled_indecies = self.pool.get_unlabeled_indecies()
        all_scores = self.get_scores(all_unlabelled_indecies, best_model)
        max_scores = np.argwhere(np.isclose(all_scores, all_scores.max())).ravel()
        idx = np.random.choice(max_scores, 1)[0]
        return all_unlabelled_indecies[idx]