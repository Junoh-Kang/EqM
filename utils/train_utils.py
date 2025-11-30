"""
Utilities for monitoring EqM training/evaluation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import torch


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

class TimestepValueLogger:
	"""
	Stores per-metric, per-timestep value lists for later aggregation/logging.
	"""

	def __init__(self) -> None:
		self.data: Dict[str, Dict[float, list]] = defaultdict(lambda: defaultdict(list))

	def reset(self) -> None:
		self.data.clear()

	@torch.no_grad()
	def __call__(self, target: torch.Tensor, pred: torch.Tensor, t_value: float) -> None:
		"""
		Compute metrics from gt/pred tensors and append their values.
		"""
		
		# error 
		diff = mean_flat((target - pred) ** 2).cpu().tolist()
		self.data["l2_error"][t_value].extend(diff)

		gt_flat = target.reshape(target.size(0), -1)
		pred_flat = pred.reshape(pred.size(0), -1)
		cos_sim = torch.sum(gt_flat * pred_flat, dim=1) / (
			gt_flat.norm(dim=1) * pred_flat.norm(dim=1) + 1e-8
		)
		self.data["cosine_sim"][t_value].extend(cos_sim.cpu().tolist())

		# norms
		pred_norm = mean_flat(pred ** 2).cpu().tolist()
		self.data["l2_pred"][t_value].extend(pred_norm)

		target_norm = mean_flat(target ** 2).cpu().tolist()
		self.data["l2_target"][t_value].extend(target_norm)
		
		norm_ratio = pred_norm / target_norm #if smaller, model underestimates noise level
		self.data["pred/target"][t_value].extend(norm_ratio)

		

	def summary(self) -> Dict[str, Dict[float, Dict[str, float]]]:
		
		output: Dict[str, Dict[float, Dict[str, float]]] = {}
		
		for metric, ts in self.data.items():
			output[metric] = {}
			for t_value, vals in ts.items():
				arr = torch.tensor(vals)
				output[metric][t_value] = {
					"mean": arr.mean().item(),
					"std": arr.std(unbiased=False).item(),
				}
		return output
