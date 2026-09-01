"""Callbacks for CEM and iCEM solvers."""

from typing import Any

import torch

from .common import Callback


class EliteCostRecorder(Callback):
    """Per-step elite cost stats (mean, min, max), per env."""

    def compute(self, **state: Any) -> dict[str, float | list[float]]:
        v: torch.Tensor = state['topk_vals'].detach()
        return {
            'mean': self._reduce(v.mean(dim=1)),
            'min': self._reduce(v.min(dim=1).values),
            'max': self._reduce(v.max(dim=1).values),
        }


class VarNormRecorder(Callback):
    """Per-step mean variance of the action distribution (per env)."""

    def compute(self, **state: Any) -> float | list[float]:
        var: torch.Tensor = state['var']
        per_env = var.detach().flatten(1).mean(dim=-1)
        return self._reduce(per_env)


class MeanShiftRecorder(Callback):
    """Per-step L2 distance between consecutive distribution means (per env)."""

    def compute(self, **state: Any) -> float | list[float] | None:
        prev_mean: torch.Tensor | None = state.get('prev_mean')
        if prev_mean is None:
            return None
        mean: torch.Tensor = state['mean']
        per_env = (mean - prev_mean).detach().flatten(1).norm(dim=-1)
        return self._reduce(per_env)


class EliteSpreadRecorder(Callback):
    """Per-step within-elite std (diversity of the top-k elites, per env)."""

    def compute(self, **state: Any) -> float | list[float]:
        topk_candidates: torch.Tensor = state['topk_candidates']
        per_env = topk_candidates.detach().std(dim=1).flatten(1).mean(dim=-1)
        return self._reduce(per_env)


class PlanRecorder(Callback):
    """Capture the per-iteration plan state so a renderer can replay CEM.

    The scalar recorders above collapse each iteration to a number. This one
    keeps tensors instead: the distribution mean (the plan the solver would
    return if it stopped at this iteration) and a subsample of the elites, so
    a video can show the candidate cloud collapsing onto the final plan.

    Everything is detached to CPU float32 with the batch dim intact, so
    ``history`` is ``list[batch][iteration]`` of dicts whose values carry a
    leading batch dimension. Consumers map a batch back to env indices by
    walking the batches in order (the solver iterates them as consecutive
    slices of the envs it was asked to solve).

    Args:
        max_elites: Number of elites to keep per iteration (the top-k are
            already cost-sorted, so this keeps the best ones).
        stride: Keep every ``stride``-th iteration. Raise it to cut the
            memory and rendering cost of long optimizations.
    """

    name = 'plan'

    def __init__(self, max_elites: int = 32, stride: int = 1) -> None:
        super().__init__(reduction='none')
        self.max_elites = max_elites
        self.stride = max(1, int(stride))

    def compute(self, **state: Any) -> dict[str, Any] | None:
        step: int = state['step']
        if step % self.stride:
            return None

        def cpu(x: torch.Tensor) -> torch.Tensor:
            return x.detach().float().cpu()

        topk_vals: torch.Tensor = state['topk_vals']
        costs: torch.Tensor = state['costs']
        return {
            'step': step,
            'elites': cpu(state['topk_candidates'][:, : self.max_elites]),
            'mean': cpu(state['mean']),
            'var': cpu(state['var']),
            'elite_cost_mean': cpu(topk_vals.mean(dim=1)),
            'elite_cost_min': cpu(topk_vals.min(dim=1).values),
            'pop_cost_mean': cpu(costs.mean(dim=1)),
        }
