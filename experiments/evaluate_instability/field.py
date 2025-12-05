"""
Thin wrapper around the EqM vector field.

The sampler in `sample_eqm.py` only uses `torch.no_grad()` and hides the exact
classifier-free guidance bookkeeping.  For instability metrics we need the
same logic with gradients enabled plus utilities for running short trajectories
that share a consistent interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TrajectoryResult:
    final_latents: torch.Tensor
    history: list[torch.Tensor] | None = None


class EqMField:
    def __init__(
        self,
        model,
        device: torch.device,
        cfg_scale: float,
        stepsize: float,
        sampler: str = "gd",
        mu: float = 0.3,
    ) -> None:
        self.model = model
        self.device = device
        self.cfg_scale = cfg_scale
        self.stepsize = stepsize
        self.sampler = sampler
        self.mu = mu
        self.use_cfg = cfg_scale > 1.0

    def _prepare_inputs(
        self,
        xt: torch.Tensor,
        labels: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_cfg:
            xt = torch.cat([xt, xt], dim=0)
            null_labels = torch.full((labels.shape[0],), 1000, device=self.device, dtype=labels.dtype)
            labels = torch.cat([labels, null_labels], dim=0)
            t = torch.cat([t, t], dim=0)
        return xt, labels, t

    def _forward_impl(
        self,
        xt: torch.Tensor,
        labels: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        xt_in, labels_in, t_in = self._prepare_inputs(xt, labels, t)
        if self.use_cfg:
            out = self.model.forward_with_cfg(xt_in, t_in, labels_in, self.cfg_scale)
            out, _ = out.chunk(2, dim=0)
        else:
            out = self.model.forward(xt_in, t_in, labels_in)
        return out

    def evaluate(
        self,
        xt: torch.Tensor,
        labels: torch.Tensor,
        t: torch.Tensor | None = None,
        requires_grad: bool = True,
    ) -> torch.Tensor:
        if t is None:
            t = torch.ones(xt.shape[0], device=self.device)
        else:
            t = t.to(self.device)
        xt = xt.to(self.device)
        labels = labels.to(self.device)
        xt = xt.clone().detach()
        if requires_grad:
            xt = xt.requires_grad_()
        return self._forward_impl(xt, labels, t)

    def evaluate_pure(
        self,
        xt: torch.Tensor,
        labels: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Evaluate the field without cloning, detaching, or toggling requires_grad.
        Used by torch.func transforms that expect a stateless callable and
        manage autograd bookkeeping on their own.
        """
        if t is None:
            t = torch.ones(xt.shape[0], device=self.device)
        else:
            t = t.to(self.device)
        xt = xt.to(self.device)
        labels = labels.to(self.device)
        return self._forward_impl(xt, labels, t)

    @torch.no_grad()
    def run_trajectory(
        self,
        latents: torch.Tensor,
        labels: torch.Tensor,
        num_steps: int,
        stepsize: float | None = None,
        store_history: bool = False,
    ) -> TrajectoryResult:
        step = stepsize if stepsize is not None else self.stepsize
        xt = latents.to(self.device)
        y = labels.to(self.device)
        t = torch.ones((xt.shape[0],), device=self.device)
        momentum = torch.zeros_like(xt)
        history: list[torch.Tensor] | None = [] if store_history else None
        if history is not None:
            history.append(xt.detach().to("cpu"))

        for _ in range(num_steps):
            if self.sampler == "gd":
                direction = self.evaluate(xt, y, t, requires_grad=False)
            elif self.sampler == "ngd":
                lookahead = xt + step * momentum * self.mu
                direction = self.evaluate(lookahead, y, t, requires_grad=False)
                momentum = direction
            else:
                raise ValueError(f"Unsupported sampler: {self.sampler}")
            xt = xt + direction * step
            t = t + step
            if history is not None:
                history.append(xt.detach().to("cpu"))

        return TrajectoryResult(final_latents=xt, history=history)
