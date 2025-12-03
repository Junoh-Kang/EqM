"""
Collection of local instability metrics defined in PLAN.md.

Each metric returns a per-sample tensor so the runner can aggregate them into
a pandas/JSON table without extra bookkeeping.

Refactored to use torch.func (jvp, vjp) for explicit and efficient
Jacobian-vector products.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch.func import jvp, vjp

from .field import EqMField

EPS = 1e-8


def _flatten(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(tensor.shape[0], -1)


def _batch_norm(tensor: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(_flatten(tensor), dim=1)


def _normalize_batch(tensor: torch.Tensor) -> torch.Tensor:
    flat = _flatten(tensor)
    norms = torch.linalg.vector_norm(flat, dim=1, keepdim=True).clamp_min(EPS)
    normalized = flat / norms
    return normalized.view_as(tensor)


def _batch_inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (_flatten(a) * _flatten(b)).sum(dim=1)


@dataclass
class LocalMetricConfig:
    hutchinson_samples: int = 1
    eigen_iters: int = 10
    perturb_steps: int = 10
    perturb_sigma: float = 1e-3
    symmetry_probes: int = 1
    cycle_stepsize: float | None = None
    dispersion_dirs: int = 4
    dispersion_steps: int = 5
    dispersion_delta: float = 1e-3


class LocalMetricSuite:
    def __init__(self, field: EqMField, config: LocalMetricConfig) -> None:
        self.field = field
        self.config = config

    def compute(self, latents: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        """Evaluate all local-instability metrics for a batch of latents."""
        # Clean inputs
        latents = latents.detach()

        results: dict[str, torch.Tensor] = {}

        import ipdb

        ipdb.set_trace()

        results["divergence_hutchinson"] = hutchinson_divergence(
            self.field, latents, labels, self.config.hutchinson_samples
        )
        results["jacobian_lambda_max"] = largest_jacobian_eigenvalue(
            self.field, latents, labels, self.config.eigen_iters
        )
        results["one_step_amplification"] = one_step_amplification(self.field, latents, labels)
        results["empirical_perturbation"] = empirical_perturbation_instability(
            self.field, latents, labels, self.config.perturb_steps, self.config.perturb_sigma
        )
        results["symmetric_jacobian_error"] = symmetric_jacobian_error(
            self.field, latents, labels, self.config.symmetry_probes
        )
        results["cycle_consistency"] = cycle_consistency_error(self.field, latents, labels, self.config.cycle_stepsize)
        results["local_flow_dispersion"] = local_flow_dispersion(
            self.field,
            latents,
            labels,
            self.config.dispersion_dirs,
            self.config.dispersion_steps,
            self.config.dispersion_delta,
        )
        return {name: tensor.detach().to("cpu") for name, tensor in results.items()}


def _make_func(field: EqMField, labels: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """Helper to create a stateless callable for torch.func."""

    def func(x: torch.Tensor) -> torch.Tensor:
        return field.evaluate_pure(x, labels)

    return func


def hutchinson_divergence(
    field: EqMField,
    latents: torch.Tensor,
    labels: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """
    Monte Carlo estimator for the divergence ∇·f(x) = tr(J(x)).

    Using torch.func.vjp:
    div(f) = E[v^T J v].
    We compute vjp_fn(v) which gives J^T v.
    Then we dot with v: v^T (J^T v). This is mathematically equivalent to v^T J v
    because the trace of a matrix equals the trace of its transpose.
    """
    if num_samples <= 0:
        raise ValueError("hutchinson estimator requires at least one probe")

    func = _make_func(field, labels)
    xt = latents.detach().to(field.device)
    acc = torch.zeros(latents.shape[0], device=field.device)

    for _ in range(num_samples):
        v = torch.randn_like(xt)

        # Compute J^T v efficiently
        _, vjp_fn = vjp(func, xt)
        jtv = vjp_fn(v)[0]

        # Sample = v^T (J^T v)
        sample = _batch_inner(jtv, v)
        acc = acc + sample

    return acc / float(num_samples)


def largest_jacobian_eigenvalue(
    field: EqMField,
    latents: torch.Tensor,
    labels: torch.Tensor,
    num_iters: int,
) -> torch.Tensor:
    """
    Power-iteration approximation of the dominant Jacobian eigenvalue λ_max.

    Uses torch.func.jvp to compute J(x)v explicitly.
    """
    func = _make_func(field, labels)
    xt = latents.detach().to(field.device)

    # Initialize random unit vector
    v = torch.randn_like(xt)
    v = _normalize_batch(v)

    # Power iteration: v_{k+1} = J v_k / ||J v_k||
    for _ in range(max(1, num_iters)):
        _, jv = jvp(func, (xt,), (v,))
        v = _normalize_batch(jv)

    # Rayleigh quotient: (v^T J v) / (v^T v)
    # Since v is normalized, denominator is 1.
    _, final_jv = jvp(func, (xt,), (v,))
    numerator = _batch_inner(v, final_jv)

    return numerator


def one_step_amplification(field: EqMField, latents: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Closed-form amplification score for a single EqM step.
    S(x) = η * ||f(x)||_2 / ||x||_2
    """
    base = latents.to(field.device)
    direction = field.evaluate(base, labels, requires_grad=False)
    numerator = _batch_norm(direction)
    denominator = _batch_norm(base).clamp_min(EPS)
    return field.stepsize * numerator / denominator


def empirical_perturbation_instability(
    field: EqMField,
    latents: torch.Tensor,
    labels: torch.Tensor,
    num_steps: int,
    sigma: float,
) -> torch.Tensor:
    """
    Finite-horizon Lyapunov proxy.
    Δ = ||Φ^{num_steps}(x + σ ε) - Φ^{num_steps}(x)||_2
    """
    base_result = field.run_trajectory(latents, labels, num_steps=num_steps, store_history=False)
    perturbed = latents.to(field.device) + sigma * torch.randn_like(latents.to(field.device))
    pert_result = field.run_trajectory(perturbed, labels, num_steps=num_steps, store_history=False)
    return _batch_norm(pert_result.final_latents - base_result.final_latents)


def symmetric_jacobian_error(
    field: EqMField,
    latents: torch.Tensor,
    labels: torch.Tensor,
    num_probes: int,
) -> torch.Tensor:
    """
    Relative magnitude of the antisymmetric Jacobian component.

    S_sym(x) = ||J(x) v - J(x)^T v||_2 / ||J(x) v||_2.

    Uses torch.func to explicitly compute:
    1. jvp: J v
    2. vjp: J^T v
    """
    if num_probes <= 0:
        raise ValueError("symmetry error needs at least one probe")

    func = _make_func(field, labels)
    xt = latents.detach().to(field.device)
    estimates = torch.zeros(latents.shape[0], device=field.device)

    for _ in range(num_probes):
        # Random probe v
        # We need f(x) to get the shape, or just use randn_like(xt) assuming output shape == input shape
        # (EqM fields are R^d -> R^d)
        v = _normalize_batch(torch.randn_like(xt))

        # Compute J v (Forward Mode)
        _, jv = jvp(func, (xt,), (v,))

        # Compute J^T v (Reverse Mode)
        _, vjp_fn = vjp(func, xt)
        jtv = vjp_fn(v)[0]

        antisym = jv - jtv

        # Normalize by magnitude of Jv to be scale invariant
        estimates = estimates + _batch_norm(antisym) / _batch_norm(jv).clamp_min(EPS)

    return estimates / float(num_probes)


def cycle_consistency_error(
    field: EqMField,
    latents: torch.Tensor,
    labels: torch.Tensor,
    custom_stepsize: float | None = None,
) -> torch.Tensor:
    """
    Short cycle-consistency check for the explicit Euler integrator.
    Forward: y = x + η f(x)
    Backward: x' = y - η f(y)
    Error: ||x' - x||
    """
    step = custom_stepsize if custom_stepsize is not None else field.stepsize
    base = latents.to(field.device)
    # Forward
    forward = base + field.evaluate(base, labels, requires_grad=False) * step
    # Backward
    backward = forward - field.evaluate(forward, labels, requires_grad=False) * step
    return _batch_norm(backward - base)


def local_flow_dispersion(
    field: EqMField,
    latents: torch.Tensor,
    labels: torch.Tensor,
    num_dirs: int,
    num_steps: int,
    delta: float,
) -> torch.Tensor:
    """
    Log-volume estimate for a bundle of short trajectories.
    Measures expansion/contraction of a local simplex volume.
    """
    if num_dirs < 2:
        raise ValueError("dispersion requires at least two perturbation directions")

    base_traj = field.run_trajectory(latents, labels, num_steps=num_steps, store_history=False)
    base_final = base_traj.final_latents
    diffs = []

    for _ in range(num_dirs):
        noise = _normalize_batch(torch.randn_like(latents.to(field.device)))
        perturbed = latents.to(field.device) + delta * noise
        traj = field.run_trajectory(perturbed, labels, num_steps=num_steps, store_history=False)
        diffs.append(traj.final_latents - base_final)

    diff_tensor = torch.stack(diffs, dim=0)  # (num_dirs, batch, C, H, W)
    flattened = diff_tensor.permute(1, 0, 2, 3, 4).reshape(latents.shape[0], num_dirs, -1)

    # Gram matrix G = (1/d) * Δ Δ^T
    grams = torch.matmul(flattened, flattened.transpose(1, 2)) / flattened.shape[-1]

    # Log-determinant for volume
    eye = torch.eye(num_dirs, device=grams.device).unsqueeze(0) * EPS
    grams = grams + eye
    sign, logabsdet = torch.linalg.slogdet(grams)

    # Volume must be positive; clamp invalid logabsdet
    log_volume = torch.where(sign > 0, logabsdet, torch.full_like(logabsdet, float("-inf")))
    return log_volume
