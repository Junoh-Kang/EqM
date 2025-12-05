"""
High-level orchestration for the instability evaluation experiment.

Usage is driven via `cli.py`, but the functions here are separated for easier
unit testing and composition.
"""

from __future__ import annotations

import json
import math
import os
from csv import DictWriter
from dataclasses import asdict
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import torch
from diffusers.models import AutoencoderKL
from PIL import Image
from tqdm import tqdm

from download import find_model
from models import EqM_models
from utils.sampling_utils import decode_latents

from .data import build_latent_cache
from .field import EqMField
from .metrics import LocalMetricConfig, LocalMetricSuite

GT_EPS = 1e-8


def _batch_l2(tensor: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(tensor.view(tensor.shape[0], -1), dim=1)


def _trajectory_curvature(history: torch.Tensor) -> torch.Tensor:
    # history shape: (num_steps + 1, batch, C, H, W)
    total_steps = history.shape[0] - 1
    if total_steps < 2:
        return torch.zeros(history.shape[1])
    curvature = torch.zeros(history.shape[1])
    for step in range(1, total_steps):
        prev_state = history[step - 1]
        curr_state = history[step]
        next_state = history[step + 1]
        numerator = _batch_l2(next_state - 2 * curr_state + prev_state)
        denominator = _batch_l2(next_state - curr_state).clamp_min(GT_EPS)
        curvature = curvature + numerator / denominator
    return curvature


def _normalize_step_list(step_arg: str | list[int] | None) -> list[int]:
    """
    Normalize user-provided trajectory step arguments into a sorted, unique list.
    Accepts comma-separated strings or iterables of ints.
    """
    if step_arg is None:
        return []
    if isinstance(step_arg, str):
        tokens = [token.strip() for token in step_arg.split(",")]
    else:
        tokens = step_arg
    normalized: set[int] = set()
    for token in tokens:
        if token is None or token == "":
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid trajectory image step '{token}'.") from exc
        if value < 0:
            raise ValueError("Trajectory image steps must be non-negative integers.")
        normalized.add(value)
    return sorted(normalized)


def _save_selected_step_images(
    history_tensor: torch.Tensor,
    steps: list[int],
    vae: AutoencoderKL,
    image_root: str,
    dataset_indices: torch.Tensor,
) -> None:
    """Decode selected trajectory steps and persist images to disk."""
    if not steps:
        return
    dataset_indices_cpu = dataset_indices.to("cpu")
    for step in steps:
        latents = history_tensor[step].to(vae.device)
        decoded_images = decode_latents(vae, latents)
        step_dir = os.path.join(image_root, f"step_{step:03d}")
        os.makedirs(step_dir, exist_ok=True)
        for image_array, dataset_index in zip(decoded_images, dataset_indices_cpu):
            file_index = int(dataset_index)
            image_path = os.path.join(step_dir, f"{file_index:06d}.png")
            Image.fromarray(image_array).save(image_path)


def _compute_ground_truth_metrics(
    field: EqMField,
    latents: torch.Tensor,
    labels: torch.Tensor,
    num_steps: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    traj = field.run_trajectory(latents, labels, num_steps=num_steps, store_history=True)
    if traj.history is None:
        raise RuntimeError("Trajectory history missing despite store_history=True.")
    history_tensor = torch.stack(traj.history, dim=0)
    drift = _batch_l2(history_tensor[-1] - history_tensor[0])
    curvature = _trajectory_curvature(history_tensor)
    return {"gt_long_drift": drift, "gt_local_curvature": curvature}, history_tensor


def _save_results(records: list[dict[str, float | int]], out_dir: str) -> None:
    if not records:
        raise ValueError("No metric records to save.")
    metrics_path = os.path.join(out_dir, "metrics.csv")
    jsonl_path = os.path.join(out_dir, "metrics.jsonl")
    summary_path = os.path.join(out_dir, "summary.json")

    fieldnames = list(records[0].keys())
    with open(metrics_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    with open(jsonl_path, "w", encoding="utf-8") as fout:
        for row in records:
            fout.write(json.dumps(row) + "\n")

    metric_columns = [c for c in fieldnames if c not in ("dataset_index", "class_label")]
    summary = {"num_samples": len(records)}
    for column in metric_columns:
        values = [rec[column] for rec in records]
        summary[column] = {
            "mean": float(mean(values)),
            "std": float(pstdev(values)) if len(values) > 1 else 0.0,
            "min": float(min(values)),
            "max": float(max(values)),
        }
    with open(summary_path, "w", encoding="utf-8") as fout:
        json.dump(summary, fout, indent=2)


def _plot_long_drift_correlations(records: list[dict[str, float | int]], out_dir: str) -> None:
    if not records:
        return
    metric_columns = [c for c in records[0].keys() if c not in ("dataset_index", "class_label")]
    if "gt_long_drift" not in metric_columns:
        print("gt_long_drift missing; skipping correlation plots.")
        return

    metrics = [name for name in metric_columns if name != "gt_long_drift"]
    if not metrics:
        return

    gt_values = torch.tensor([float(rec["gt_long_drift"]) for rec in records], dtype=torch.float32)
    cols = min(3, len(metrics))
    rows = math.ceil(len(metrics) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), squeeze=False)
    flat_axes = axes.flatten()

    gt_np = gt_values.numpy()
    mean_centered_gt = gt_values - gt_values.mean()

    for idx, metric_name in enumerate(metrics):
        metric_values = torch.tensor([float(rec[metric_name]) for rec in records], dtype=torch.float32)
        centered_metric = metric_values - metric_values.mean()
        denom = torch.sqrt(centered_metric.square().sum() * mean_centered_gt.square().sum())
        if denom.item() > 0:
            corr = float(centered_metric.mul(mean_centered_gt).sum() / denom)
        else:
            corr = float("nan")

        ax = flat_axes[idx]
        ax.scatter(gt_np, metric_values.numpy(), s=8, alpha=0.6)
        ax.set_xlabel("gt_long_drift")
        ax.set_ylabel(metric_name)
        corr_str = "nan" if math.isnan(corr) else f"{corr:.2f}"
        ax.set_title(f"corr={corr_str}")

    for idx in range(len(metrics), len(flat_axes)):
        fig.delaxes(flat_axes[idx])

    fig.tight_layout()
    plot_path = os.path.join(out_dir, "long_drift_correlations.png")
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved correlation plot to {plot_path}")


def _load_eqm_model(args, device: torch.device):
    latent_size = args.image_size // 8
    model = EqM_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        uncond=args.uncond,
        ebm=args.ebm,
    ).to(device)
    state_dict = find_model(args.ckpt)
    if "model" in state_dict:
        model.load_state_dict(state_dict["ema"])
    else:
        model.load_state_dict(state_dict)
    model.eval()
    return model


def _load_vae(args, device: torch.device):
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    return vae


def _metric_config_from_args(args) -> LocalMetricConfig:
    return LocalMetricConfig(
        hutchinson_samples=args.hutchinson_samples,
        eigen_iters=args.eigen_iters,
        perturb_steps=args.perturb_steps,
        perturb_sigma=args.perturb_sigma,
        symmetry_probes=args.symmetry_probes,
        cycle_stepsize=args.cycle_stepsize,
        dispersion_dirs=args.dispersion_dirs,
        dispersion_steps=args.dispersion_steps,
        dispersion_delta=args.dispersion_delta,
    )


def _save_run_configuration(args, metric_config: LocalMetricConfig, out_dir: str) -> None:
    config_path = os.path.join(out_dir, "config.json")
    payload = {
        "cli_args": dict(vars(args)),
        "metric_config": asdict(metric_config),
    }
    with open(config_path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2)


def _determine_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for instability evaluation.")
    return torch.device("cuda:0")


def run_experiment(args) -> None:
    os.makedirs(args.out, exist_ok=True)
    metric_config = _metric_config_from_args(args)
    _save_run_configuration(args, metric_config, args.out)
    device = _determine_device()
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(f"Running instability evaluation on device {device}")

    if args.ebm != "none" and device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    model = _load_eqm_model(args, device)
    vae = _load_vae(args, device)
    latent_cache = build_latent_cache(
        data_path=args.val_data_path,
        vae=vae,
        device=device,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    field = EqMField(
        model=model,
        device=device,
        cfg_scale=args.cfg_scale,
        stepsize=args.stepsize,
        sampler=args.sampler,
        mu=args.mu,
    )
    metric_suite = LocalMetricSuite(field, metric_config)

    total_batches = math.ceil(len(latent_cache) / args.batch_size)
    iterator = latent_cache.iter_batches(args.batch_size, device=device)

    raw_image_steps = getattr(args, "trajectory_image_steps", None)
    trajectory_image_steps = _normalize_step_list(raw_image_steps)
    if trajectory_image_steps:
        max_history_step = args.gt_steps
        invalid_steps = [step for step in trajectory_image_steps if step > max_history_step]
        if invalid_steps:
            print(f"Skipping trajectory image steps beyond gt-steps ({args.gt_steps}): {invalid_steps}")
            trajectory_image_steps = [step for step in trajectory_image_steps if step <= max_history_step]
    image_root = None
    if trajectory_image_steps:
        image_root = os.path.join(args.out, "trajectory_images")
        os.makedirs(image_root, exist_ok=True)

    traj_dir = None
    if args.save_trajectories:
        traj_dir = os.path.join(args.out, "trajectories")
        os.makedirs(traj_dir, exist_ok=True)

    records: list[dict[str, float | int]] = []
    for batch_idx, batch in enumerate(tqdm(iterator, total=total_batches, desc="Batches")):
        print(f"Batch {batch_idx} of {total_batches}")
        local_metrics = metric_suite.compute(batch.latents, batch.labels)
        gt_metrics, history_tensor = _compute_ground_truth_metrics(field, batch.latents, batch.labels, args.gt_steps)

        batch_size = batch.latents.shape[0]
        for sample_idx in range(batch_size):
            record = {
                "dataset_index": int(batch.indices[sample_idx].item()),
                "class_label": int(batch.labels[sample_idx].to("cpu").item()),
            }
            for name, tensor in local_metrics.items():
                record[name] = float(tensor[sample_idx].item())
            for name, tensor in gt_metrics.items():
                record[name] = float(tensor[sample_idx].item())
            records.append(record)

        if traj_dir is not None:
            torch.save(
                {
                    "indices": batch.indices.clone(),
                    "history": history_tensor.clone(),
                },
                os.path.join(traj_dir, f"batch_{batch_idx:04d}.pt"),
            )
        if trajectory_image_steps:
            _save_selected_step_images(
                history_tensor=history_tensor,
                steps=trajectory_image_steps,
                vae=vae,
                image_root=image_root,
                dataset_indices=batch.indices,
            )

    _save_results(records, args.out)
    _plot_long_drift_correlations(records, args.out)
    print(f"Saved metrics to {args.out}")
