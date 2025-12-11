"""
Compute model output norm with null vs correct class labels for existing evaluation.

This script adds two metrics to existing evaluation results:
- null_output_norm: norm of model output with null class label (1000, used in CFG)
- cond_output_norm: norm of model output with correct class label

Comparing these helps determine if instability is related to the null class label
specifically or if it's a general issue.

Usage:
    python -m experiments.evaluate_instability.compute_null_norm \
        --results-dir ./output/evaluate_instability/251205_imagenet_val_1k_200steps
"""

from __future__ import annotations

import argparse
import json
import math
import os
from csv import DictWriter

import matplotlib.pyplot as plt
import torch
from diffusers.models import AutoencoderKL
from tqdm import tqdm

from download import find_model
from models import EqM_models

from .data import build_latent_cache

NULL_CLASS_LABEL = 1000


def _batch_norm(tensor: torch.Tensor) -> torch.Tensor:
    """Compute L2 norm for each sample in a batch."""
    return torch.linalg.vector_norm(tensor.view(tensor.shape[0], -1), dim=1)


@torch.no_grad()
def compute_output_norms(
    model,
    latents: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute norm of model output using null and correct class labels.

    Args:
        model: EqM model
        latents: Input latents (batch, C, H, W)
        labels: Correct class labels (batch,)
        device: Computation device

    Returns:
        Tuple of (null_norms, cond_norms), each of shape (batch,)
    """
    xt = latents.to(device)
    labels = labels.to(device)
    batch_size = xt.shape[0]
    null_labels = torch.full((batch_size,), NULL_CLASS_LABEL, device=device, dtype=torch.long)
    t = torch.ones(batch_size, device=device)

    # Forward pass with null label
    null_output = model.forward(xt, t, null_labels)
    null_norms = _batch_norm(null_output).cpu()

    # Forward pass with correct label
    cond_output = model.forward(xt, t, labels)
    cond_norms = _batch_norm(cond_output).cpu()

    return null_norms, cond_norms


def load_config(results_dir: str) -> dict:
    """Load the config from the results directory."""
    config_path = os.path.join(results_dir, "config.json")
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def load_existing_metrics(results_dir: str) -> list[dict]:
    """Load existing metrics from JSONL file."""
    jsonl_path = os.path.join(results_dir, "metrics.jsonl")
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


def save_updated_metrics(records: list[dict], results_dir: str) -> None:
    """Save updated metrics to CSV and JSONL files."""
    metrics_path = os.path.join(results_dir, "metrics.csv")
    jsonl_path = os.path.join(results_dir, "metrics.jsonl")

    fieldnames = list(records[0].keys())
    with open(metrics_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")


def _compute_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute Pearson correlation coefficient."""
    mean_x = x - x.mean()
    mean_y = y - y.mean()
    denom = torch.sqrt(mean_x.square().sum() * mean_y.square().sum())
    if denom.item() > 0:
        return float(mean_x.mul(mean_y).sum() / denom)
    return float("nan")


def plot_output_norm_correlations(records: list[dict], results_dir: str) -> None:
    """Plot correlation between output norms (null vs cond) and gt_long_drift."""
    required = ["gt_long_drift", "null_output_norm", "cond_output_norm"]
    if not records or any(key not in records[0] for key in required):
        print("Required metrics not found, skipping correlation plot.")
        return

    gt_values = torch.tensor([float(rec["gt_long_drift"]) for rec in records], dtype=torch.float32)
    null_norm_values = torch.tensor([float(rec["null_output_norm"]) for rec in records], dtype=torch.float32)
    cond_norm_values = torch.tensor([float(rec["cond_output_norm"]) for rec in records], dtype=torch.float32)

    # Compute correlations
    null_corr = _compute_correlation(gt_values, null_norm_values)
    cond_corr = _compute_correlation(gt_values, cond_norm_values)

    # Create side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: null class label
    ax1 = axes[0]
    ax1.scatter(gt_values.numpy(), null_norm_values.numpy(), s=12, alpha=0.6, c="tab:blue")
    ax1.set_xlabel("gt_long_drift")
    ax1.set_ylabel("null_output_norm")
    null_corr_str = "nan" if math.isnan(null_corr) else f"{null_corr:.3f}"
    ax1.set_title(f"Null Class Label (corr={null_corr_str})")

    # Right plot: correct class label
    ax2 = axes[1]
    ax2.scatter(gt_values.numpy(), cond_norm_values.numpy(), s=12, alpha=0.6, c="tab:orange")
    ax2.set_xlabel("gt_long_drift")
    ax2.set_ylabel("cond_output_norm")
    cond_corr_str = "nan" if math.isnan(cond_corr) else f"{cond_corr:.3f}"
    ax2.set_title(f"Correct Class Label (corr={cond_corr_str})")

    fig.suptitle("Model Output Norm vs Long Drift", fontsize=14)
    fig.tight_layout()
    plot_path = os.path.join(results_dir, "output_norm_correlations.png")
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved correlation plot to {plot_path}")


def load_model(config: dict, device: torch.device):
    """Load EqM model from config."""
    cli_args = config["cli_args"]
    latent_size = cli_args["image_size"] // 8
    model = EqM_models[cli_args["model"]](
        input_size=latent_size,
        num_classes=cli_args["num_classes"],
        uncond=cli_args["uncond"],
        ebm=cli_args["ebm"],
    ).to(device)
    state_dict = find_model(cli_args["ckpt"])
    if "model" in state_dict:
        model.load_state_dict(state_dict["ema"])
    else:
        model.load_state_dict(state_dict)
    model.eval()
    return model


def load_vae(config: dict, device: torch.device):
    """Load VAE from config."""
    cli_args = config["cli_args"]
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{cli_args['vae']}").to(device)
    vae.eval()
    return vae


def run(args) -> None:
    """Main entry point for computing output norms (null and conditional)."""
    results_dir = args.results_dir

    # Load config and existing metrics
    print(f"Loading config from {results_dir}")
    config = load_config(results_dir)
    cli_args = config["cli_args"]

    print("Loading existing metrics...")
    records = load_existing_metrics(results_dir)
    print(f"Loaded {len(records)} records")

    # Create index mapping for fast lookup
    index_to_record = {rec["dataset_index"]: rec for rec in records}

    # Setup device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required.")
    device = torch.device("cuda:0")
    torch.manual_seed(cli_args["seed"])
    torch.cuda.set_device(device)

    # Disable flash attention for EBM models
    if cli_args["ebm"] != "none":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    # Load model and VAE
    print("Loading model...")
    model = load_model(config, device)
    print("Loading VAE...")
    vae = load_vae(config, device)

    # Build latent cache using the same parameters
    print("Building latent cache...")
    latent_cache = build_latent_cache(
        data_path=cli_args["val_data_path"],
        vae=vae,
        device=device,
        image_size=cli_args["image_size"],
        batch_size=cli_args["batch_size"],
        num_samples=cli_args["num_samples"],
        seed=cli_args["seed"],
        num_workers=cli_args["num_workers"],
    )

    # Compute output norms for each batch
    batch_size = cli_args["batch_size"]
    total_batches = math.ceil(len(latent_cache) / batch_size)
    iterator = latent_cache.iter_batches(batch_size, device=device)

    print("Computing output norms (null and conditional)...")
    for batch in tqdm(iterator, total=total_batches, desc="Batches"):
        null_norms, cond_norms = compute_output_norms(model, batch.latents, batch.labels, device)

        for sample_idx in range(batch.latents.shape[0]):
            dataset_idx = int(batch.indices[sample_idx].item())
            if dataset_idx in index_to_record:
                index_to_record[dataset_idx]["null_output_norm"] = float(null_norms[sample_idx].item())
                index_to_record[dataset_idx]["cond_output_norm"] = float(cond_norms[sample_idx].item())

    # Save updated metrics
    print("Saving updated metrics...")
    save_updated_metrics(records, results_dir)

    # Plot correlations
    plot_output_norm_correlations(records, results_dir)

    print("Done!")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute null output norm metric for existing evaluation results")
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Path to the results directory containing config.json and metrics.jsonl",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
