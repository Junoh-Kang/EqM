"""
Command-line entry point for the EqM instability evaluation experiment.

Example:
    python -m experiments.evaluate-instability.cli \\
        --val-data-path /path/to/imagenet/val \\
        --ckpt /path/to/eqm.pt \\
        --num-samples 64 \\
        --batch-size 8 \\
        --out eval_runs/run_001
"""

from __future__ import annotations

import argparse

from models import EqM_models

from .runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EqM instability evaluation suite")

    # Dataset + IO
    parser.add_argument("--val-data-path", type=str, required=True, help="Path to ImageNet validation directory")
    parser.add_argument("--ckpt", type=str, required=True, help="Path or HF identifier for the EqM checkpoint")
    parser.add_argument("--out", type=str, required=True, help="Folder to store outputs")
    parser.add_argument("--num-samples", type=int, required=True, help="Number of validation images to evaluate")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size for metric computation")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers for latent caching")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling subsets")
    parser.add_argument(
        "--save-trajectories",
        action="store_true",
        help="Save per-batch latent trajectories used for ground-truth metrics",
    )
    parser.add_argument(
        "--trajectory-image-steps",
        type=str,
        default=None,
        help="Comma-separated trajectory steps (e.g., '0,50,100') to decode and save as PNGs",
    )

    # Model configuration
    parser.add_argument("--model", type=str, choices=list(EqM_models.keys()), default="EqM-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--uncond", type=bool, default=True)
    parser.add_argument("--ebm", type=str, choices=["none", "l2", "dot", "mean"], default="none")

    # Sampling configuration
    parser.add_argument("--sampler", type=str, default="gd", choices=["gd", "ngd"])
    parser.add_argument("--stepsize", type=float, default=0.0017)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--mu", type=float, default=0.3)

    # Metric hyperparameters
    parser.add_argument("--gt-steps", type=int, required=True, help="Steps for long-horizon drift/curvature metrics")
    parser.add_argument("--hutchinson-samples", type=int, default=1)
    parser.add_argument("--eigen-iters", type=int, default=10)
    parser.add_argument("--perturb-steps", type=int, default=10)
    parser.add_argument("--perturb-sigma", type=float, default=1e-3)
    parser.add_argument("--symmetry-probes", type=int, default=1)
    parser.add_argument("--cycle-stepsize", type=float, default=None)
    parser.add_argument("--dispersion-dirs", type=int, default=4)
    parser.add_argument("--dispersion-steps", type=int, default=5)
    parser.add_argument("--dispersion-delta", type=float, default=1e-3)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
