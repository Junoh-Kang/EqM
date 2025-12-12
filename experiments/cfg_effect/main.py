"""
Experiment: CFG Effect on Image Collapse

Investigates how classifier-free guidance (CFG) affects image collapse during
extended EqM sampling. The hypothesis is that collapse occurs when combining
model outputs from original class labels and null class labels (CFG > 1.0).

Experiment Design:
- Baseline: Full sampling with cfg_scale for all steps
- Switch experiments: For each switch_step in switch_steps list, use cfg_scale
  for first switch_step steps, then switch to null class label only (cfg=1.0)
  for remaining steps.

Implementation:
- Run sample_eqm() with cfg_scale, capture intermediate latents at switch steps
- For each captured latent, resume sample_eqm() with cfg=1.0 for remaining steps
"""

import argparse
import json
import math
import os

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from diffusers.models import AutoencoderKL
from tqdm import tqdm

from download import find_model
from models import EqM_models
from utils.sampling_hooks import (
    GradientNormTracker,
    IntermediateImageSaver,
    SamplingHookContext,
)
from utils.sampling_utils import create_npz_from_sample_folder, sample_eqm


class LatentCaptureHook:
    """Hook to capture intermediate latents at specified steps."""

    def __init__(self, capture_steps):
        self.capture_steps = set(capture_steps)
        self.captured_latents = {}  # step -> latent tensor

    def __call__(self, context: SamplingHookContext):
        if context.step_idx not in self.capture_steps:
            return

        # Extract conditional part if using CFG
        xt_save = context.xt
        if context.use_cfg:
            batch_size = context.xt.shape[0] // 2
            xt_save = context.xt[:batch_size]

        self.captured_latents[context.step_idx] = xt_save.clone()


def run_baseline_and_capture(
    model: torch.nn.Module,
    vae: AutoencoderKL,
    device: torch.device,
    initial_latent: torch.Tensor,
    class_labels: torch.Tensor,
    latent_size: int,
    num_sampling_steps: int,
    stepsize: float,
    cfg_scale: float,
    sampler: str,
    mu: float,
    save_steps_list: list[int],
    switch_steps: list[int],
    output_folder: str,
    batch_size: int,
    num_samples: int,
    track_grad_norm: bool = False,
):
    """
    Run baseline with cfg_scale for all steps and capture intermediate latents.

    Returns:
        captured_latents: dict mapping switch_step -> list of latent tensors (per batch)
    """
    os.makedirs(output_folder, exist_ok=True)

    total_samples = int(math.ceil(num_samples / batch_size) * batch_size)
    iterations = int(total_samples // batch_size)

    # Store captured latents per switch step
    all_captured = {step: [] for step in switch_steps}

    # Create hooks
    hooks = []

    capture_hook = LatentCaptureHook(switch_steps)
    hooks.append(capture_hook)

    if num_sampling_steps not in save_steps_list:
        save_steps_list.append(num_sampling_steps)
    img_saver = IntermediateImageSaver(save_steps_list, output_folder=output_folder)
    hooks.append(img_saver)

    grad_tracker = None
    if track_grad_norm:
        grad_tracker = GradientNormTracker(num_sampling_steps)
        hooks.append(grad_tracker)

    # Sampling loop
    total_saved = 0
    for batch_idx in tqdm(range(iterations), desc="Baseline CFG"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_samples)
        actual_batch_size = batch_end - batch_start

        batch_latent = initial_latent[batch_start:batch_end]
        batch_labels = class_labels[batch_start:batch_end]

        sample_eqm(
            model=model,
            vae=vae,
            device=device,
            batch_size=actual_batch_size,
            latent_size=latent_size,
            initial_latent=batch_latent,
            class_labels=batch_labels,
            num_sampling_steps=num_sampling_steps,
            stepsize=stepsize,
            cfg_scale=cfg_scale,
            sampler=sampler,
            mu=mu,
            hooks=hooks,
        )

        # Store captured latents
        for step in switch_steps:
            if step in capture_hook.captured_latents:
                all_captured[step].append(capture_hook.captured_latents[step].cpu())

        total_saved += actual_batch_size

    if save_steps_list:
        for step in save_steps_list:
            step_folder = f"{output_folder}/step_{step:03d}"
            if os.path.exists(step_folder):
                create_npz_from_sample_folder(step_folder, num_samples)

    # Finalize gradient norm tracking
    if grad_tracker is not None:
        # Create args-like object for finalize
        grad_tracker.finalize(output_folder, num_sampling_steps, stepsize, sampler)

    return all_captured, grad_tracker


def run_switch_experiment(
    model: torch.nn.Module,
    vae: AutoencoderKL,
    device: torch.device,
    captured_latents: list[torch.Tensor],
    latent_size: int,
    num_sampling_steps: int,
    switch_step: int,
    stepsize: float,
    sampler: str,
    mu: float,
    save_steps_list: list[int],
    output_folder: str,
    track_grad_norm: bool = False,
    baseline_grad_tracker: GradientNormTracker | None = None,
):
    """
    Run switch experiment: start from captured latent at switch_step,
    continue with null class label (cfg=1.0) for remaining steps.
    """
    os.makedirs(output_folder, exist_ok=True)
    remaining_steps = num_sampling_steps - switch_step

    # Create hooks
    hooks = []

    adjusted_save_steps = [s - switch_step for s in (save_steps_list or []) if s > switch_step]
    if remaining_steps not in adjusted_save_steps:  # Always save the final step
        adjusted_save_steps.append(remaining_steps)
    img_saver = IntermediateImageSaver(
        adjusted_save_steps,
        folder_pattern=lambda ctx: f"{output_folder}/step_{ctx.step_idx + switch_step:03d}",
    )
    hooks.append(img_saver)

    grad_tracker = None
    if track_grad_norm:
        grad_tracker = GradientNormTracker(remaining_steps)
        hooks.append(grad_tracker)

    # Sampling loop
    total_saved = 0
    for _, batch_latent in enumerate(tqdm(captured_latents, desc=f"Switch {switch_step}")):
        batch_latent = batch_latent.to(device)
        actual_batch_size = batch_latent.shape[0]

        # Use null class labels (1000)
        null_labels = torch.tensor([1000] * actual_batch_size, device=device)

        sample_eqm(
            model=model,
            vae=vae,
            device=device,
            batch_size=actual_batch_size,
            latent_size=latent_size,
            initial_latent=batch_latent,
            class_labels=null_labels,
            num_sampling_steps=remaining_steps,
            stepsize=stepsize,
            cfg_scale=1.0,  # No guidance for remaining steps
            sampler=sampler,
            mu=mu,
            hooks=hooks,
        )

        total_saved += actual_batch_size

    # Create .npz files for intermediate steps
    for step in adjusted_save_steps:
        step_folder = f"{output_folder}/step_{step + switch_step:03d}"
        if os.path.exists(step_folder):
            create_npz_from_sample_folder(step_folder, total_saved)

    # Finalize gradient norm tracking
    if grad_tracker is not None:
        # Prepend baseline gradient norms to get full trajectory
        if baseline_grad_tracker is not None:
            baseline_norms = baseline_grad_tracker.gradient_norms[:switch_step]
            grad_tracker.gradient_norms = baseline_norms + grad_tracker.gradient_norms

        grad_tracker.finalize(output_folder, num_sampling_steps, stepsize, sampler)


def main(args):
    """Main function to run CFG effect experiments."""
    assert torch.cuda.is_available(), "Sampling requires at least one GPU."

    if args.ebm != "none":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    torch.cuda.set_device(device)
    print(f"Using device: {device}, seed: {args.seed}")

    assert args.image_size % 8 == 0, "Image size must be divisible by 8"
    latent_size = args.image_size // 8
    ema_model = EqM_models[args.model](
        input_size=latent_size, num_classes=args.num_classes, uncond=args.uncond, ebm=args.ebm
    ).to(device)

    if args.ckpt is None:
        raise ValueError("Checkpoint is required")
    print(f"Loading checkpoint from {args.ckpt}")
    state_dict = find_model(args.ckpt)
    if "model" in state_dict.keys():
        ema_model.load_state_dict(state_dict["ema"])
    else:
        ema_model.load_state_dict(state_dict)
    ema_model.eval()
    print(f"EqM Parameters: {sum(p.numel() for p in ema_model.parameters()):,}")

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    os.makedirs(args.out, exist_ok=True)

    switch_steps = [int(s.strip()) for s in args.switch_steps.split(",")]
    print(f"Switch steps: {switch_steps}")

    save_steps_list = []
    if args.save_steps is not None:
        save_steps_list = [int(s.strip()) for s in args.save_steps.split(",")]
    print(f"Save steps: {save_steps_list}")

    # Generate shared initial latent and class labels
    print("Generating shared initial latent and class labels...")
    total_samples = int(math.ceil(args.num_samples / args.batch_size) * args.batch_size)
    initial_latent = torch.randn(total_samples, 4, latent_size, latent_size, device=device)

    if args.class_labels is not None:
        class_ids = [int(c.strip()) for c in args.class_labels.split(",")]
        class_ids_tensor = torch.tensor(class_ids, device=device, dtype=torch.long)
        class_labels = class_ids_tensor[torch.randint(0, class_ids_tensor.numel(), (total_samples,), device=device)]
    else:
        class_labels = torch.randint(0, 1000, (total_samples,), device=device)

    # Save initial latent
    latent_path = f"{args.out}/initial_latent.npz"
    np.savez(
        latent_path,
        latent=initial_latent.cpu().numpy(),
        class_labels=class_labels.cpu().numpy(),
    )
    print(f"Saved initial latent and class labels to {latent_path}")

    # Save metadata
    metadata = {
        "seed": args.seed,
        "model": args.model,
        "image_size": args.image_size,
        "num_classes": args.num_classes,
        "batch_size": args.batch_size,
        "num_samples": args.num_samples,
        "num_sampling_steps": args.num_sampling_steps,
        "stepsize": args.stepsize,
        "cfg_scale": args.cfg_scale,
        "switch_steps": switch_steps,
        "save_steps": save_steps_list,
        "sampler": args.sampler,
        "mu": args.mu,
        "vae": args.vae,
        "ckpt": args.ckpt,
        "class_labels_arg": args.class_labels,
        "uncond": args.uncond,
        "ebm": args.ebm,
        "track_grad_norm": args.track_grad_norm,
    }
    with open(f"{args.out}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {args.out}/metadata.json")

    # Run baseline with full CFG and capture intermediate latents
    print(f"\n{'=' * 60}")
    print(f"Running baseline with cfg_scale={args.cfg_scale} (capturing latents at switch steps)")
    print(f"{'=' * 60}")
    captured_latents, baseline_grad_tracker = run_baseline_and_capture(
        model=ema_model,
        vae=vae,
        device=device,
        initial_latent=initial_latent,
        class_labels=class_labels,
        latent_size=latent_size,
        num_sampling_steps=args.num_sampling_steps,
        stepsize=args.stepsize,
        cfg_scale=args.cfg_scale,
        sampler=args.sampler,
        mu=args.mu,
        save_steps_list=save_steps_list,
        switch_steps=switch_steps,
        output_folder=f"{args.out}/baseline_cfg{args.cfg_scale}",
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        track_grad_norm=args.track_grad_norm,
    )

    # Run switch experiments using captured latents
    for switch_step in switch_steps:
        print(f"\n{'=' * 60}")
        print(f"Running switch experiment: cfg={args.cfg_scale} for steps 1-{switch_step}, then null-only")
        print(f"{'=' * 60}")
        run_switch_experiment(
            model=ema_model,
            vae=vae,
            device=device,
            captured_latents=captured_latents[switch_step],
            latent_size=latent_size,
            num_sampling_steps=args.num_sampling_steps,
            switch_step=switch_step,
            stepsize=args.stepsize,
            sampler=args.sampler,
            mu=args.mu,
            save_steps_list=save_steps_list,
            output_folder=f"{args.out}/switch_{switch_step:03d}",
            track_grad_norm=args.track_grad_norm,
            baseline_grad_tracker=baseline_grad_tracker,
        )

    print("\nAll experiments completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CFG Effect Experiment")
    parser.add_argument("--model", type=str, choices=list(EqM_models.keys()), default="EqM-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size for sampling")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--cfg-scale", type=float, default=4.0, help="CFG scale for initial steps")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to EqM checkpoint")
    parser.add_argument(
        "--class-labels",
        type=str,
        default=None,
        help="Class labels to sample (comma-separated). If not specified, samples random classes.",
    )
    parser.add_argument("--stepsize", type=float, default=0.0017, help="Step size eta")
    parser.add_argument("--num-sampling-steps", type=int, default=1000, help="Total sampling steps")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--sampler",
        type=str,
        default="gd",
        choices=["gd", "ngd"],
        help="Sampler type: 'gd' or 'ngd'",
    )
    parser.add_argument("--mu", type=float, default=0.3, help="NAG-GD momentum hyperparameter")
    parser.add_argument("--num-samples", type=int, required=True, help="Total samples to generate")
    parser.add_argument(
        "--save-steps",
        type=str,
        default=None,
        help="Comma-separated list of steps to save intermediate images",
    )
    parser.add_argument(
        "--switch-steps",
        type=str,
        required=True,
        help="Comma-separated list of steps at which to switch to null-only",
    )
    parser.add_argument(
        "--track-grad-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable gradient norm tracking and visualization",
    )
    parser.add_argument("--uncond", type=bool, default=True, help="Enable noise conditioning")
    parser.add_argument(
        "--ebm",
        type=str,
        choices=["none", "l2", "dot", "mean"],
        default="none",
        help="Energy formulation",
    )

    args = parser.parse_args()
    main(args)
