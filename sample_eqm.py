"""
A simplified single-GPU sampling script for EqM.
Uses hooks from sampling_utils.py for cleaner code.
"""

import math

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse
import os

from diffusers.models import AutoencoderKL
from PIL import Image
from tqdm import tqdm

from download import find_model
from models import EqM_models
from utils.sampling_utils import GradientNormTracker, IntermediateImageSaver, create_npz_from_sample_folder, sample_eqm


def main(args):
    """
    Generate samples using EqM model on a single GPU.
    """
    assert torch.cuda.is_available(), "Sampling currently requires at least one GPU."

    # Disable flash attention for energy-based models
    if args.ebm != "none":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    # Setup device and seed
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    torch.cuda.set_device(device)
    print(f"Using device: {device}, seed: {args.seed}")

    # Create model
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    ema_model = EqM_models[args.model](
        input_size=latent_size, num_classes=args.num_classes, uncond=args.uncond, ebm=args.ebm
    ).to(device)

    # Load checkpoint
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

    # Load VAE
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    final_step_folder = (
        f"{args.out}/{args.sampler}-{args.stepsize}-cfg{args.cfg_scale}/step_{args.num_sampling_steps:03d}"
    )
    os.makedirs(final_step_folder, exist_ok=True)

    # Calculate total samples and iterations
    total_samples = int(math.ceil(args.num_samples / args.batch_size) * args.batch_size)
    print(f"Total number of images that will be sampled: {total_samples}")
    iterations = int(total_samples // args.batch_size)

    # Create hooks
    hooks = []
    save_steps_list = []
    if args.save_steps is not None:
        save_steps_list = [int(s.strip()) for s in args.save_steps.split(",")]
        img_saver = IntermediateImageSaver(
            save_steps_list, output_folder=f"{args.out}/{args.sampler}-{args.stepsize}-cfg{args.cfg_scale}"
        )
        hooks.append(img_saver)
        print(f"Created IntermediateImageSaver hook for steps: {save_steps_list}")

    grad_tracker = None
    if args.track_grad_norm:
        grad_tracker = GradientNormTracker(args.num_sampling_steps)
        hooks.append(grad_tracker)
        print("Created GradientNormTracker hook")

    # Sampling loop
    print(f"Starting sampling with {args.sampler.upper()} sampler...")
    total_saved = 0
    for _ in tqdm(range(iterations), desc="Generating samples"):
        # Generate samples for this batch
        if args.class_labels is not None:
            class_ids = [int(c.strip()) for c in args.class_labels.split(",")]
            class_ids_tensor = torch.tensor(class_ids, device=device, dtype=torch.long)
            class_labels = class_ids_tensor[
                torch.randint(0, class_ids_tensor.numel(), (args.batch_size,), device=device)
            ]
        else:
            class_labels = None
        samples = sample_eqm(
            model=ema_model,
            vae=vae,
            device=device,
            batch_size=args.batch_size,
            latent_size=latent_size,
            class_labels=class_labels,
            num_sampling_steps=args.num_sampling_steps,
            stepsize=args.stepsize,
            cfg_scale=args.cfg_scale,
            sampler=args.sampler,
            mu=args.mu,
            hooks=hooks,
        )

        # Save final samples
        for i_sample, sample in enumerate(samples):
            index = total_saved + i_sample
            Image.fromarray(sample).save(f"{final_step_folder}/{index:06d}.png")

        total_saved += args.batch_size
    print(f"Saved {total_saved} samples to {final_step_folder}")

    # Finalize gradient norm statistics if enabled
    if grad_tracker is not None:
        print("Computing gradient norm statistics...")
        grad_tracker.finalize(args.out, args.num_sampling_steps, args.stepsize, args.sampler)

    # Create .npz files for FID evaluation
    print("Creating .npz file for final samples...")
    create_npz_from_sample_folder(final_step_folder, args.num_samples)

    # Create .npz files for intermediate steps
    if len(save_steps_list) > 0:
        print("Creating .npz files for intermediate steps...")
        for step in save_steps_list:
            step_folder = f"{args.out}/step_{step:03d}"
            if os.path.exists(step_folder):
                print(f"Creating .npz file for step {step}")
                create_npz_from_sample_folder(step_folder, args.num_samples)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-GPU EqM sampling script")
    parser.add_argument("--model", type=str, choices=list(EqM_models.keys()), default="EqM-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size for sampling")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--cfg-scale", type=float, default=4.0, help="Classifier-free guidance scale (default: 4.0)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to EqM checkpoint")
    parser.add_argument(
        "--class-labels",
        type=str,
        default=None,
        help="Class labels to sample (single or comma-separated, e.g., '207' or '207,360,388'). If not specified, samples random classes.",
    )
    parser.add_argument(
        "--stepsize", type=float, default=0.0017, help="Step size eta for gradient descent (default: 0.0017)"
    )
    parser.add_argument(
        "--num-sampling-steps", type=int, default=250, help="Number of sampling iterations (default: 250)"
    )
    parser.add_argument("--out", type=str, default="samples", help="Output directory for samples (default: 'samples')")
    parser.add_argument(
        "--sampler",
        type=str,
        default="gd",
        choices=["gd", "ngd"],
        help="Sampler type: 'gd' (gradient descent) or 'ngd' (NAG-GD) (default: 'gd')",
    )
    parser.add_argument("--mu", type=float, default=0.3, help="NAG-GD momentum hyperparameter mu (default: 0.3)")
    parser.add_argument("--num-samples", type=int, required=True, help="Total number of samples to generate")
    parser.add_argument(
        "--save-steps",
        type=str,
        default=None,
        help="Comma-separated list of sampling steps to save intermediate images (e.g., '0,50,100,249')",
    )
    parser.add_argument(
        "--track-grad-norm", action="store_true", help="Enable gradient norm tracking and visualization"
    )
    parser.add_argument("--uncond", type=bool, default=True, help="Disable/enable noise conditioning (default: True)")
    parser.add_argument(
        "--ebm",
        type=str,
        choices=["none", "l2", "dot", "mean"],
        default="none",
        help="Energy formulation (default: 'none')",
    )

    args = parser.parse_args()
    main(args)
