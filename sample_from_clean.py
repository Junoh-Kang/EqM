"""
Experiment: Clean Image Stability Analysis for EqM.

This script analyzes which ImageNet validation images become distorted during EqM sampling.
Instead of starting from noise, it loads clean validation images and runs EqM sampling
from their latent representations to identify unstable images.

Key features:
- Loads ImageNet validation images as starting points
- Tracks L2 distortion from original images at each step
- Analyzes correlation between high-frequency content and distortion
- Saves top-N most/least distorted images at specified steps
"""

import math

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse
import os

import torchvision.transforms as transforms
from diffusers.models import AutoencoderKL
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from download import find_model
from models import EqM_models
from sampling_utils import (
    DistortionTracker,
    GradientNormTracker,
    IntermediateImageSaver,
    compute_high_frequency_content,
    create_npz_from_sample_folder,
    encode_images_to_latent,
    sample_eqm,
)


def main(args):
    """
    Analyze distortion of clean ImageNet validation images during EqM sampling.
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

    # Prepare validation image transforms
    transform = transforms.Compose(
        [
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )

    # Load validation dataset
    print(f"Loading ImageNet validation images from {args.val_data_path}...")
    val_dataset = ImageFolder(args.val_data_path, transform=transform)

    # Randomly sample num_samples images
    import random
    random.seed(args.seed)
    all_indices = list(range(len(val_dataset)))
    random.shuffle(all_indices)
    indices = all_indices[:args.num_samples]
    indices.sort()  # Sort for reproducibility
    subset_dataset = Subset(val_dataset, indices)
    print(f"Selected {len(subset_dataset)} validation images (random sample)")

    # Create dataloader
    val_loader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Create model
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    ema_model = EqM_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        uncond=args.uncond,
        ebm=args.ebm,
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

    # Create folder for original images
    original_folder = f"{args.out}/step_000_original"
    os.makedirs(original_folder, exist_ok=True)

    final_step_folder = f"{args.out}/step_{args.num_sampling_steps:03d}"
    os.makedirs(final_step_folder, exist_ok=True)

    # Load all validation images and encode to latent space
    print("Loading and encoding validation images...")
    all_images = []
    all_latents = []
    all_class_labels = []

    for images, labels in tqdm(val_loader, desc="Loading images"):
        all_images.append(images)
        all_class_labels.extend(labels.tolist())

        # Encode to latent space
        latents = encode_images_to_latent(vae, images, device)
        all_latents.append(latents)

    # Concatenate all batches
    all_images = torch.cat(all_images, dim=0)
    all_latents = torch.cat(all_latents, dim=0)
    all_class_labels = torch.tensor(all_class_labels, dtype=torch.long)

    print(f"Encoded {all_latents.shape[0]} images to latent space")
    print(f"Latent shape: {all_latents.shape}")

    # Save original clean images for comparison
    print("Saving original clean images...")
    from sampling_utils import decode_latents

    original_images = decode_latents(vae, all_latents.to(device))
    for idx, img in enumerate(original_images):
        Image.fromarray(img).save(f"{original_folder}/{idx:06d}.png")
    print(f"Saved {len(original_images)} original images to {original_folder}")

    # Compute high-frequency content for all images
    print("Computing high-frequency content metrics...")
    # Convert images from [-1, 1] to [0, 255] for frequency analysis
    images_uint8 = ((all_images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    high_freq_metrics = compute_high_frequency_content(images_uint8)
    print(
        f"Computed high-frequency metrics: min={high_freq_metrics.min():.4f}, max={high_freq_metrics.max():.4f}, mean={high_freq_metrics.mean():.4f}"
    )

    # Calculate total samples and iterations
    total_samples = all_latents.shape[0]
    print(f"Total number of images that will be sampled: {total_samples}")
    iterations = int(math.ceil(total_samples / args.batch_size))

    # Parse save_steps
    if args.save_steps is not None:
        save_steps_list = [int(s.strip()) for s in args.save_steps.split(",")]
    else:
        save_steps_list = []

    # Create hooks
    hooks = []

    if len(save_steps_list) > 0:
        img_saver = IntermediateImageSaver(save_steps_list, args.out)
        hooks.append(img_saver)
        print(f"Created IntermediateImageSaver hook for steps: {save_steps_list}")

    grad_tracker = None
    if args.track_grad_norm:
        grad_tracker = GradientNormTracker(args.num_sampling_steps)
        hooks.append(grad_tracker)
        print(f"Created GradientNormTracker hook")

    # Create distortion tracker - always include final step
    distortion_steps = save_steps_list.copy() if len(save_steps_list) > 0 else []
    final_step = args.num_sampling_steps
    if final_step not in distortion_steps:
        distortion_steps.append(final_step)

    distortion_tracker = DistortionTracker(
        original_latents=all_latents,
        high_freq_metrics=high_freq_metrics,
        save_steps=distortion_steps,
        output_folder=args.out,
        top_n=args.top_n,
    )
    hooks.append(distortion_tracker)
    print(f"Created DistortionTracker hook for steps: {distortion_steps} (top_n={args.top_n})")

    # Sampling loop
    print(f"Starting sampling from clean images with {args.sampler.upper()} sampler...")
    total_saved = 0

    for batch_idx in tqdm(range(iterations), desc="Generating samples"):
        # Get batch of latents and class labels
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, total_samples)
        actual_batch_size = batch_end - batch_start

        batch_latents = all_latents[batch_start:batch_end]
        batch_labels = all_class_labels[batch_start:batch_end]

        # Generate samples for this batch, starting from clean latents
        samples = sample_eqm(
            model=ema_model,
            vae=vae,
            device=device,
            batch_size=actual_batch_size,
            latent_size=latent_size,
            initial_latent=batch_latents,  # Start from clean image latents
            class_labels=batch_labels,  # Use actual image class labels
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

        total_saved += actual_batch_size

        # Notify distortion tracker that batch is complete
        distortion_tracker.on_batch_complete(actual_batch_size)

    print(f"Saved {total_saved} samples to {final_step_folder}")

    # Finalize gradient norm statistics if enabled
    if grad_tracker is not None:
        print("Computing gradient norm statistics...")
        grad_tracker.finalize(args, args.out)

    # Finalize distortion analysis
    print("Finalizing distortion analysis...")
    distortion_tracker.finalize(vae, args.out)

    # Create .npz files for FID evaluation
    print("Creating .npz file for final samples...")
    create_npz_from_sample_folder(final_step_folder, total_samples)

    # Create .npz files for intermediate steps
    if len(save_steps_list) > 0:
        print("Creating .npz files for intermediate steps...")
        for step in save_steps_list:
            step_folder = f"{args.out}/step_{step:03d}"
            if os.path.exists(step_folder):
                print(f"Creating .npz file for step {step}")
                create_npz_from_sample_folder(step_folder, total_samples)

    print("Done!")
    print(f"\nResults saved to: {args.out}")
    print(f"  - Distortion analysis: {args.out}/distortion_analysis.json")
    print(f"  - Correlation plots: {args.out}/correlation_step_*.png")
    print(f"  - Top distorted images: {args.out}/step_*/top_distorted/")
    print(f"  - Top undistorted images: {args.out}/step_*/top_undistorted/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean Image Stability Analysis for EqM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python sample_from_clean.py \\
    --val-data-path /path/to/imagenet/val \\
    --ckpt /path/to/checkpoint.pt \\
    --batch-size 8 \\
    --num-samples 100 \\
    --save-steps 0,50,100,249 \\
    --top-n 10 \\
    --out clean_stability_analysis
        """,
    )

    # Required arguments
    parser.add_argument(
        "--val-data-path",
        type=str,
        required=True,
        help="Path to ImageNet validation directory",
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to EqM checkpoint")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size for sampling")
    parser.add_argument(
        "--num-samples",
        type=int,
        required=True,
        help="Number of validation images to process (takes first N images)",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        choices=list(EqM_models.keys()),
        default="EqM-XL/2",
        help="EqM model architecture (default: EqM-XL/2)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        choices=[256, 512],
        default=256,
        help="Image resolution (default: 256)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1000,
        help="Number of classes (default: 1000)",
    )
    parser.add_argument(
        "--vae",
        type=str,
        choices=["ema", "mse"],
        default="ema",
        help="VAE variant (default: ema)",
    )
    parser.add_argument(
        "--uncond",
        type=bool,
        default=True,
        help="Disable/enable noise conditioning (default: True)",
    )
    parser.add_argument(
        "--ebm",
        type=str,
        choices=["none", "l2", "dot", "mean"],
        default="none",
        help="Energy formulation (default: none)",
    )

    # Sampling configuration
    parser.add_argument(
        "--sampler",
        type=str,
        default="gd",
        choices=["gd", "ngd"],
        help="Sampler type: 'gd' (gradient descent) or 'ngd' (NAG-GD) (default: gd)",
    )
    parser.add_argument(
        "--num-sampling-steps",
        type=int,
        default=250,
        help="Number of sampling iterations (default: 250)",
    )
    parser.add_argument(
        "--stepsize",
        type=float,
        default=0.0017,
        help="Step size eta for gradient descent (default: 0.0017)",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=0.3,
        help="NAG-GD momentum hyperparameter mu (default: 0.3)",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=4.0,
        help="Classifier-free guidance scale (default: 4.0)",
    )

    # Analysis configuration
    parser.add_argument(
        "--save-steps",
        type=str,
        default=None,
        help="Comma-separated list of sampling steps to analyze and save images (e.g., '0,50,100,249')",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top distorted/undistorted images to save per step (default: 10)",
    )
    parser.add_argument(
        "--track-grad-norm",
        action="store_true",
        help="Enable gradient norm tracking and visualization",
    )

    # Output configuration
    parser.add_argument(
        "--out",
        type=str,
        default="clean_stability_analysis",
        help="Output directory for results (default: clean_stability_analysis)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")

    args = parser.parse_args()
    main(args)
