"""
A simplified single-GPU sampling script for EqM.
Uses hooks from sampling_utils.py for cleaner code.
"""

import math

import numpy as np
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
from utils.sampling_utils import (
    GradientNormTracker,
    IntermediateImageSaver,
    create_npz_from_sample_folder,
    decode_latents,
    sample_eqm,
)


def load_initial_latents(latents_path, latent_size):
    """
    Load initial latents tensor from an .npz file using the default arr_0 key.
    """
    if not os.path.exists(latents_path):
        raise FileNotFoundError(f"Initial latents file not found: {latents_path}")

    with np.load(latents_path, allow_pickle=False) as npz_file:
        if "arr_0" not in npz_file.files:
            raise ValueError(f"'arr_0' not found in {latents_path}. Ensure the file was created with np.savez.")
        latents = npz_file["arr_0"]

    if latents.ndim != 4 or latents.shape[1:] != (4, latent_size, latent_size):
        raise ValueError(f"Latents must have shape (N, 4, {latent_size}, {latent_size}). Got {latents.shape}.")

    latents_tensor = torch.from_numpy(latents).to(torch.float32)
    if latents_tensor.shape[0] == 0:
        raise ValueError("Initial latents file is empty.")
    return latents_tensor


def load_initial_class_labels(labels_path, expected_count):
    """
    Load class labels (one per line) for initial latents.
    """
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Initial class labels file not found: {labels_path}")

    labels = []
    with open(labels_path, encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                labels.append(int(stripped))
            except ValueError as err:
                raise ValueError(f"Invalid class label '{stripped}' on line {line_idx} in {labels_path}") from err

    if len(labels) != expected_count:
        raise ValueError(
            f"Expected {expected_count} class labels, but found {len(labels)} in {labels_path}. "
            "Ensure there is exactly one label per latent."
        )

    return torch.tensor(labels, dtype=torch.long)


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
    final_step_folder = f"{args.out}/step_{args.num_sampling_steps:03d}"
    os.makedirs(final_step_folder, exist_ok=True)

    # Optionally load initial latents
    initial_latents_tensor = None
    initial_class_labels = None
    if args.initial_latents is not None:
        initial_latents_tensor = load_initial_latents(args.initial_latents, latent_size)
        available_latents = initial_latents_tensor.shape[0]

        if args.initial_class_labels is None:
            raise ValueError("--initial-class-labels must be provided when --initial-latents is used.")
        initial_class_labels = load_initial_class_labels(args.initial_class_labels, available_latents)

        if args.class_labels is not None:
            raise ValueError("--class-labels cannot be combined with --initial-latents. Use --initial-class-labels.")

        total_samples = min(args.num_samples, available_latents)
        if args.num_samples > available_latents:
            print(
                f"Requested {args.num_samples} samples but initial latents file only provides {available_latents}. "
                "Sampling available latents and stopping."
            )
        elif args.num_samples < available_latents:
            print(f"Using first {total_samples} initial latents from {args.initial_latents}")
        else:
            print(f"Using all {total_samples} initial latents from {args.initial_latents}")

        initial_latents_tensor = initial_latents_tensor[:total_samples]
        initial_class_labels = initial_class_labels[:total_samples]
    else:
        total_samples = args.num_samples

    if total_samples <= 0:
        raise ValueError("Total samples must be greater than zero.")

    print(f"Total number of images that will be sampled: {total_samples}")
    iterations = int(math.ceil(total_samples / args.batch_size))

    # Create hooks
    hooks = []
    save_steps_list = []
    if args.save_steps is not None:
        save_steps_list = [int(s.strip()) for s in args.save_steps.split(",")]
        img_saver = IntermediateImageSaver(save_steps_list, args.out)
        hooks.append(img_saver)
        print(f"Created IntermediateImageSaver hook for steps: {save_steps_list}")

    grad_tracker = None
    if args.track_grad_norm:
        grad_tracker = GradientNormTracker(args.num_sampling_steps)
        hooks.append(grad_tracker)
        print("Created GradientNormTracker hook")

    # Sampling loop
    print(f"Starting sampling with {args.sampler.upper()} sampler...")
    if initial_class_labels is None and args.class_labels is not None:
        class_ids = [int(c.strip()) for c in args.class_labels.split(",") if c.strip()]
        if len(class_ids) == 0:
            raise ValueError("At least one valid class label must be provided.")
        class_ids_tensor = torch.tensor(class_ids, device=device, dtype=torch.long)
    else:
        class_ids_tensor = None

    total_saved = 0
    for batch_idx in tqdm(range(iterations), desc="Generating samples"):
        batch_start = batch_idx * args.batch_size
        if batch_start >= total_samples:
            break
        batch_end = min(batch_start + args.batch_size, total_samples)
        current_batch_size = batch_end - batch_start

        batch_latents = None
        if initial_latents_tensor is not None:
            batch_latents = initial_latents_tensor[batch_start:batch_end].to(device)

            # Save initial latents as images
            initial_step_folder = f"{args.out}/step_000"
            os.makedirs(initial_step_folder, exist_ok=True)
            initial_images = decode_latents(vae, batch_latents)
            for i_sample, img in enumerate(initial_images):
                index = batch_start + i_sample
                Image.fromarray(img).save(f"{initial_step_folder}/{index:06d}.png")

        if initial_class_labels is not None:
            class_labels = initial_class_labels[batch_start:batch_end].to(device)
        elif class_ids_tensor is not None:
            random_indices = torch.randint(0, class_ids_tensor.numel(), (current_batch_size,), device=device)
            class_labels = class_ids_tensor[random_indices]
        else:
            class_labels = None

        samples = sample_eqm(
            model=ema_model,
            vae=vae,
            device=device,
            batch_size=current_batch_size,
            latent_size=latent_size,
            initial_latent=batch_latents,
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

        total_saved += current_batch_size
    print(f"Saved {total_saved} samples to {final_step_folder}")

    # Finalize gradient norm statistics if enabled
    if grad_tracker is not None:
        print("Computing gradient norm statistics...")
        grad_tracker.finalize(args, args.out)

    # Create .npz files for FID evaluation
    print("Creating .npz file for final samples...")
    npz_sample_count = min(args.num_samples, total_saved)
    create_npz_from_sample_folder(final_step_folder, npz_sample_count)

    # Create .npz files for intermediate steps
    if len(save_steps_list) > 0:
        print("Creating .npz files for intermediate steps...")
        for step in save_steps_list:
            step_folder = f"{args.out}/step_{step:03d}"
            if os.path.exists(step_folder):
                print(f"Creating .npz file for step {step}")
                create_npz_from_sample_folder(step_folder, npz_sample_count)

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
        "--initial-latents",
        type=str,
        default=None,
        help="Path to a .npz file containing initial latents (expects default arr_0 key)",
    )
    parser.add_argument(
        "--initial-class-labels",
        type=str,
        default=None,
        help="Path to a text file with one class label per line for initial latents",
    )
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
