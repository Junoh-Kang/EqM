# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sampling utility functions for EqM models.
"""
import torch
import numpy as np
from PIL import Image


def sample_eqm(
    model,
    vae,
    device,
    batch_size,
    latent_size,
    num_classes=1000,
    num_sampling_steps=250,
    stepsize=0.0017,
    cfg_scale=4.0,
    sampler='gd',
    mu=0.3,
    seed=None,
    class_labels=None,
    initial_noise=None
):
    """
    Generate samples using EqM model with gradient descent sampling.

    Args:
        model: The EqM model (should be in eval mode)
        vae: VAE decoder for latent to image conversion
        device: torch device to run on
        batch_size: Number of samples to generate
        latent_size: Size of latent space (image_size // 8)
        num_classes: Number of classes for conditional generation (default: 1000)
        num_sampling_steps: Number of sampling iterations (default: 250)
        stepsize: Step size eta for gradient updates (default: 0.0017)
        cfg_scale: Classifier-free guidance scale (default: 4.0)
        sampler: Sampling method, 'gd' or 'ngd' (default: 'gd')
        mu: NAG-GD hyperparameter for momentum (default: 0.3)
        seed: Random seed for reproducibility (optional)
        class_labels: Specific class labels to use, shape (batch_size,). If None, random labels are used.
        initial_noise: Initial latent noise tensor, shape (batch_size, 4, latent_size, latent_size). If None, random noise is generated.

    Returns:
        samples: Generated images as numpy array, shape (batch_size, H, W, 3), dtype uint8
    """
    if seed is not None:
        torch.manual_seed(seed)

    use_cfg = cfg_scale > 1.0
    n = batch_size

    # Initialize random latent noise
    if initial_noise is not None:
        z = initial_noise.clone()
    else:
        z = torch.randn(n, 4, latent_size, latent_size, device=device)

    # Generate or use provided class labels
    if class_labels is None:
        y = torch.randint(0, num_classes, (n,), device=device)
    else:
        y = class_labels.to(device)

    # Initialize timestep
    t = torch.ones((n,)).to(device)

    # Setup classifier-free guidance
    if use_cfg:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=device)
        y = torch.cat([y, y_null], 0)
        t = torch.cat([t, t], 0)
        model_fn = model.forward_with_cfg
    else:
        model_fn = model.forward

    # Initialize latent variable and momentum
    xt = z
    m = torch.zeros_like(xt).to(device)

    # Sampling loop
    with torch.no_grad():
        for _ in range(num_sampling_steps - 1):
            if sampler == 'gd':
                # Standard gradient descent
                out = model_fn(xt, t, y, cfg_scale)
                if not torch.is_tensor(out):
                    out = out[0]
            elif sampler == 'ngd':
                # Nesterov accelerated gradient descent
                x_ = xt + stepsize * m * mu
                out = model_fn(x_, t, y, cfg_scale)
                if not torch.is_tensor(out):
                    out = out[0]
                m = out
            else:
                raise ValueError(f"Unknown sampler: {sampler}")

            # Update latent and timestep
            xt = xt + out * stepsize
            t += stepsize

        # Remove duplicates from CFG
        if use_cfg:
            xt, _ = xt.chunk(2, dim=0)

        # Decode latents to images
        samples = vae.decode(xt / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255)
        samples = samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

    return samples


def save_samples(samples, output_dir, start_idx=0):
    """
    Save generated samples as PNG files.

    Args:
        samples: numpy array of images, shape (N, H, W, 3), dtype uint8
        output_dir: Directory to save images
        start_idx: Starting index for filenames (default: 0)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    for i, sample in enumerate(samples):
        index = start_idx + i
        Image.fromarray(sample).save(f"{output_dir}/{index:06d}.png")


def sample_and_save(
    model,
    vae,
    device,
    output_dir,
    total_samples=50000,
    batch_size=256,
    latent_size=32,
    **sampling_kwargs
):
    """
    Generate and save a large number of samples in batches.

    Args:
        model: The EqM model (should be in eval mode)
        vae: VAE decoder
        device: torch device
        output_dir: Directory to save samples
        total_samples: Total number of samples to generate (default: 50000)
        batch_size: Batch size for generation (default: 256)
        latent_size: Latent space size (default: 32)
        **sampling_kwargs: Additional arguments passed to sample_eqm()
    """
    import os
    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)

    iterations = (total_samples + batch_size - 1) // batch_size
    total_generated = 0

    for _ in tqdm(range(iterations), desc="Generating samples"):
        # Adjust batch size for last iteration
        current_batch = min(batch_size, total_samples - total_generated)

        samples = sample_eqm(
            model=model,
            vae=vae,
            device=device,
            batch_size=current_batch,
            latent_size=latent_size,
            **sampling_kwargs
        )

        save_samples(samples, output_dir, start_idx=total_generated)
        total_generated += current_batch

    print(f"Generated {total_generated} samples in {output_dir}")
