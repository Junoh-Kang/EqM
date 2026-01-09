"""
Sampling utility functions for EqM models.

This module provides reusable sampling functions for Equilibrium Matching models.

Main Components:
    - sample_eqm(): Core sampling function with GD/NAG-GD support
    - create_npz_from_sample_folder(): Create .npz file for FID evaluation
    - decode_latents(): Decode VAE latents to images
    - encode_images_to_latent(): Encode images to VAE latents
    - compute_high_frequency_content(): Compute high-frequency content metric using FFT

For sampling hooks, see utils.sampling_hooks module.

Example Usage:
    >>> # Basic sampling
    >>> samples = sample_eqm(model, vae, device, batch_size=16, latent_size=32)
    >>>
    >>> # Sampling with hooks
    >>> from utils.sampling_hooks import IntermediateImageSaver, GradientNormTracker
    >>> img_hook = IntermediateImageSaver([0, 50, 100], "outputs")
    >>> grad_hook = GradientNormTracker(num_sampling_steps=250)
    >>> samples = sample_eqm(model, vae, device, batch_size=16, latent_size=32, hooks=[img_hook, grad_hook])
    >>> grad_hook.finalize("outputs", num_sampling_steps=250, stepsize=1.0, sampler="euler")
"""

import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from utils.sampling_hooks import GradientNormTracker, SamplingHookContext


@torch.no_grad()
def sample_eqm(
    model,
    vae,
    device,
    batch_size,
    latent_size,
    initial_latent=None,
    class_labels=None,
    num_sampling_steps=250,
    stepsize=0.0017,
    cfg_scale=4.0,
    sampler="gd",
    mu=0.3,
    hooks=None,
    return_cfg_components=False,
):
    """
    Generate samples using EqM model with gradient descent sampling.

    Args:
        model: The EqM model (should be in eval mode)
        vae: VAE decoder for latent to image conversion
        device: torch device to run on
        batch_size: Number of samples to generate
        latent_size: Size of latent space (image_size // 8)
        initial_latent: Initial latent noise tensor, shape (batch_size, 4, latent_size, latent_size).
                       If None, random noise is generated.
        class_labels: Specific class labels to use, shape (batch_size,).
                     If None, random labels from 0-999 are used.
        num_sampling_steps: Number of sampling iterations (default: 250)
        stepsize: Step size eta for gradient updates (default: 0.0017)
        cfg_scale: Classifier-free guidance scale (default: 4.0).
                   Set to > 1.0 to enable CFG.
        sampler: Sampling method, 'gd' (gradient descent) or 'ngd' (NAG-GD) (default: 'gd')
        mu: NAG-GD momentum hyperparameter (default: 0.3, only used when sampler='ngd')
        hooks: List of hook callables. Each hook receives a SamplingHookContext object
               at each sampling step. Use for monitoring, logging, or saving intermediate results.
               Example hooks: IntermediateImageSaver, GradientNormTracker.
        return_cfg_components: If True and CFG is enabled, passes individual cond/uncond outputs
                               to hooks via out_cond and out_uncond context fields.

    Returns:
        samples: Generated images as numpy array, shape (batch_size, H, W, 3), dtype uint8

    Example:
        >>> # Basic usage
        >>> samples = sample_eqm(model, vae, device, batch_size=16, latent_size=32)
        >>>
        >>> # With hooks for monitoring
        >>> from sampling_utils import IntermediateImageSaver, GradientNormTracker
        >>> img_hook = IntermediateImageSaver([0, 50, 100, 249], output_folder="outputs")
        >>> grad_hook = GradientNormTracker(num_sampling_steps=250)
        >>> samples = sample_eqm(model, vae, device, batch_size=16, latent_size=32, hooks=[img_hook, grad_hook])
        >>> grad_hook.finalize("outputs", num_sampling_steps=250, stepsize=1.0, sampler="euler")
    """
    if hooks is None:
        hooks = []

    # Auto-enable return_cfg_components if GradientNormTracker is present
    if any(isinstance(h, GradientNormTracker) for h in hooks):
        return_cfg_components = True

    use_cfg = cfg_scale > 1.0
    n = batch_size

    # Initialize random latent noise
    if initial_latent is not None:
        z = initial_latent.clone()
    else:
        z = torch.randn(n, 4, latent_size, latent_size, device=device)

    # Generate or use provided class labels
    if class_labels is None:
        y = torch.randint(0, 1000, (n,), device=device)
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
        for step_idx in tqdm(range(1, num_sampling_steps + 1)):
            out_cond = None
            out_uncond = None

            if sampler == "gd":
                # Standard gradient descent
                if use_cfg and return_cfg_components:
                    out, out_cond, out_uncond = model_fn(xt, t, y, cfg_scale, return_components=True)
                else:
                    out = model_fn(xt, t, y, cfg_scale)
                    if not torch.is_tensor(out):
                        out = out[0]
            elif sampler == "ngd":
                # Nesterov accelerated gradient descent
                x_ = xt + stepsize * m * mu
                if use_cfg and return_cfg_components:
                    out, out_cond, out_uncond = model_fn(x_, t, y, cfg_scale, return_components=True)
                else:
                    out = model_fn(x_, t, y, cfg_scale)
                    if not torch.is_tensor(out):
                        out = out[0]
                m = out
            else:
                raise ValueError(f"Unknown sampler: {sampler}")

            # Update latent and timestep
            xt = xt + out * stepsize
            t += stepsize

            # Call hooks with context
            if hooks:
                context = SamplingHookContext(
                    xt=xt,
                    t=t,
                    y=y,
                    out=out,
                    step_idx=step_idx,
                    use_cfg=use_cfg,
                    vae=vae,
                    device=device,
                    total_steps=num_sampling_steps,
                    out_cond=out_cond,
                    out_uncond=out_uncond,
                )
                for hook in hooks:
                    hook(context)

        # Remove duplicates from CFG
        if use_cfg:
            xt, _ = xt.chunk(2, dim=0)

        samples = decode_latents(vae, xt)

    return samples


@torch.no_grad()
def sample_eqm_two(
    model,
    vae,
    device,
    batch_size,
    latent_size,
    initial_latent=None,
    class_labels=None,
    num_sampling_steps=250,
    stepsize=0.0017,
    cfg_scale=4.0,
    sampler="gd",
    mu=0.3,
    hooks=None,
    return_cfg_components=False,
):
    """
    Generate samples using EqM model with gradient descent sampling.

    Args:
        model: The EqM model (should be in eval mode)
        vae: VAE decoder for latent to image conversion
        device: torch device to run on
        batch_size: Number of samples to generate
        latent_size: Size of latent space (image_size // 8)
        initial_latent: Initial latent noise tensor, shape (batch_size, 4, latent_size, latent_size).
                       If None, random noise is generated.
        class_labels: Specific class labels to use, shape (batch_size,).
                     If None, random labels from 0-999 are used.
        num_sampling_steps: Number of sampling iterations (default: 250)
        stepsize: Step size eta for gradient updates (default: 0.0017)
        cfg_scale: Classifier-free guidance scale (default: 4.0).
                   Set to > 1.0 to enable CFG.
        sampler: Sampling method, 'gd' (gradient descent) or 'ngd' (NAG-GD) (default: 'gd')
        mu: NAG-GD momentum hyperparameter (default: 0.3, only used when sampler='ngd')
        hooks: List of hook callables. Each hook receives a SamplingHookContext object
               at each sampling step. Use for monitoring, logging, or saving intermediate results.
               Example hooks: IntermediateImageSaver, GradientNormTracker.
        return_cfg_components: If True and CFG is enabled, passes individual cond/uncond outputs
                               to hooks via out_cond and out_uncond context fields.

    Returns:
        samples: Generated images as numpy array, shape (batch_size, H, W, 3), dtype uint8

    Example:
        >>> # Basic usage
        >>> samples = sample_eqm(model, vae, device, batch_size=16, latent_size=32)
        >>>
        >>> # With hooks for monitoring
        >>> from sampling_utils import IntermediateImageSaver, GradientNormTracker
        >>> img_hook = IntermediateImageSaver([0, 50, 100, 249], "outputs")
        >>> grad_hook = GradientNormTracker(num_sampling_steps=250)
        >>> samples = sample_eqm(model, vae, device, batch_size=16, latent_size=32, hooks=[img_hook, grad_hook])
        >>> grad_hook.finalize("outputs", num_sampling_steps=250, stepsize=1.0, sampler="euler")
    """
    if hooks is None:
        hooks = []

    # Auto-enable return_cfg_components if GradientNormTracker is present
    if any(isinstance(h, GradientNormTracker) for h in hooks):
        return_cfg_components = True

    use_cfg = cfg_scale > 1.0
    n = batch_size

    # Initialize random latent noise
    if initial_latent is not None:
        z = initial_latent.clone()
    else:
        z = torch.randn(n, 4, latent_size, latent_size, device=device)

    # Generate or use provided class labels
    if class_labels is None:
        y = torch.randint(0, 1000, (n,), device=device)
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
        for step_idx in range(1, num_sampling_steps + 1):
            out_cond = None
            out_uncond = None

            if sampler == "gd":
                # Standard gradient descent
                if use_cfg and return_cfg_components:
                    out, out_cond, out_uncond = model_fn(xt, t, y, cfg_scale, return_components=True)
                else:
                    out = model_fn(xt, t, y, cfg_scale)
                    if not torch.is_tensor(out):
                        out = out[0]
            elif sampler == "ngd":
                # Nesterov accelerated gradient descent
                x_ = xt + stepsize * m * mu
                if use_cfg and return_cfg_components:
                    out, out_cond, out_uncond = model_fn(x_, t, y, cfg_scale, return_components=True)
                else:
                    out = model_fn(x_, t, y, cfg_scale)
                    if not torch.is_tensor(out):
                        out = out[0]
                m = out
            else:
                raise ValueError(f"Unknown sampler: {sampler}")

            if step_idx == (num_sampling_steps + 1) // 2:
                if use_cfg:
                    original_y, _ = y.chunk(2, dim=0)
                else:
                    original_y = y
                new_y = []
                for orig_class in original_y:
                    # 원래 class를 제외한 랜덤 class 선택
                    available_classes = list(range(1000))
                    available_classes.remove(orig_class.item())
                    new_class = np.random.choice(available_classes)
                    new_y.append(new_class)

                y = torch.tensor(new_y, device=device)

                if use_cfg:
                    y_null = torch.tensor([1000] * n, device=device)
                    y = torch.cat([y, y_null], 0)
                    model_fn = model.forward_with_cfg
                else:
                    model_fn = model.forward

            # Update latent and timestep
            xt = xt + out * stepsize
            t += stepsize

            # Call hooks with context
            if hooks:
                context = SamplingHookContext(
                    xt=xt,
                    t=t,
                    y=y,
                    out=out,
                    step_idx=step_idx,
                    use_cfg=use_cfg,
                    vae=vae,
                    device=device,
                    total_steps=num_sampling_steps,
                    out_cond=out_cond,
                    out_uncond=out_uncond,
                )
                for hook in hooks:
                    hook(context)

        # Remove duplicates from CFG
        if use_cfg:
            xt, _ = xt.chunk(2, dim=0)

        samples = decode_latents(vae, xt)

    return samples


def create_npz_from_sample_folder(sample_dir, num=None):
    """
    Builds a single .npz file from a folder of .png samples.
    Compatible with ADM's TensorFlow evaluation suite.

    Args:
        sample_dir: Directory containing .png samples named as {i:06d}.png
        num: Number of samples to include in the .npz file. If None, auto-detect from directory.

    Returns:
        npz_path: Path to the created .npz file
    """
    if num is None:
        png_files = [f for f in os.listdir(sample_dir) if f.endswith(".png")]
        num = len(png_files)
        print(f"Auto-detected {num} samples in {sample_dir}")

    samples = []
    print(f"Building .npz file from {num} samples in {sample_dir}...")
    for i in range(num):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
        if (i + 1) % 10000 == 0:
            print(f"  Loaded {i + 1}/{num} samples...")

    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}/samples.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


@torch.no_grad()
def decode_latents(vae, latents):
    """
    Decode latent tensors to images using VAE decoder.

    Args:
        vae: VAE decoder (e.g., Stable Diffusion VAE)
        latents: Latent tensor of shape (batch_size, 4, H, W).
                 Will be scaled by 1/0.18215 before decoding.

    Returns:
        numpy array of shape (batch_size, H*8, W*8, 3), dtype uint8.
        Images are in [0, 255] range, RGB format.
    """
    samples = vae.decode(latents / 0.18215).sample
    samples = torch.clamp(127.5 * samples + 128.0, 0, 255)
    samples = samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
    return samples


@torch.no_grad()
def encode_images_to_latent(vae, images, device):
    """
    Encode images to latent space using VAE encoder.

    Args:
        vae: VAE encoder (e.g., Stable Diffusion VAE)
        images: Image tensor of shape (batch_size, 3, H, W).
                Should be normalized to [-1, 1] range.
        device: Device to run encoding on

    Returns:
        Latent tensor of shape (batch_size, 4, H//8, W//8), scaled by 0.18215.
        Ready to be used as initial_latent in sample_eqm().
    """
    images = images.to(device)
    latents = vae.encode(images).latent_dist.sample().mul_(0.18215)
    return latents


def compute_high_frequency_content(images):
    """
    Compute high-frequency content metric for images using 2D FFT.

    Args:
        images: Image tensor of shape (batch_size, 3, H, W) or numpy array (batch_size, H, W, 3).
                Values should be in [0, 255] range (uint8) or [-1, 1] range (float).

    Returns:
        high_freq_metrics: Numpy array of shape (batch_size,) containing high-frequency
                          energy for each image. Higher values indicate more high-frequency content.
    """
    # Convert numpy to torch if needed
    if isinstance(images, np.ndarray):
        # Assume (B, H, W, 3) format
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float()

    # Convert to grayscale for frequency analysis (simpler and more interpretable)
    # Using standard RGB to grayscale conversion: 0.299*R + 0.587*G + 0.114*B
    if images.shape[1] == 3:
        grayscale = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
    else:
        grayscale = images.squeeze(1)

    batch_size = grayscale.shape[0]
    high_freq_metrics = []

    for i in range(batch_size):
        # Compute 2D FFT
        fft = torch.fft.fft2(grayscale[i])
        fft_shifted = torch.fft.fftshift(fft)  # Shift zero frequency to center
        magnitude = torch.abs(fft_shifted)

        # Get dimensions
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2

        # Create a mask for high-frequency components (outer 50% of spectrum)
        # High frequencies are far from the center
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        dist_from_center = torch.sqrt(((y - center_h) ** 2 + (x - center_w) ** 2).float())
        max_dist = torch.sqrt(torch.tensor(center_h**2 + center_w**2).float())

        # High-frequency mask: distance > 50% of max distance
        high_freq_mask = dist_from_center > (0.5 * max_dist)

        # Compute energy in high-frequency bands
        high_freq_energy = (magnitude[high_freq_mask] ** 2).sum().item()
        total_energy = (magnitude**2).sum().item()

        # Normalize by total energy to get relative high-frequency content
        if total_energy > 0:
            high_freq_ratio = high_freq_energy / total_energy
        else:
            high_freq_ratio = 0.0

        high_freq_metrics.append(high_freq_ratio)

    return np.array(high_freq_metrics)
