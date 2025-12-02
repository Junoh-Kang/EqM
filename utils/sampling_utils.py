"""
Sampling utility functions for EqM models.

This module provides reusable sampling functions and hooks for Equilibrium Matching models.

Main Components:
    - sample_eqm(): Core sampling function with GD/NAG-GD support
    - SamplingHookContext: Context object passed to hooks during sampling
    - IntermediateImageSaver: Hook for saving intermediate images at specified steps
    - GradientNormTracker: Hook for tracking and analyzing gradient norms
    - DistortionTracker: Hook for analyzing image distortion and high-frequency correlation
    - create_npz_from_sample_folder(): Create .npz file for FID evaluation
    - decode_latents(): Decode VAE latents to images
    - encode_images_to_latent(): Encode images to VAE latents
    - compute_high_frequency_content(): Compute high-frequency content metric using FFT

Hook System:
    Hooks are callables that receive a SamplingHookContext at each sampling step.
    They enable monitoring, logging, and analysis without cluttering the core sampling loop.

Example Usage:
    >>> # Basic sampling
    >>> samples = sample_eqm(model, vae, device, batch_size=16, latent_size=32)
    >>>
    >>> # Sampling with hooks
    >>> img_hook = IntermediateImageSaver([0, 50, 100], "outputs")
    >>> grad_hook = GradientNormTracker(num_sampling_steps=250)
    >>> samples = sample_eqm(model, vae, device, batch_size=16, latent_size=32, hooks=[img_hook, grad_hook])
    >>> grad_hook.finalize(args, "outputs")
"""

import json
import os
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


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
    hooks=[],
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
        >>> grad_hook.finalize(args, "outputs")
    """
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
            if sampler == "gd":
                # Standard gradient descent
                out = model_fn(xt, t, y, cfg_scale)
                if not torch.is_tensor(out):
                    out = out[0]
            elif sampler == "ngd":
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
    hooks=[],
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
        >>> grad_hook.finalize(args, "outputs")
    """
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
            if sampler == "gd":
                # Standard gradient descent
                out = model_fn(xt, t, y, cfg_scale)
                if not torch.is_tensor(out):
                    out = out[0]
            elif sampler == "ngd":
                # Nesterov accelerated gradient descent
                x_ = xt + stepsize * m * mu
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
                )
                for hook in hooks:
                    hook(context)

        # Remove duplicates from CFG
        if use_cfg:
            xt, _ = xt.chunk(2, dim=0)

        samples = decode_latents(vae, xt)

    return samples


@dataclass
class SamplingHookContext:
    """Context object passed to sampling hooks containing all relevant state."""

    xt: torch.Tensor  # Current latent state
    t: torch.Tensor  # Current timestep
    y: torch.Tensor  # Class labels
    out: torch.Tensor  # Model output/gradient
    step_idx: int  # Current step index (1-indexed)
    use_cfg: bool  # Whether CFG is enabled
    vae: Any  # VAE decoder for image conversion
    device: torch.device  # Device
    total_steps: int  # Total number of sampling steps


class IntermediateImageSaver:
    """
    Hook for saving intermediate images during sampling.

    Args:
        save_steps: List of step indices at which to save images (e.g., [0, 50, 100, 250])
        output_folder: Base folder for saving images
    """

    def __init__(self, save_steps, output_folder):
        self.save_steps = set(save_steps)  # Use set for O(1) lookup
        self.output_folder = output_folder
        # Track global sample counter for each step to avoid overwriting
        self.step_counters = dict.fromkeys(save_steps, 0)
        # self.return_images = return_images

    def __call__(self, context: SamplingHookContext):
        """Save images if current step is in save_steps list."""
        if context.step_idx not in self.save_steps:
            return

        step_folder = f"{self.output_folder}/step_{context.step_idx:03d}"
        os.makedirs(step_folder, exist_ok=True)

        # Extract conditional part if using CFG
        xt_save = context.xt
        if context.use_cfg:
            batch_size = context.xt.shape[0] // 2
            xt_save = context.xt[:batch_size]

        samples = decode_latents(context.vae, xt_save)

        # Save images with global sequential indexing across batches
        start_idx = self.step_counters[context.step_idx]
        for i_sample, sample in enumerate(samples):
            global_idx = start_idx + i_sample
            Image.fromarray(sample).save(f"{step_folder}/{global_idx:06d}.png")

        # Update counter for this step
        self.step_counters[context.step_idx] += len(samples)


class WandBImageLogger:
    """
    Hook for logging intermediate images during sampling directly to WandB.

    Args:
        save_steps: List of step indices at which to log images (e.g., [5, 10, 250])
        train_step: Current training step (for WandB logging)
        output_folder: Folder to save logged images
        wandb_module: wandb module (pass wandb if imported, or None to skip logging)
    """

    def __init__(self, save_steps, train_step, output_folder, wandb_module=None):
        self.save_steps = set(save_steps)
        self.train_step = train_step
        self.output_folder = output_folder
        self.wandb = wandb_module
        self.logged_images = {step: [] for step in save_steps}
        self.step_counters = dict.fromkeys(save_steps, 0)

    def __call__(self, context: SamplingHookContext):
        """Log images to WandB if current step is in save_steps list."""
        if context.step_idx not in self.save_steps or self.wandb is None:
            return

        folder = f"{self.output_folder}/train_{self.train_step:04d}/sample_{context.step_idx:03d}"
        os.makedirs(folder, exist_ok=True)

        # Extract conditional part if using CFG
        xt_save = context.xt
        if context.use_cfg:
            batch_size = context.xt.shape[0] // 2
            xt_save = context.xt[:batch_size]

        # Decode latents to images
        samples = decode_latents(context.vae, xt_save)

        # Convert to wandb.Image objects
        start_idx = self.step_counters[context.step_idx]
        for i_sample, sample in enumerate(samples):
            global_idx = start_idx + i_sample
            img = Image.fromarray(sample)
            img.save(f"{folder}/{global_idx:03d}.png")
            self.logged_images[context.step_idx].append(self.wandb.Image(img, caption=f"Sample {global_idx:03d}"))

    def finalize(self):
        """Log all collected images to WandB. Call this after sampling is complete."""
        if self.wandb is None:
            return

        for step_idx in sorted(self.save_steps):
            if len(self.logged_images[step_idx]) > 0:
                self.wandb.log({f"samples/step_{step_idx:03d}": self.logged_images[step_idx]}, step=self.train_step)


class DistortionTracker:
    """
    Hook for tracking distortion of clean images during sampling.

    Computes L2 distance between original and current latents at specified steps,
    correlates with high-frequency content, and saves top distorted/undistorted images.

    Args:
        original_latents: Original clean latents, shape (batch_size, 4, H, W)
        high_freq_metrics: High-frequency content metrics per image, shape (batch_size,)
        save_steps: List of step indices at which to track distortion
        output_folder: Base folder for saving results
        top_n: Number of top distorted/undistorted images to save per step
    """

    def __init__(self, original_latents, high_freq_metrics, save_steps, output_folder, top_n=10):
        self.original_latents = original_latents.clone()
        self.high_freq_metrics = high_freq_metrics
        self.save_steps = set(save_steps)
        self.output_folder = output_folder
        self.top_n = top_n

        # Storage for distortion metrics at each step
        # Key: step_idx, Value: list of L2 distances
        self.distortions = {step: [] for step in save_steps}

        # Storage for batch information
        # Key: step_idx, Value: list of (batch_start_idx, batch_latents)
        self.latent_batches = {step: [] for step in save_steps}

        self.batch_counter = 0

    def __call__(self, context: SamplingHookContext):
        """Track distortion if current step is in save_steps list."""
        if context.step_idx not in self.save_steps:
            return

        # Extract conditional part if using CFG
        xt_current = context.xt
        if context.use_cfg:
            batch_size = context.xt.shape[0] // 2
            xt_current = context.xt[:batch_size]

        # Get corresponding original latents for this batch
        batch_size = xt_current.shape[0]
        batch_start = self.batch_counter
        batch_end = batch_start + batch_size
        original_batch = self.original_latents[batch_start:batch_end]

        # Compute L2 distance in latent space for each sample
        diff = xt_current - original_batch
        l2_distances = torch.linalg.norm(diff.reshape(diff.shape[0], -1), dim=1)  # shape: (batch_size,)

        # Store distortion metrics
        self.distortions[context.step_idx].extend(l2_distances.cpu().tolist())

        # Store latents for later saving
        self.latent_batches[context.step_idx].append((batch_start, xt_current.cpu().clone()))

    def on_batch_complete(self, batch_size):
        """Call this after each batch to update the batch counter."""
        self.batch_counter += batch_size

    def finalize(self, vae, output_folder):
        """
        Compute statistics, save top images, and create visualizations.
        Call this after all sampling is complete.

        Args:
            vae: VAE decoder for converting latents to images
            output_folder: Output directory for saving results
        """
        print("Analyzing distortion metrics and creating visualizations...")

        results = {"steps": {}, "high_freq_metrics": self.high_freq_metrics.tolist()}

        for step_idx in sorted(self.save_steps):
            print(f"  Processing step {step_idx}...")

            # Get all distortions for this step
            distortions = np.array(self.distortions[step_idx])

            if len(distortions) == 0:
                print(f"    Warning: No distortions recorded for step {step_idx}")
                continue

            # Reconstruct full latent tensor from batches
            latent_list = []
            for batch_start, batch_latents in sorted(self.latent_batches[step_idx]):
                latent_list.append(batch_latents)
            all_latents = torch.cat(latent_list, dim=0)

            # Get indices of top-N most and least distorted
            top_distorted_indices = np.argsort(distortions)[-self.top_n :][::-1]
            top_undistorted_indices = np.argsort(distortions)[: self.top_n]

            # Save top distorted images
            distorted_folder = f"{output_folder}/step_{step_idx:03d}/top_distorted"
            os.makedirs(distorted_folder, exist_ok=True)
            for rank, idx in enumerate(top_distorted_indices):
                latent = all_latents[idx : idx + 1].to(vae.device)  # Move to VAE device
                image = decode_latents(vae, latent)[0]
                Image.fromarray(image).save(
                    f"{distorted_folder}/rank{rank:02d}_idx{idx:06d}_dist{distortions[idx]:.4f}.png"
                )

            # Save top undistorted images
            undistorted_folder = f"{output_folder}/step_{step_idx:03d}/top_undistorted"
            os.makedirs(undistorted_folder, exist_ok=True)
            for rank, idx in enumerate(top_undistorted_indices):
                latent = all_latents[idx : idx + 1].to(vae.device)  # Move to VAE device
                image = decode_latents(vae, latent)[0]
                Image.fromarray(image).save(
                    f"{undistorted_folder}/rank{rank:02d}_idx{idx:06d}_dist{distortions[idx]:.4f}.png"
                )

            # Compute correlation with high-frequency content
            # Use only valid indices (in case of batch size mismatch)
            valid_indices = min(len(distortions), len(self.high_freq_metrics))
            distortions_valid = distortions[:valid_indices]
            high_freq_valid = self.high_freq_metrics[:valid_indices]

            correlation = np.corrcoef(high_freq_valid, distortions_valid)[0, 1]

            # Create scatter plot
            plt.figure(figsize=(10, 8))
            plt.scatter(high_freq_valid, distortions_valid, alpha=0.5, s=20)
            plt.xlabel("High-Frequency Content (Ratio)", fontsize=12)
            plt.ylabel("L2 Distance from Original (Latent Space)", fontsize=12)
            plt.title(
                f"Distortion vs High-Frequency Content at Step {step_idx}\nPearson Correlation: {correlation:.4f}",
                fontsize=14,
            )
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plot_path = f"{output_folder}/correlation_step_{step_idx:03d}.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"    Saved correlation plot to {plot_path}")

            # Store statistics
            results["steps"][str(step_idx)] = {
                "mean_distortion": float(np.mean(distortions)),
                "std_distortion": float(np.std(distortions)),
                "min_distortion": float(np.min(distortions)),
                "max_distortion": float(np.max(distortions)),
                "correlation_with_high_freq": float(correlation),
                "num_samples": len(distortions),
                "top_distorted_indices": top_distorted_indices.tolist(),
                "top_undistorted_indices": top_undistorted_indices.tolist(),
                "top_distorted_values": distortions[top_distorted_indices].tolist(),
                "top_undistorted_values": distortions[top_undistorted_indices].tolist(),
            }

        # Save results to JSON
        json_path = f"{output_folder}/distortion_analysis.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved distortion analysis to {json_path}")


class GradientNormTracker:
    """
    Hook for tracking gradient L2 norms during sampling.

    Args:
        num_steps: Number of sampling steps (for pre-allocating storage)
    """

    def __init__(self, num_steps):
        self.gradient_norms = [[] for _ in range(num_steps)]

    def __call__(self, context: SamplingHookContext):
        """Accumulate gradient L2 norms for the current step."""
        # Extract conditional part if using CFG
        out_for_norm = context.out
        if context.use_cfg:
            batch_size = context.out.shape[0] // 2
            out_for_norm = context.out[:batch_size]

        # Compute L2 norm for each sample in the batch
        norms = torch.linalg.norm(out_for_norm.reshape(out_for_norm.shape[0], -1), dim=1)  # shape: (batch_size,)
        self.gradient_norms[context.step_idx - 1].extend(norms.cpu().tolist())

    def finalize(self, args, folder):
        """
        Compute statistics and create visualization for gradient norms.
        Call this after all sampling is complete.

        Args:
            args: Arguments containing sampling parameters (num_sampling_steps, stepsize, sampler)
            folder: Output directory for saving JSON and plot
        """
        print("Computing gradient norm statistics...")
        gradient_means = []
        gradient_stds = []

        for step_norms in self.gradient_norms:
            if len(step_norms) > 0:
                gradient_means.append(np.mean(step_norms))
                gradient_stds.append(np.std(step_norms))
            else:
                gradient_means.append(0.0)
                gradient_stds.append(0.0)

        # Save statistics to JSON
        stats = {
            "num_sampling_steps": args.num_sampling_steps,
            "total_samples": len(self.gradient_norms[0]) if len(self.gradient_norms[0]) > 0 else 0,
            "mean": gradient_means,
            "std": gradient_stds,
            "stepsize": args.stepsize,
            "sampler": args.sampler,
            "note": "Statistics computed from individual gradient L2 norms across all samples (batch-size independent)",
        }
        json_path = f"{folder}/gradient_norms.json"
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved gradient norm statistics to {json_path}")

        # Create plot
        print("Creating gradient norm plot...")
        steps = np.arange(0, args.num_sampling_steps)
        gradient_means = np.array(gradient_means)
        gradient_stds = np.array(gradient_stds)

        plt.figure(figsize=(10, 6))
        plt.plot(steps, gradient_means, linewidth=2, label="Mean L2 Norm")
        plt.fill_between(
            steps,
            gradient_means - gradient_stds,
            gradient_means + gradient_stds,
            alpha=0.3,
            label="Mean ± Std",
        )
        plt.xlabel("Sampling Step", fontsize=12)
        plt.ylabel("Gradient L2 Norm", fontsize=12)
        plt.title(
            f"Gradient L2 Norm during Sampling ({args.sampler.upper()}, stepsize={args.stepsize})",
            fontsize=14,
        )
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = f"{folder}/gradient_norms.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Saved gradient norm plot to {plot_path}")
        plt.close()

    def finalize_for_wandb(self, wandb_module, train_step):
        """
        Log gradient norm statistics to WandB.
        Call this after all sampling is complete.

        Args:
            wandb_module: wandb module for logging
            train_step: Current training step for WandB logging
        """
        if wandb_module is None:
            return

        # Compute statistics
        gradient_means = []
        gradient_stds = []

        for step_norms in self.gradient_norms:
            if len(step_norms) > 0:
                gradient_means.append(np.mean(step_norms))
                gradient_stds.append(np.std(step_norms))
            else:
                gradient_means.append(0.0)
                gradient_stds.append(0.0)

        # Create a table for gradient norms vs sampling steps
        # This allows WandB to plot with sampling_step on x-axis
        data = []
        for sampling_step, (mean_norm, std_norm) in enumerate(zip(gradient_means, gradient_stds)):
            data.append(
                [
                    sampling_step,
                    mean_norm,
                    std_norm,
                    mean_norm + std_norm,  # Upper bound
                    mean_norm - std_norm,  # Lower bound
                ]
            )

        table = wandb_module.Table(
            columns=[
                "sampling_step",
                "gradient_norm_mean",
                "gradient_norm_std",
                "gradient_norm_upper",
                "gradient_norm_lower",
            ],
            data=data,
        )
        # Log the table with a consistent key (train_step is already in the table data)
        wandb_module.log({"gradient_norms": table}, step=train_step)


def create_npz_from_sample_folder(sample_dir, num):
    """
    Builds a single .npz file from a folder of .png samples.
    Compatible with ADM's TensorFlow evaluation suite.

    Args:
        sample_dir: Directory containing .png samples named as {i:06d}.png
        num: Number of samples to include in the .npz file (default: 50000)

    Returns:
        npz_path: Path to the created .npz file
    """
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
