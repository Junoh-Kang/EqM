"""
Sampling hooks for EqM models.

This module provides hook classes for monitoring, logging, and analysis during sampling.

Hook System:
    Hooks are callables that receive a SamplingHookContext at each sampling step.
    They enable monitoring, logging, and analysis without cluttering the core sampling loop.

Main Components:
    - SamplingHookContext: Context object passed to hooks during sampling
    - IntermediateImageSaver: Hook for saving intermediate images at specified steps
    - WandBImageLogger: Hook for logging intermediate images to WandB
    - DistortionTracker: Hook for analyzing image distortion and high-frequency correlation
    - GradientNormTracker: Hook for tracking and analyzing gradient norms

Example Usage:
    >>> from utils.sampling_hooks import IntermediateImageSaver, GradientNormTracker
    >>> img_hook = IntermediateImageSaver([0, 50, 100], "outputs")
    >>> grad_hook = GradientNormTracker(num_sampling_steps=250)
    >>> samples = sample_eqm(model, vae, device, batch_size=16, latent_size=32, hooks=[img_hook, grad_hook])
    >>> grad_hook.finalize("outputs", num_sampling_steps=250, stepsize=1.0, sampler="euler")
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


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
        output_folder: Base folder for saving images. Required if folder_pattern is not provided.
        folder_pattern: Callable (context) -> str that returns the folder path.
                        If not provided, defaults to "{output_folder}/step_{step_idx:03d}"
    """

    def __init__(
        self,
        save_steps: list[int],
        output_folder: str | None = None,
        folder_pattern: Callable[[SamplingHookContext], str] | None = None,
    ):
        if output_folder is None and folder_pattern is None:
            raise ValueError("Either output_folder or folder_pattern must be provided")
        self.save_steps = set(save_steps)  # Use set for O(1) lookup
        # Track global sample counter for each step to avoid overwriting
        self.step_counters = dict.fromkeys(save_steps, 0)
        if folder_pattern is None:
            self.folder_pattern = lambda ctx: f"{output_folder}/step_{ctx.step_idx:03d}"
        else:
            self.folder_pattern = folder_pattern

    def __call__(self, context: SamplingHookContext):
        """Save images if current step is in save_steps list."""
        from utils.sampling_utils import decode_latents

        if context.step_idx not in self.save_steps:
            return

        step_folder = self.folder_pattern(context)
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
        from utils.sampling_utils import decode_latents

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
        from utils.sampling_utils import decode_latents

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
            for _batch_start, batch_latents in sorted(self.latent_batches[step_idx]):
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

    def finalize(self, folder: str, num_sampling_steps: int, stepsize: float, sampler: str):
        """
        Compute statistics and create visualization for gradient norms.
        Call this after all sampling is complete.

        Args:
            folder: Output directory for saving JSON and plot
            num_sampling_steps: Number of sampling steps
            stepsize: Step size used during sampling
            sampler: Sampler name (e.g., 'euler', 'heun')
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
            "num_sampling_steps": num_sampling_steps,
            "total_samples": len(self.gradient_norms[0]) if len(self.gradient_norms[0]) > 0 else 0,
            "mean": gradient_means,
            "std": gradient_stds,
            "stepsize": stepsize,
            "sampler": sampler,
            "note": "Statistics computed from individual gradient L2 norms across all samples (batch-size independent)",
        }
        json_path = f"{folder}/gradient_norms.json"
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved gradient norm statistics to {json_path}")

        # Create plot
        print("Creating gradient norm plot...")
        steps = np.arange(0, num_sampling_steps)
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
            f"Gradient L2 Norm during Sampling ({sampler.upper()}, stepsize={stepsize})",
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
