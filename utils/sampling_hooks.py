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
    out: torch.Tensor  # Model output/gradient (CFG-combined when use_cfg=True)
    step_idx: int  # Current step index (1-indexed)
    use_cfg: bool  # Whether CFG is enabled
    vae: Any  # VAE decoder for image conversion
    device: torch.device  # Device
    total_steps: int  # Total number of sampling steps
    out_cond: torch.Tensor | None = None  # Class label output before CFG (only when use_cfg=True)
    out_uncond: torch.Tensor | None = None  # Null label output before CFG (only when use_cfg=True)


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

    When CFG is used, tracks three norms:
    - gradient_norms: CFG-applied output (combined)
    - gradient_norms_cond: conditional output (out_cond)
    - gradient_norms_uncond: unconditional output (out_uncond)

    When CFG is not used, only gradient_norms is tracked.

    Args:
        num_steps: Number of sampling steps (for pre-allocating storage)
    """

    def __init__(self, num_steps):
        self.gradient_norms = [[] for _ in range(num_steps)]  # CFG-applied output norms
        self.gradient_norms_cond = [[] for _ in range(num_steps)]  # conditional output norms
        self.gradient_norms_uncond = [[] for _ in range(num_steps)]  # unconditional output norms

    def __call__(self, context: SamplingHookContext):
        """Accumulate gradient L2 norms for the current step."""
        # Track CFG-applied output norm (first half, since duplicated)
        out_for_norm = context.out
        if context.use_cfg:
            batch_size = context.out.shape[0] // 2
            out_for_norm = context.out[:batch_size]
        norms = torch.linalg.norm(out_for_norm.reshape(out_for_norm.shape[0], -1), dim=1)
        self.gradient_norms[context.step_idx - 1].extend(norms.cpu().tolist())

        # Track cond/uncond norms when CFG is used
        if context.use_cfg and context.out_cond is not None and context.out_uncond is not None:
            out_cond = context.out_cond
            out_uncond = context.out_uncond

            norms_cond = torch.linalg.norm(out_cond.reshape(out_cond.shape[0], -1), dim=1)
            self.gradient_norms_cond[context.step_idx - 1].extend(norms_cond.cpu().tolist())

            norms_uncond = torch.linalg.norm(out_uncond.reshape(out_uncond.shape[0], -1), dim=1)
            self.gradient_norms_uncond[context.step_idx - 1].extend(norms_uncond.cpu().tolist())

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

        def compute_stats(norm_list):
            means, stds = [], []
            for step_norms in norm_list:
                if len(step_norms) > 0:
                    means.append(np.mean(step_norms))
                    stds.append(np.std(step_norms))
                else:
                    means.append(0.0)
                    stds.append(0.0)
            return means, stds

        # Compute stats for CFG-applied output
        cfg_means, cfg_stds = compute_stats(self.gradient_norms)

        # Compute stats for cond/uncond outputs
        has_cond = any(len(s) > 0 for s in self.gradient_norms_cond)
        has_uncond = any(len(s) > 0 for s in self.gradient_norms_uncond)
        cond_means, cond_stds = compute_stats(self.gradient_norms_cond) if has_cond else ([], [])
        uncond_means, uncond_stds = compute_stats(self.gradient_norms_uncond) if has_uncond else ([], [])

        # Save statistics to JSON
        stats = {
            "num_sampling_steps": num_sampling_steps,
            "total_samples": len(self.gradient_norms[0]) if len(self.gradient_norms[0]) > 0 else 0,
            "cfg_output": {"mean": cfg_means, "std": cfg_stds},
            "stepsize": stepsize,
            "sampler": sampler,
            "note": "Statistics computed from individual gradient L2 norms across all samples",
        }
        if has_cond:
            stats["cond_output"] = {"mean": cond_means, "std": cond_stds}
        if has_uncond:
            stats["uncond_output"] = {"mean": uncond_means, "std": uncond_stds}

        json_path = f"{folder}/gradient_norms.json"
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved gradient norm statistics to {json_path}")

        # Create plot
        print("Creating gradient norm plot...")
        steps = np.arange(0, num_sampling_steps)
        cfg_means = np.array(cfg_means)
        cfg_stds = np.array(cfg_stds)

        plt.figure(figsize=(10, 6))

        # Plot CFG-applied output norms
        plt.plot(steps, cfg_means, linewidth=2, label="CFG Output Mean", color="green")
        plt.fill_between(
            steps,
            cfg_means - cfg_stds,
            cfg_means + cfg_stds,
            alpha=0.2,
            color="green",
            label="CFG Output ± Std",
        )

        # Plot conditional output norms if available
        if has_cond:
            cond_means = np.array(cond_means)
            cond_stds = np.array(cond_stds)
            plt.plot(steps, cond_means, linewidth=2, label="Cond Output Mean", color="blue")
            plt.fill_between(
                steps,
                cond_means - cond_stds,
                cond_means + cond_stds,
                alpha=0.2,
                color="blue",
                label="Cond Output ± Std",
            )

        # Plot unconditional output norms if available
        if has_uncond:
            uncond_means = np.array(uncond_means)
            uncond_stds = np.array(uncond_stds)
            plt.plot(steps, uncond_means, linewidth=2, label="Uncond Output Mean", color="orange")
            plt.fill_between(
                steps,
                uncond_means - uncond_stds,
                uncond_means + uncond_stds,
                alpha=0.2,
                color="orange",
                label="Uncond Output ± Std",
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

        def compute_stats(norm_list):
            means, stds = [], []
            for step_norms in norm_list:
                if len(step_norms) > 0:
                    means.append(np.mean(step_norms))
                    stds.append(np.std(step_norms))
                else:
                    means.append(0.0)
                    stds.append(0.0)
            return means, stds

        # Compute stats for CFG-applied output
        cfg_means, cfg_stds = compute_stats(self.gradient_norms)

        # Compute stats for cond/uncond outputs
        has_cond = any(len(s) > 0 for s in self.gradient_norms_cond)
        has_uncond = any(len(s) > 0 for s in self.gradient_norms_uncond)
        cond_means, cond_stds = compute_stats(self.gradient_norms_cond) if has_cond else ([], [])
        uncond_means, uncond_stds = compute_stats(self.gradient_norms_uncond) if has_uncond else ([], [])

        # Create a table for gradient norms vs sampling steps
        data = []
        for step in range(len(cfg_means)):
            row = [
                step,
                cfg_means[step],
                cfg_stds[step],
                cfg_means[step] + cfg_stds[step],
                cfg_means[step] - cfg_stds[step],
            ]
            if has_cond:
                row.extend(
                    [
                        cond_means[step],
                        cond_stds[step],
                        cond_means[step] + cond_stds[step],
                        cond_means[step] - cond_stds[step],
                    ]
                )
            if has_uncond:
                row.extend(
                    [
                        uncond_means[step],
                        uncond_stds[step],
                        uncond_means[step] + uncond_stds[step],
                        uncond_means[step] - uncond_stds[step],
                    ]
                )
            data.append(row)

        columns = [
            "sampling_step",
            "cfg_output_norm_mean",
            "cfg_output_norm_std",
            "cfg_output_norm_upper",
            "cfg_output_norm_lower",
        ]
        if has_cond:
            columns.extend(
                [
                    "cond_output_norm_mean",
                    "cond_output_norm_std",
                    "cond_output_norm_upper",
                    "cond_output_norm_lower",
                ]
            )
        if has_uncond:
            columns.extend(
                [
                    "uncond_output_norm_mean",
                    "uncond_output_norm_std",
                    "uncond_output_norm_upper",
                    "uncond_output_norm_lower",
                ]
            )

        table = wandb_module.Table(columns=columns, data=data)
        wandb_module.log({"gradient_norms": table}, step=train_step)
