# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A single-GPU sampling script for EqM with intermediate image logging.
"""

import torch

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse
import os

from diffusers.models import AutoencoderKL
from PIL import Image
from tqdm import tqdm

from download import find_model
from models import EqM_models


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def decode_latents(vae, latents):
    """
    Decode latent tensors to images.

    Args:
        vae: VAE decoder
        latents: Latent tensor of shape (batch_size, 4, H, W)

    Returns:
        numpy array of shape (batch_size, H, W, 3), dtype uint8
    """
    with torch.no_grad():
        samples = vae.decode(latents / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255)
        samples = samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
    return samples


def save_image(image_array, path):
    """
    Save a single image array to disk.

    Args:
        image_array: numpy array of shape (H, W, 3), dtype uint8
        path: Path to save the image
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(image_array).save(path)


def main(args):
    """
    Single-GPU sampling script with intermediate image logging.
    """
    assert torch.cuda.is_available(), "Sampling currently requires at least one GPU."

    # Setup device
    device = torch.device("cuda:0")
    torch.manual_seed(args.global_seed)
    torch.cuda.set_device(device)
    print(f"Using device: {device}")

    # Disable flash for energy training
    if args.ebm != "none":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = EqM_models[args.model](
        input_size=latent_size, num_classes=args.num_classes, uncond=args.uncond, ebm=args.ebm
    ).to(device)

    # Load checkpoint (use EMA weights for better sample quality)
    if args.ckpt is not None:
        ckpt_path = args.ckpt
        state_dict = find_model(ckpt_path)
        if "model" in state_dict.keys():
            # Checkpoint has both 'model' and 'ema' keys - use EMA for sampling
            model.load_state_dict(state_dict["ema"])
            print(f"Loaded EMA weights from checkpoint: {ckpt_path}")
        else:
            # Raw state dict without keys
            model.load_state_dict(state_dict)
            print(f"Loaded model from checkpoint: {ckpt_path}")
    else:
        print("Warning: No checkpoint specified, using randomly initialized model")

    requires_grad(model, False)
    model.eval()

    # Load VAE
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    print(f"EqM Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Parse class labels
    if args.class_labels is not None:
        # Parse comma-separated list or single class
        class_ids = [int(c.strip()) for c in args.class_labels.split(",")]
        # Generate num_samples for each class
        class_labels = []
        for class_id in class_ids:
            class_labels.extend([class_id] * args.num_samples)
        print(
            f"Sampling {len(class_ids)} class(es) {class_ids}, {args.num_samples} samples each (total: {len(class_labels)} samples)"
        )
    else:
        # Default: random classes
        class_labels = [None] * args.num_samples
        print(f"Sampling {args.num_samples} samples with random classes")

    # Create output folder
    os.makedirs(args.folder, exist_ok=True)
    print(f"Output folder: {args.folder}")
    if args.save_interval is not None:
        print(f"Saving intermediate images every {args.save_interval} steps")

    # Setup classifier-free guidance
    use_cfg = args.cfg_scale > 1.0
    if use_cfg:
        print(f"Using classifier-free guidance with scale: {args.cfg_scale}")

    # Sample each image
    total_samples = len(class_labels)
    actual_labels = []  # Track actual class labels used for each sample
    for sample_idx in tqdm(range(total_samples), desc="Generating samples"):
        # Get class label for this sample
        if class_labels[sample_idx] is None:
            y = torch.randint(0, args.num_classes, (1,), device=device)
        else:
            y = torch.tensor([class_labels[sample_idx]], device=device)

        # Store the actual class label used
        actual_labels.append(y.item())

        # Create initial noise
        z = torch.randn(1, 4, latent_size, latent_size, device=device)
        t = torch.ones((1,), device=device)

        # Setup CFG
        if use_cfg:
            z_combined = torch.cat([z, z], 0)
            y_null = torch.tensor([1000], device=device)  # Unconditional class
            y_combined = torch.cat([y, y_null], 0)
            t_combined = torch.cat([t, t], 0)
            model_fn = model.forward_with_cfg
        else:
            z_combined = z
            y_combined = y
            t_combined = t
            model_fn = model.forward

        # Initialize sampling variables
        xt = z_combined
        m = torch.zeros_like(xt).to(device)

        # Create sample-specific directory
        sample_dir = os.path.join(args.folder, f"sample_{sample_idx:03d}")
        os.makedirs(sample_dir, exist_ok=True)

        # Sampling loop
        with torch.no_grad():
            for step_idx in range(args.num_sampling_steps - 1):
                # Compute gradient/output
                if args.sampler == "gd":
                    out = model_fn(xt, t_combined, y_combined, args.cfg_scale)
                    if not torch.is_tensor(out):
                        out = out[0]
                elif args.sampler == "ngd":
                    x_ = xt + args.stepsize * m * args.mu
                    out = model_fn(x_, t_combined, y_combined, args.cfg_scale)
                    if not torch.is_tensor(out):
                        out = out[0]
                    m = out

                # Update latent
                xt = xt + out * args.stepsize
                t_combined += args.stepsize

                # Save intermediate image if at interval
                if args.save_interval is not None and (step_idx + 1) % args.save_interval == 0:
                    # Extract conditional part only if using CFG
                    if use_cfg:
                        xt_for_decode = xt[:1]
                    else:
                        xt_for_decode = xt

                    # Decode and save
                    intermediate_image = decode_latents(vae, xt_for_decode)[0]
                    save_path = os.path.join(sample_dir, f"step_{step_idx + 1:03d}.png")
                    save_image(intermediate_image, save_path)

            # Save final image
            if use_cfg:
                xt_final = xt[:1]
            else:
                xt_final = xt

            final_image = decode_latents(vae, xt_final)[0]
            save_path = os.path.join(sample_dir, "final.png")
            save_image(final_image, save_path)

            # Also save to root folder for easy access
            save_path_root = os.path.join(args.folder, f"{sample_idx:06d}.png")
            save_image(final_image, save_path_root)

    # Save class labels to text file
    labels_path = os.path.join(args.folder, "labels.txt")
    with open(labels_path, "w") as f:
        for label in actual_labels:
            f.write(f"{label}\n")

    print(f"\nDone! Generated {total_samples} samples in {args.folder}")
    print(f"Final images: {args.folder}/######.png")
    print(f"Class labels: {labels_path}")
    if args.save_interval is not None:
        print(f"Intermediate images: {args.folder}/sample_###/step_###.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(EqM_models.keys()), default="EqM-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--ckpt", type=str, default=None, help="Path to a EqM checkpoint")
    parser.add_argument("--stepsize", type=float, default=0.0017, help="step size eta")
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--folder", type=str, default="samples_single")
    parser.add_argument("--sampler", type=str, default="gd", choices=["gd", "ngd"])
    parser.add_argument("--mu", type=float, default=0.3, help="NAG-GD hyperparameter mu")
    parser.add_argument("--uncond", type=bool, default=True, help="disable/enable noise conditioning")
    parser.add_argument(
        "--ebm", type=str, choices=["none", "l2", "dot", "mean"], default="none", help="energy formulation"
    )

    # New arguments for class labels
    parser.add_argument(
        "--class-labels",
        type=str,
        default=None,
        help="Class labels to sample (single or comma-separated, e.g., '207' or '207,360,388'). If not specified, samples random classes.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=1, help="Number of samples to generate per class (default: 1)"
    )

    # New argument for intermediate image saving
    parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="Save intermediate images every N steps (optional, if not specified, intermediate images won't be saved)",
    )

    args = parser.parse_args()

    main(args)
