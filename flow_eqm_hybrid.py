# ruff: noqa: E402

"""
Hybrid flow matching and EqM sampling.
"""

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import argparse
import math
import os
import sys
from datetime import datetime

from diffusers.models import AutoencoderKL
from PIL import Image

from download import find_model
from models import EqM_models
from transport import Sampler, create_transport
from utils.arg_utils import parse_ode_args, parse_sde_args, parse_transport_args
from utils.sampling_utils import IntermediateImageSaver, decode_latents, sample_eqm


def main(mode, args):
    # Validate mutually exclusive arguments
    if args.class_labels and args.num_samples:
        print("Error: --class-labels and --num-samples cannot be provided simultaneously")
        sys.exit(1)
    if not args.class_labels and not args.num_samples:
        print("Error: Either --class-labels or --num-samples must be provided")
        sys.exit(1)

    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Load model
    latent_size = args.image_size // 8
    model = EqM_models[args.model](input_size=latent_size, num_classes=args.num_classes, uncond=True, ebm="none").to(
        device
    )
    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    if "model" in state_dict.keys():
        model.load_state_dict(state_dict["ema"])
    else:
        model.load_state_dict(state_dict)
    model.eval()

    # Setup flow matching sampler
    transport = create_transport(args.path_type, args.prediction, args.loss_weight, args.train_eps, args.sample_eps)
    sampler = Sampler(transport)
    if mode == "ODE":
        if args.likelihood:
            assert args.cfg_scale == 1, "Likelihood is incompatible with guidance"
            sample_fn = sampler.sample_ode_likelihood(
                sampling_method=args.sampling_method,
                num_steps=args.fm_num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
            )
        else:
            sample_fn = sampler.sample_ode(
                sampling_method=args.sampling_method,
                num_steps=args.fm_num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
                reverse=args.reverse,
            )

    elif mode == "SDE":
        sample_fn = sampler.sample_sde(
            sampling_method=args.sampling_method,
            diffusion_form=args.diffusion_form,
            diffusion_norm=args.diffusion_norm,
            last_step=args.last_step,
            last_step_size=args.last_step_size,
            num_steps=args.fm_num_sampling_steps,
        )

    # Prepare class labels based on mode
    if args.class_labels:
        # Parse provided class labels
        class_labels = [int(c.strip()) for c in args.class_labels.split(",")]
        all_class_labels = torch.tensor(class_labels, device=device)
        num_samples = len(class_labels)
    else:
        # Generate random class labels
        num_samples = args.num_samples
        all_class_labels = torch.randint(0, 1000, (num_samples,), device=device)

    # Calculate number of batches
    batch_size = args.batch_size
    num_batches = math.ceil(num_samples / batch_size)

    # Parse save-steps argument
    fm_save_steps_list = []
    if args.fm_save_steps is not None:
        fm_save_steps_list = [int(s.strip()) for s in args.fm_save_steps.split(",")]
        if args.fm_num_sampling_steps not in fm_save_steps_list:
            # Always save the final step
            fm_save_steps_list.append(args.fm_num_sampling_steps)

    eqm_save_steps_list = []
    if args.eqm_save_steps is not None:
        eqm_save_steps_list = [int(s.strip()) for s in args.eqm_save_steps.split(",")]
        if args.eqm_num_sampling_steps not in eqm_save_steps_list:
            # Always save the final step
            eqm_save_steps_list.append(args.eqm_num_sampling_steps)

    # Create output directory
    output_dir = build_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)

    # Save class labels
    class_labels_path = f"{output_dir}/class_labels.txt"
    with open(class_labels_path, "w") as f:
        for label in all_class_labels.cpu().tolist():
            f.write(f"{label}\n")
    print(f"Saved class labels to {class_labels_path}")

    # Initialize hooks for EqM sampling (shared across batches)
    eqm_hooks_by_fm_step = {}
    for flow_step_idx in fm_save_steps_list:
        step_folder = f"{output_dir}/fm_step_{flow_step_idx:03d}"
        img_saver = IntermediateImageSaver(eqm_save_steps_list, step_folder)
        eqm_hooks_by_fm_step[flow_step_idx] = [img_saver]

    print(f"Starting batch processing: {num_batches} batches of size {batch_size}...")
    # Process samples in batches
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_samples)
        current_batch_size = batch_end - batch_start

        print(f"\nBatch {batch_idx + 1}/{num_batches}: samples {batch_start} to {batch_end - 1}")

        # Get class labels for this batch
        batch_class_labels = all_class_labels[batch_start:batch_end]

        # Generate random noise for this batch
        z = torch.randn(current_batch_size, 4, latent_size, latent_size, device=device)
        y = batch_class_labels

        # Setup CFG
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * current_batch_size, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = {"y": y, "cfg_scale": args.cfg_scale}

        print("  Starting flow matching sampling...")
        all_samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)  # (num_steps, batch*2, C, H, W)
        print("  Flow matching sampling done!")

        # EqM sampling from specified flow matching steps
        for flow_step_idx in fm_save_steps_list:
            print(f"  Starting EqM sampling from flow matching step {flow_step_idx}...")

            step_folder = f"{output_dir}/fm_step_{flow_step_idx:03d}"
            os.makedirs(f"{step_folder}/step_000", exist_ok=True)

            # Get flow matching latents at this step
            flow_matching_latents = all_samples[flow_step_idx - 1] if flow_step_idx > 0 else z
            flow_matching_latents, _ = flow_matching_latents.chunk(2, dim=0)  # Remove null class samples (CFG)

            # Save flow matching samples
            flow_matching_samples = decode_latents(vae, flow_matching_latents)
            for i_sample, sample in enumerate(flow_matching_samples):
                global_idx = batch_start + i_sample
                Image.fromarray(sample).save(f"{step_folder}/step_000/{global_idx:06d}.png")

            # Run EqM sampling from intermediate samples
            _eqm_samples = sample_eqm(
                model=model,
                vae=vae,
                device=device,
                batch_size=current_batch_size,
                latent_size=latent_size,
                initial_latent=flow_matching_latents,
                class_labels=batch_class_labels,
                num_sampling_steps=args.eqm_num_sampling_steps,
                stepsize=args.eqm_stepsize,
                cfg_scale=args.cfg_scale,
                sampler=args.eqm_sampler,
                mu=args.eqm_mu,
                hooks=eqm_hooks_by_fm_step[flow_step_idx],
            )
            print(f"  EqM sampling from flow matching step {flow_step_idx} done!")

    print(f"\nDone! Saved {num_samples} samples to {output_dir}/")


def build_output_dir(args):
    """
    Build a descriptive subdirectory name from key argument/value pairs.
    """
    components = [datetime.now().strftime("%Y%m%d-%H%M%S")]

    # Parse save-steps argument
    fm_save_steps_list = []
    if args.fm_save_steps is not None:
        fm_save_steps_list = [int(s.strip()) for s in args.fm_save_steps.split(",")]
        if args.fm_num_sampling_steps not in fm_save_steps_list:
            fm_save_steps_list.append(args.fm_num_sampling_steps)

    eqm_save_steps_list = []
    if args.eqm_save_steps is not None:
        eqm_save_steps_list = [int(s.strip()) for s in args.eqm_save_steps.split(",")]
        if args.eqm_num_sampling_steps not in eqm_save_steps_list:
            eqm_save_steps_list.append(args.eqm_num_sampling_steps)

    for key, value in [
        ("num_samples", args.num_samples if args.num_samples else None),
        ("fm", "-".join([str(s) for s in fm_save_steps_list])),
        ("eqm", "-".join([str(s) for s in eqm_save_steps_list])),
        ("eqm_stepsize", args.eqm_stepsize),
    ]:
        if value is None:
            continue
        safe_value = str(value).replace("/", "-").replace(" ", "")
        components.append(f"{key}-{safe_value}")
    return os.path.join(args.output_dir, "__".join(components))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    if len(sys.argv) < 2:
        print("Usage: program.py <mode> [options]")
        sys.exit(1)

    mode = sys.argv[1]

    assert mode[:2] != "--", "Usage: program.py <mode> [options]"
    assert mode in ["ODE", "SDE"], "Invalid mode. Please choose 'ODE' or 'SDE'"

    parser.add_argument("--model", type=str, choices=list(EqM_models.keys()), default="EqM-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Optional path to a EqM checkpoint (default: auto-download a pre-trained EqM-XL/2 model).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/debug",
        help="Directory to save intermediate samples (default: samples)",
    )
    parser.add_argument(
        "--class-labels",
        type=str,
        default=None,
        help="Comma-separated list of class labels to sample (mutually exclusive with --num-samples)",
    )
    parser.add_argument("--fm-num-sampling-steps", type=int, default=250)
    parser.add_argument(
        "--fm-save-steps",
        type=str,
        default=None,
        help="Comma-separated list of sampling steps to save intermediate images (e.g., '10,20,30')",
    )
    parser.add_argument("--eqm-num-sampling-steps", type=int, default=250)
    parser.add_argument(
        "--eqm-save-steps",
        type=str,
        default=None,
        help="Comma-separated list of sampling steps to save intermediate images (e.g., '10,20,30')",
    )
    parser.add_argument(
        "--eqm-stepsize", type=float, default=0.0017, help="Step size eta for eqm sampling (default: 0.0017)"
    )
    parser.add_argument(
        "--eqm-sampler",
        type=str,
        default="gd",
        choices=["gd", "ngd"],
        help="Sampler type: 'gd' (gradient descent) or 'ngd' (NAG-GD) (default: 'gd')",
    )
    parser.add_argument("--eqm-mu", type=float, default=0.3, help="NAG-GD momentum hyperparameter mu (default: 0.3)")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Total number of samples to generate with random class labels (mutually exclusive with --class-labels)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Batch size for processing samples",
    )

    parse_transport_args(parser)
    if mode == "ODE":
        parse_ode_args(parser)
        # Further processing for ODE
    elif mode == "SDE":
        parse_sde_args(parser)
        # Further processing for SDE

    args = parser.parse_known_args()[0]
    main(mode, args)
