# ruff: noqa: E402

"""
Hybrid flow matching and EqM sampling.
"""

import torch

from sampling_utils import IntermediateImageSaver, decode_latents, sample_eqm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import argparse
import os
import sys
from datetime import datetime

from diffusers.models import AutoencoderKL
from PIL import Image

from download import find_model
from models import EqM_models
from train_utils import parse_ode_args, parse_sde_args, parse_transport_args
from transport import Sampler, create_transport


def main(mode, args):
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

    # Sample images
    class_labels = [int(c.strip()) for c in args.class_labels.split(",")]
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = {"y": y, "cfg_scale": args.cfg_scale}
    all_samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)  # (num_steps, batch*2, C, H, W)

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

    # Save intermediate samples at specified steps
    for flow_step_idx in range(0, len(all_samples) + 1):
        # Save this step if it's in the save_steps_list or if it's the final step
        if flow_step_idx in fm_save_steps_list:
            step_folder = f"{output_dir}/fm_step_{flow_step_idx:03d}"
            os.makedirs(f"{step_folder}/step_000", exist_ok=True)

            flow_matching_latents = all_samples[flow_step_idx - 1] if flow_step_idx > 0 else z
            flow_matching_latents, _ = flow_matching_latents.chunk(2, dim=0)  # Remove null class samples (CFG)
            flow_matching_samples = decode_latents(vae, flow_matching_latents)
            for i_sample, sample in enumerate(flow_matching_samples):
                Image.fromarray(sample).save(f"{step_folder}/step_000/{i_sample:06d}.png")

            # Start eqm sampling from intermediate samples
            hooks = []
            img_saver = IntermediateImageSaver(eqm_save_steps_list, step_folder)
            hooks.append(img_saver)

            _eqm_samples = sample_eqm(
                model=model,
                vae=vae,
                device=device,
                batch_size=n,
                latent_size=latent_size,
                initial_latent=flow_matching_latents,
                class_labels=torch.tensor(class_labels, device=device),
                num_sampling_steps=args.eqm_num_sampling_steps,
                stepsize=args.eqm_stepsize,
                cfg_scale=args.cfg_scale,
                sampler=args.eqm_sampler,
                mu=args.eqm_mu,
                hooks=hooks,
            )

    print(f"Done! Saved samples to {output_dir}/")


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
        "--output-dir", type=str, default="samples", help="Directory to save intermediate samples (default: samples)"
    )
    parser.add_argument(
        "--class-labels",
        type=str,
        default="207,360,387,974,88,979,417,279",
        help="Comma-separated list of class labels to sample (default: 207,360,387,974,88,979,417,279)",
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

    parse_transport_args(parser)
    if mode == "ODE":
        parse_ode_args(parser)
        # Further processing for ODE
    elif mode == "SDE":
        parse_sde_args(parser)
        # Further processing for SDE

    args = parser.parse_known_args()[0]
    main(mode, args)
