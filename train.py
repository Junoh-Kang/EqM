# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for EqM using Hugging Face Accelerate.
"""

import torch

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse
import logging
import os
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from time import time

import numpy as np
import wandb
from accelerate import Accelerator
from diffusers.models import AutoencoderKL
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

import utils.wandb_utils as wandb_utils
from download import find_model
from models import EqM_models
from transport import create_transport
from utils.arg_utils import (
    parse_sample_args,
    parse_transport_args,
)
from utils.sampling_utils import (
    GradientNormTracker,
    WandBImageLogger,
    sample_eqm,
)
from utils.utils import imagenet_label_from_idx

try:
    import yaml
except ImportError:
    yaml = None
################################################################################
#                             Training Helper Functions                         #
################################################################################


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if logging_dir:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def save_config_yaml(args, output_path, logger=None):
    """
    Convert the argparse Namespace into a YAML (with plaintext fallback) file.
    """
    config_dict = wandb_utils.namespace_to_dict(args)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if yaml is not None:
        with output_path.open("w") as f:
            yaml.safe_dump(config_dict, f, sort_keys=False)
        message = f"Saved config to {output_path}"
    else:
        with output_path.open("w") as f:
            for key, value in config_dict.items():
                f.write(f"{key}: {value}\n")
        message = f"Saved config to {output_path} (plain text fallback; install PyYAML for YAML)."

    if logger is not None:
        logger.info(message)


################################################################################
#                                  Training Loop                                #
################################################################################


def main(args):
    """
    Trains a new EqM model.
    """
    # Set up accelerator
    accelerator = Accelerator()
    device = accelerator.device

    assert args.global_batch_size % accelerator.num_processes == 0, "Batch size must be divisible by world size."
    local_batch_size = int(args.global_batch_size // accelerator.num_processes)

    accelerator.print(f"Found {accelerator.num_processes} processes, trying to use device {device}. ")

    rank = accelerator.process_index
    seed = args.global_seed * accelerator.num_processes + rank
    torch.manual_seed(seed)
    accelerator.print(f"Starting rank={rank}, seed={seed}, local_batch_size={local_batch_size}.")

    # disable flash for energy training
    if args.ebm != "none":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(f"_trained/{args.project}", exist_ok=True)  # Make results folder (holds all experiment subfolders)
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_string_name = args.model.replace("/", "-")  # e.g., SiT-XL/2 --> SiT-XL-2 (for naming folders)
        condition = "uncond" if args.uncond else "cond"
        experiment_name = f"{timestamp}-{model_string_name}-{condition}-{args.const_type}"
        if args.adv is not None:
            experiment_name += f"-{args.adv}"
        experiment_dir = f"_trained/{args.project}/{experiment_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        sample_dir = f"{experiment_dir}/samples"
        os.makedirs(sample_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        save_config_yaml(args, f"{experiment_dir}/config.yaml", logger)

        if args.wandb:
            entity = os.environ.get("ENTITY", "junohwandb")
            project = os.environ.get("PROJECT", "EqM-debug")
            wandb_utils.initialize(args, entity, experiment_name, project, dir=f"{experiment_dir}/wandb")
    else:
        logger = create_logger(None)

    # Setup dataloader:
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    dataset = ImageFolder(args.data_path, transform=transform)

    if args.single_class_idx is not None:
        target_idx = args.single_class_idx
        keep_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == target_idx]
        assert len(keep_indices) > 0, f"No samples found for class index {target_idx}; check the dataset paths."
        class_name = dataset.classes[target_idx]
        human_label = imagenet_label_from_idx(target_idx)
        label_desc = f"class idx {target_idx} ({class_name}, {human_label})"
        dataset = Subset(dataset, keep_indices)
        logger.info(f"Filtering dataset to {label_desc} with {len(keep_indices):,} samples.")

    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = EqM_models[args.model](
        input_size=latent_size, num_classes=args.num_classes, uncond=args.uncond, ebm=args.ebm
    ).to(device)

    # Note that parameter initialization is done within the EqM constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.adam_weigth_decay)

    # Resume training state (only if --resume is set)
    resume_step = 0
    start_epoch = 0
    if args.ckpt is not None:
        ckpt_path = args.ckpt
        state_dict = find_model(ckpt_path)
        if "model" in state_dict.keys():
            model.load_state_dict(state_dict["model"])
            ema.load_state_dict(state_dict["ema"])
            opt.load_state_dict(state_dict["opt"])

            # Resume training state only if --resume is set
            if args.resume:
                if "train_steps" in state_dict:
                    resume_step = state_dict["train_steps"]
                    logger.info(f"Resuming training from step {resume_step}")
                else:
                    # Try to parse step from checkpoint filename (e.g., "0050000.pt" -> 50000)
                    filename = os.path.basename(ckpt_path)
                    resume_step = int(filename.replace(".pt", ""))
                    logger.info(f"Resuming training from step {resume_step} (parsed from filename)")

                if "epoch" in state_dict:
                    start_epoch = state_dict["epoch"] + 1
                    logger.info(f"Resuming from epoch {start_epoch}")
                else:
                    steps_per_epoch = len(dataset) // args.global_batch_size
                    start_epoch = resume_step // steps_per_epoch
                    logger.info(
                        f"Calculated start_epoch={start_epoch} from resume_step={resume_step} (steps_per_epoch={steps_per_epoch})"
                    )
        else:
            logger.info("Loading checkpoint weights only (not resuming training state)")
            model.load_state_dict(state_dict)
            ema.load_state_dict(state_dict)

        ema = ema.to(device)
        model = model.to(device)
    requires_grad(ema, False)

    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps,
        args.const_type,
    )  # default: velocity;
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"EqM Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # EMA is initialized with synced weights
    model.train()  # Enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Variables for monitoring/logging purposes:
    train_steps = resume_step
    log_steps = 0
    running_loss = 0
    if args.adv is not None:
        running_loss_adv = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            opt.zero_grad()
            total_loss = 0

            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            model_kwargs = {"y": y, "return_act": args.disp, "train": True}

            loss_dict = transport.training_losses(model, x, model_kwargs)
            loss = loss_dict["loss"].mean()
            total_loss += loss

            if args.adv is not None:
                if args.adv == "consistency":
                    # implement (f(x_fake) = - f(x))
                    adv_kwargs = {"type": "consistency", "stepsize": 0.003}

                loss_dict_adv = transport.adv_training_losses(model, x, model_kwargs, adv_kwargs)
                loss_adv = loss_dict_adv["loss"].mean()
                total_loss += loss_adv

            accelerator.backward(total_loss)
            opt.step()

            update_ema(ema, model.module if hasattr(model, "module") else model)

            # Log loss values:
            running_loss += loss.item()
            if args.adv is not None:
                running_loss_adv += loss_adv.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = accelerator.gather(avg_loss).mean().item()

                if args.adv is not None:
                    avg_loss_adv = torch.tensor(running_loss_adv / log_steps, device=device)
                    avg_loss_adv = accelerator.gather(avg_loss_adv).mean().item()
                    logger.info(
                        f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Adv Loss: {avg_loss_adv:.4f}, Train Steps/Sec: {steps_per_sec:.2f}"
                    )
                else:
                    logger.info(
                        f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}"
                    )

                if args.wandb and accelerator.is_main_process:
                    if args.adv is not None:
                        wandb_utils.log(
                            {"train loss": avg_loss, "adv loss": avg_loss_adv, "train steps per second": steps_per_sec}
                        )
                    else:
                        wandb_utils.log(
                            {"train loss": avg_loss, "train steps per second": steps_per_sec}, step=train_steps
                        )

                # Reset monitoring variables:
                running_loss = 0
                if args.adv is not None:
                    running_loss_adv = 0
                log_steps = 0
                start_time = time()

            # Generate and log samples:
            if train_steps % args.sample_every == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    logger.info("Generating samples for visualization...")
                    # Setup hooks
                    save_steps_list = [args.num_sampling_steps // 2, args.num_sampling_steps]
                    wandb_image_logger = WandBImageLogger(
                        save_steps=save_steps_list,
                        train_step=train_steps,
                        output_folder=sample_dir,
                        wandb_module=wandb if args.wandb else None,
                    )
                    grad_tracker = GradientNormTracker(args.num_sampling_steps)
                    hooks = [wandb_image_logger, grad_tracker]

                    # Fixed initial latent and class labels for consistent comparison
                    latent_size = args.image_size // 8
                    torch.manual_seed(args.global_seed)  # Use same seed for reproducibility
                    initial_latent = torch.randn(args.num_samples, 4, latent_size, latent_size, device=device)
                    class_labels = torch.randint(0, args.num_classes, (args.num_samples,), device=device)
                    if args.single_class_idx is not None:
                        class_labels.fill_(args.single_class_idx)

                    # Generate samples
                    sample_eqm(
                        model=ema,
                        vae=vae,
                        device=device,
                        batch_size=args.num_samples,
                        latent_size=latent_size,
                        initial_latent=initial_latent,
                        class_labels=class_labels,
                        num_sampling_steps=args.num_sampling_steps,
                        stepsize=args.stepsize,
                        cfg_scale=args.cfg_scale,
                        sampler=args.sampler,
                        hooks=hooks,
                    )

                    # Finalize WandB logging
                    wandb_image_logger.finalize()
                    grad_tracker.finalize_for_wandb(
                        wandb_module=wandb if args.wandb else None,
                        train_step=train_steps,
                    )
                    logger.info(f"Samples logged to WandB at step {train_steps}")
                accelerator.wait_for_everyone()

            # Save EqM checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": accelerator.unwrap_model(model).state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                # FIXME: 나중에 accelerate으로 저장하는게 더 좋을 수 있음.
                # save_path = f"{checkpoint_dir}/step_{train_steps:07d}"
                # accelerator.save_state(save_path)
                # accelerator.wait_for_everyone()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")


if __name__ == "__main__":
    # Default args here will train EqM-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()

    # Dataset argument
    group = parser.add_argument_group("Dataset argument")
    group.add_argument("--data-path", type=str, required=True)
    group.add_argument("--single-class-idx", type=int, default=None)
    group.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    group.add_argument("--num-classes", type=int, default=1000)

    # Model arguments
    group = parser.add_argument_group("Model arguments")
    group.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    group.add_argument("--model", type=str, choices=list(EqM_models.keys()), default="EqM-XL/2")
    group.add_argument(
        "--uncond",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="disable/enable noise conditioning",
    )
    group.add_argument(
        "--ebm", type=str, choices=["none", "l2", "dot", "mean"], default="none", help="energy formulation"
    )

    # Training arguments
    group = parser.add_argument_group("Training arguments")
    group.add_argument("--epochs", type=int, default=80)
    group.add_argument("--global-batch-size", type=int, default=256)
    group.add_argument("--global-seed", type=int, default=0)
    group.add_argument("--lr", type=float, default=1e-4)
    group.add_argument("--adam-weigth-decay", type=float, default=0)
    group.add_argument("--ckpt", type=str, default=None, help="Optional path to a custom EqM checkpoint")
    group.add_argument("--resume", action="store_true", help="Toggle to enable resume")
    group.add_argument("--disp", action="store_true", help="Toggle to enable Dispersive Loss")
    group.add_argument("--adv", type=str, help="Adversarial Loss Types")

    parse_transport_args(parser)
    parse_sample_args(parser)

    # Logging arguments
    group = parser.add_argument_group("Logging arguments")
    group.add_argument("--project", type=str, default="exp")
    group.add_argument("--log-every", type=int, default=100)
    group.add_argument("--sample-every", type=int, default=10000)
    group.add_argument("--ckpt-every", type=int, default=10000)
    group.add_argument("--wandb", action="store_true")
    group.add_argument("--timestep-analysis", action="store_true")

    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()

    main(args)
