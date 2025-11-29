# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for EqM using Hugging Face Accelerate.
"""

import csv
import os
import argparse
import logging
from time import time
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from accelerate import Accelerator
from diffusers.models import AutoencoderKL
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt

from models import EqM_models
from download import find_model
from transport import create_transport
from utils.utils import imagenet_label_from_idx
import utils.wandb_utils as wandb_utils
from utils.arg_utils import (
    parse_transport_args,
    parse_sample_args,
)
from utils.sampling_utils import (
    sample_eqm,
    WandBImageLogger,
    GradientNormTracker,
)
from utils.train_utils import TimestepValueLogger

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
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
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
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


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
        message = (
            f"Saved config to {output_path} (plain text fallback; install PyYAML for YAML)."
        )

    if logger is not None:
        logger.info(message)

def write_timestep_summary_csv(summary, output_dir, ckpt_label):
    """
    Save aggregated timestep statistics as a CSV for downstream analysis.
    """
    if not summary:
        return

    os.makedirs(output_dir, exist_ok=True)
    ckpt_value = ckpt_label or "none"

    rows = []
    for metric, ts in summary.items():
        for t_value, stats in ts.items():
            row = {"ckpt": ckpt_value, "metric": metric, "timestep": t_value}
            row.update(stats)
            rows.append(row)

    if not rows:
        return

    rows.sort(key=lambda r: (r["metric"], r["timestep"]))
    stat_keys = sorted(
        {key for row in rows for key in row.keys() if key not in {"ckpt", "metric", "timestep"}}
    )
    fieldnames = ["ckpt", "metric", "timestep", *stat_keys]

    csv_path = os.path.join(output_dir, "timestep_summary.csv")
    file_exists = os.path.exists(csv_path)
    mode = "a" if file_exists else "w"

    with open(csv_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


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

    assert args.global_batch_size % accelerator.num_processes == 0, \
        f"Batch size must be divisible by world size."
    local_batch_size = int(args.global_batch_size // accelerator.num_processes)

    accelerator.print(
        f"Found {accelerator.num_processes} processes, "
        f"trying to use device {device}. "
    )

    rank = accelerator.process_index
    seed = args.global_seed * accelerator.num_processes + rank
    torch.manual_seed(seed)
    accelerator.print(
        f"Starting rank={rank}, seed={seed}, "
        f"local_batch_size={local_batch_size}."
    )

    # disable flash for energy training
    if args.ebm != 'none':
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    experiment_dir = os.path.join("output", args.project)
    if accelerator.is_main_process:
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        # save_config_yaml(args, f"{experiment_dir}/config.yaml", logger)

    # Setup dataloader:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)

    if args.single_class_idx is not None:
        target_idx = args.single_class_idx
        keep_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == target_idx]
        assert len(keep_indices) > 0, f"No samples found for class index {target_idx}; check the dataset paths."
        class_name = dataset.classes[target_idx]
        human_label = imagenet_label_from_idx(target_idx)
        label_desc = f"class idx {target_idx} ({class_name}, {human_label})"
        dataset = Subset(dataset, keep_indices)
        logger.info(  # type: ignore[has-type]
            f"Filtering dataset to {label_desc} with {len(keep_indices):,} samples."
        )

    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")  # type: ignore[has-type]

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = EqM_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        uncond=args.uncond,
        ebm=args.ebm,
    ).to(device)

    # Note that parameter initialization is done within the EqM constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.adam_weigth_decay
    )

    # Resume training state (only if --resume is set)
    resume_step = 0
    start_epoch = 0
    if args.ckpt is not None:
        ckpt_path = args.ckpt
        state_dict = find_model(ckpt_path)
        if 'model' in state_dict.keys():
            model.load_state_dict(state_dict["model"])
            ema.load_state_dict(state_dict["ema"])
            opt.load_state_dict(state_dict["opt"])

            # Resume training state only if --resume is set
            if args.resume:
                if 'train_steps' in state_dict:
                    resume_step = state_dict['train_steps']
                    logger.info(f"Resuming training from step {resume_step}")  # type: ignore[has-type]
                else:
                    # Try to parse step from checkpoint filename (e.g., "0050000.pt" -> 50000)
                    filename = os.path.basename(ckpt_path)
                    resume_step = int(filename.replace('.pt', ''))
                    logger.info(f"Resuming training from step {resume_step} (parsed from filename)")  # type: ignore[has-type]

                if 'epoch' in state_dict:
                    start_epoch = state_dict['epoch'] + 1
                    logger.info(f"Resuming from epoch {start_epoch}")  # type: ignore[has-type]
                else:
                    steps_per_epoch = len(dataset) // args.global_batch_size
                    start_epoch = resume_step // steps_per_epoch
                    logger.info(  # type: ignore[has-type]
                        f"Calculated start_epoch={start_epoch} from resume_step={resume_step} (steps_per_epoch={steps_per_epoch})"
                    )
        else:
            logger.info("Loading checkpoint weights only (not resuming training state)")  # type: ignore[has-type]
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
    logger.info(f"EqM Parameters: {sum(p.numel() for p in model.parameters()):,}")  # type: ignore[has-type]

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # EMA is initialized with synced weights
    model.train()  # Enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Variables for monitoring/logging purposes:
    train_steps = resume_step
    log_steps = 0
    running_loss = 0
    start_time = time()

    timesteplogger = TimestepValueLogger()

    if accelerator.is_main_process:
        with torch.no_grad():
            t_values = [float(t_value) for t_value in args.t_values.split(",")]
            max_samples = args.metric_max_samples

            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                model_kwargs = dict(y=y, return_act=args.disp, train=True)

                for t_value in t_values:
                    t = torch.full((x.shape[0],), t_value, device=x.device, dtype=x.dtype)
                    loss_dict = transport.training_losses(model, x, t, model_kwargs)
                    timesteplogger(loss_dict['target'], loss_dict['pred'], t_value)

                first_t = t_values[0]
                current_count = len(timesteplogger.data['l2_error'].get(first_t, []))
                if current_count >= max_samples:
                    break

        summary = timesteplogger.summary()
        write_timestep_summary_csv(summary, "output", args.ckpt)
        # plot_timestep_summary(summary, experiment_dir)

    accelerator.wait_for_everyone()

    # The remainder of the original training loop has been left commented out,
    # as this script is currently used for analysis only.


if __name__ == "__main__":
    # Default args here will train EqM-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--t-values", type=str, required=True)
    parser.add_argument("--metric-max-samples", type=int, default=1000,
                        help="Maximum number of samples to collect per timestep before plotting.")
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
    group.add_argument("--uncond",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="disable/enable noise conditioning",
    )
    group.add_argument("--ebm", type=str, choices=["none", "l2", "dot", "mean"], default="none",
                        help="energy formulation")

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

    parse_transport_args(parser)
    parse_sample_args(parser)

    # Logging arguments
    group = parser.add_argument_group("Logging arguments")
    group.add_argument("--project", type=str, default="exp")
    group.add_argument("--log-every", type=int, default=100)
    group.add_argument("--sample-every", type=int, default=10000)
    group.add_argument("--ckpt-every", type=int, default=10000)
    group.add_argument("--wandb", action="store_true")

    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()

    main(args)

