"""
Utility script for creating `.npz` latents and class label files consumed by
`sample_eqm.py`. Supports synthetic random latents as well as latents encoded
from the ImageNet validation split using the Stable Diffusion VAE.
"""

import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from utils.sampling_utils import encode_images_to_latent


def _validate_image_size(image_size: int) -> int:
    if image_size <= 0:
        raise ValueError(f"Image size must be positive, got {image_size}.")
    if image_size % 8 != 0:
        raise ValueError("Image size must be divisible by 8 (required by the VAE encoder).")
    return image_size // 8


def _validate_num_samples(num_samples: int) -> None:
    if num_samples <= 0:
        raise ValueError("num_samples must be greater than zero.")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _save_latents(latents: torch.Tensor, output_path: Path) -> Path:
    _ensure_parent(output_path)
    latents_np = latents.cpu().numpy().astype(np.float32)
    np.savez(output_path, arr_0=latents_np)
    print(f"Saved latents to {output_path} with shape {latents_np.shape}.")
    return output_path


def _save_labels(labels: Iterable[int], output_path: Path) -> Path:
    label_list = [int(label) for label in labels]
    _ensure_parent(output_path)
    with open(output_path, "w", encoding="utf-8") as handle:
        for label in label_list:
            handle.write(f"{label}\n")
    print(f"Saved {len(label_list)} class labels to {output_path}.")
    return output_path


def _resolve_output_paths(latents_out: str, labels_out: str) -> Tuple[Path, Path]:
    latents_path = Path(latents_out).expanduser()
    labels_path = Path(labels_out).expanduser()
    return latents_path, labels_path


def _resolve_device(requested: Optional[str]) -> torch.device:
    if requested is not None:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(f"CUDA device '{requested}' requested, but CUDA is not available.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _prepare_random_latents(
    num_samples: int,
    latent_size: int,
    num_classes: int,
    seed: int,
    latents_out: Path,
    labels_out: Path,
) -> Tuple[Path, Path]:
    if num_classes <= 0:
        raise ValueError("num_classes must be greater than zero.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    latents = torch.randn(
        (num_samples, 4, latent_size, latent_size),
        generator=generator,
        dtype=torch.float32,
    )
    class_labels = torch.randint(
        low=0,
        high=num_classes,
        size=(num_samples,),
        generator=generator,
        dtype=torch.long,
    )

    latents_path = _save_latents(latents, latents_out)
    labels_path = _save_labels(class_labels.tolist(), labels_out)
    return latents_path, labels_path


def _prepare_imagenet_latents(
    val_data_path: Path,
    num_samples: int,
    image_size: int,
    batch_size: int,
    num_workers: int,
    vae_variant: str,
    device_str: Optional[str],
    seed: int,
    latents_out: Path,
    labels_out: Path,
) -> Tuple[Path, Path]:
    try:
        from diffusers.models import AutoencoderKL
        from torch.utils.data import DataLoader, Subset
        from torchvision import transforms
        from torchvision.datasets import ImageFolder
    except ImportError as err:
        raise ImportError("ImageNet mode requires diffusers and torchvision to be installed.") from err

    if not val_data_path.exists():
        raise FileNotFoundError(f"ImageNet validation path not found: {val_data_path}")

    _validate_image_size(image_size)
    device = _resolve_device(device_str)

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    dataset = ImageFolder(str(val_data_path), transform=transform)
    dataset_size = len(dataset)
    if num_samples > dataset_size:
        raise ValueError(f"Requested {num_samples} samples, but only {dataset_size} images are available.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator)[:num_samples].tolist()
    subset = Subset(dataset, indices)
    data_loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    print(f"Loading VAE weights 'stabilityai/sd-vae-ft-{vae_variant}' on {device}...")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_variant}").to(device)
    vae.eval()

    encoded_latents = []
    encoded_labels = []

    print(f"Encoding the first {num_samples} ImageNet validation images...")
    for images, labels in tqdm(data_loader, desc="Encoding ImageNet latents"):
        latents = encode_images_to_latent(vae, images, device).to("cpu")
        encoded_latents.append(latents)
        encoded_labels.append(labels)

    latents_tensor = torch.cat(encoded_latents, dim=0)
    labels_tensor = torch.cat(encoded_labels, dim=0)

    assert latents_tensor.shape[0] == num_samples, "Mismatch in encoded latent count."

    latents_path = _save_latents(latents_tensor, latents_out)
    labels_path = _save_labels(labels_tensor.tolist(), labels_out)
    return latents_path, labels_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare initial latents (.npz) and class label files for sample_eqm.py. "
            "Supports random latent generation and ImageNet validation encoding."
        )
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["random", "imagenet"],
        required=True,
        help="Preparation mode: 'random' for synthetic latents or 'imagenet' to encode validation images.",
    )
    parser.add_argument("--num-samples", type=int, required=True, help="Number of samples to prepare.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Image resolution (must be divisible by 8). Determines latent spatial size.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1000,
        help="Number of classes to sample from when generating random labels (ignored for ImageNet mode).",
    )
    parser.add_argument(
        "--latents-out",
        type=str,
        default="initial_latents.npz",
        help="Output path for the saved latents npz file.",
    )
    parser.add_argument(
        "--labels-out",
        type=str,
        default="initial_class_labels.txt",
        help="Output path for the class labels text file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for generating synthetic latents/labels and ImageNet sampling.",
    )
    parser.add_argument(
        "--val-data-path",
        type=str,
        default=None,
        help="Path to the ImageNet validation directory (required for ImageNet mode).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size used while encoding ImageNet images.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of worker processes for the ImageNet dataloader.",
    )
    parser.add_argument(
        "--vae",
        type=str,
        choices=["ema", "mse"],
        default="ema",
        help="Stable Diffusion VAE variant to use when encoding ImageNet latents.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device for VAE encoding (e.g., 'cuda:0' or 'cpu'). Defaults to CUDA if available.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _validate_num_samples(args.num_samples)

    latent_size = _validate_image_size(args.image_size)
    latents_out, labels_out = _resolve_output_paths(args.latents_out, args.labels_out)

    if args.mode == "random":
        _prepare_random_latents(
            num_samples=args.num_samples,
            latent_size=latent_size,
            num_classes=args.num_classes,
            seed=args.seed,
            latents_out=latents_out,
            labels_out=labels_out,
        )
    else:
        if args.val_data_path is None:
            raise ValueError("--val-data-path is required when --mode=imagenet.")
        _prepare_imagenet_latents(
            val_data_path=Path(args.val_data_path).expanduser(),
            num_samples=args.num_samples,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            vae_variant=args.vae,
            device_str=args.device,
            seed=args.seed,
            latents_out=latents_out,
            labels_out=labels_out,
        )

    print("Preparation complete.")


if __name__ == "__main__":
    main()
