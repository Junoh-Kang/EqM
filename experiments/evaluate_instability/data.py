"""
Dataset helpers for the instability evaluation experiment.

The helpers below mirror the logic in `sample_from_clean.py`, but they keep
latents cached on CPU so the expensive VAE encoding step only runs once per
image.  Batches can then be streamed back to any device for metric evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from utils.sampling_utils import encode_images_to_latent


def _build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )


@dataclass
class LatentBatch:
    latents: torch.Tensor
    labels: torch.Tensor
    indices: torch.Tensor


@dataclass
class LatentCache:
    latents: torch.Tensor
    labels: torch.Tensor
    indices: torch.Tensor

    def __len__(self) -> int:
        return int(self.latents.shape[0])

    def iter_batches(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Iterator[LatentBatch]:
        total = len(self)
        num_batches = (total + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, total)
            yield LatentBatch(
                latents=self.latents[start:end].to(device),
                labels=self.labels[start:end].to(device),
                indices=self.indices[start:end],
            )


def _select_indices(num_items: int, num_samples: int | None, seed: int) -> list[int]:
    indices = list(range(num_items))
    if num_samples is not None and num_samples < len(indices):
        rng = torch.Generator()
        rng.manual_seed(seed)
        perm = torch.randperm(len(indices), generator=rng).tolist()
        indices = [indices[i] for i in perm[:num_samples]]
    indices.sort()
    return indices


def build_latent_cache(
    data_path: str,
    vae,
    device: torch.device,
    image_size: int,
    batch_size: int,
    num_samples: int | None,
    seed: int,
    num_workers: int = 4,
) -> LatentCache:
    transform = _build_transform(image_size)
    dataset = ImageFolder(data_path, transform=transform)
    indices = _select_indices(len(dataset), num_samples, seed)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    all_latents: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    print(f"Encoding {len(subset)} images into latent space...")
    for images, labels in loader:
        latents = encode_images_to_latent(vae, images, device)
        all_latents.append(latents.to("cpu", dtype=torch.float32))
        all_labels.append(labels.to("cpu", dtype=torch.long))

    latents_tensor = torch.cat(all_latents, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    index_tensor = torch.tensor(indices, dtype=torch.long)
    print(f"Latent cache built: {latents_tensor.shape[0]} samples, latent shape {latents_tensor.shape[1:]}")
    return LatentCache(latents=latents_tensor, labels=labels_tensor, indices=index_tensor)
