"""Simple test script to verify EqMField matches sample_eqm."""

import numpy as np
import torch

from experiments.evaluate_instability.field import EqMField
from utils import sampling_utils
from utils.sampling_utils import decode_latents


class _DecodeResult:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.sample = tensor


class IdentityVAE:
    def __init__(self, device: torch.device) -> None:
        self.device = device

    def decode(self, latents: torch.Tensor) -> _DecodeResult:
        return _DecodeResult(latents)


class LinearToyModel:
    def __init__(self, step_scale: float = 0.5) -> None:
        self.step_scale = step_scale

    def _base_forward(self, xt: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_term = labels.view(-1, 1, 1, 1).float() * 0.001
        time_term = t.view(-1, 1, 1, 1) * 0.01
        return self.step_scale * xt + label_term + time_term

    def forward(self, xt: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self._base_forward(xt, t, labels)

    def forward_with_cfg(
        self, xt: torch.Tensor, t: torch.Tensor, labels: torch.Tensor, cfg_scale: float
    ) -> torch.Tensor:
        batch = xt.shape[0] // 2
        cond = self._base_forward(xt[:batch], t[:batch], labels[:batch])
        uncond = self._base_forward(xt[batch:], t[batch:], labels[batch:])
        guided = uncond + cfg_scale * (cond - uncond)
        return torch.cat([guided, guided], dim=0)


def test_eqm_field_matches_sample_eqm(sampler: str) -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    vae = IdentityVAE(device)
    model = LinearToyModel()
    cfg_scale = 4.0
    stepsize = 0.05
    mu = 0.4
    num_steps = 5
    batch_size = 2
    latent_size = 2

    initial_latents = torch.randn(batch_size, 4, latent_size, latent_size)
    labels = torch.tensor([5, 7], dtype=torch.long)

    field = EqMField(model=model, device=device, cfg_scale=cfg_scale, stepsize=stepsize, sampler=sampler, mu=mu)
    field_result = field.run_trajectory(initial_latents, labels, num_steps=num_steps, store_history=False)
    field_samples = decode_latents(vae, field_result.final_latents)

    original_tqdm = sampling_utils.tqdm
    sampling_utils.tqdm = lambda iterable, *_, **__: iterable
    try:
        utils_samples = sampling_utils.sample_eqm(
            model=model,
            vae=vae,
            device=device,
            batch_size=batch_size,
            latent_size=latent_size,
            initial_latent=initial_latents,
            class_labels=labels,
            num_sampling_steps=num_steps,
            stepsize=stepsize,
            cfg_scale=cfg_scale,
            sampler=sampler,
            mu=mu,
            hooks=None,
        )
    finally:
        sampling_utils.tqdm = original_tqdm

    np.testing.assert_array_equal(utils_samples, field_samples)


if __name__ == "__main__":
    for sampler in ["gd", "ngd"]:
        print(f"Testing sampler={sampler}...", end=" ")
        test_eqm_field_matches_sample_eqm(sampler)
        print("PASSED")
    print("All tests passed!")
