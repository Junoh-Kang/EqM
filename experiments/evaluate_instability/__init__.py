"""
Instability evaluation experiment package.

The module structure is intentionally split across small files so each piece
can remain testable and reusable:

- data: dataset + latent caching helpers
- field: thin wrapper around the EqM vector field
- metrics: local instability metrics
- runner: experiment orchestration utilities
- cli: user-facing entry point
"""

from . import metrics, runner
from .data import LatentBatch, LatentCache, build_latent_cache
from .field import EqMField

__all__ = [
    "EqMField",
    "LatentBatch",
    "LatentCache",
    "build_latent_cache",
    "metrics",
    "runner",
]
