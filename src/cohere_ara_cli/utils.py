"""Small distributed helpers."""

from __future__ import annotations

import os

import torch


def accelerate_kwargs() -> dict:
    """Force Accelerate's CPU (gloo) multi-process mode when no CUDA device is present."""
    return {} if torch.cuda.is_available() else {"cpu": True}


def check_world(state) -> None:
    """Fail loudly instead of letting every process silently do the full job."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1 and state.num_processes != world:
        raise RuntimeError(
            f"WORLD_SIZE={world} but the distributed process group was not initialised "
            f"(num_processes={state.num_processes}); check your launcher / accelerate config"
        )
