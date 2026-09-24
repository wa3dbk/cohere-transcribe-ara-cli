"""Vectorised SpecAugment on normalized log-mel features of shape (B, T, F)."""

from __future__ import annotations

import torch


def spec_augment(
    feats: torch.Tensor,
    mask: torch.Tensor,
    num_freq_masks: int = 2,
    freq_width: int = 27,
    num_time_masks: int = 10,
    time_width_ratio: float = 0.05,
    max_time_width: int = 40,
) -> torch.Tensor:
    """Zero out random frequency bands and time spans (0 == per-utterance mean after normalization).

    ``mask`` is the (B, T) frame-validity mask; time masks are placed inside the valid region only.
    """
    b, t, f = feats.shape
    device = feats.device
    out = feats
    if num_freq_masks > 0 and freq_width > 0:
        w = torch.randint(0, freq_width + 1, (b, num_freq_masks), device=device)
        f0 = (torch.rand(b, num_freq_masks, device=device) * (f - w).clamp(min=1)).long()
        ar = torch.arange(f, device=device)[None, None, :]
        fm = ((ar >= f0[..., None]) & (ar < (f0 + w)[..., None])).any(dim=1)  # (B, F)
        out = out.masked_fill(fm[:, None, :], 0.0)
    if num_time_masks > 0 and time_width_ratio > 0:
        lengths = mask.sum(dim=1).long()  # (B,)
        maxw = (lengths.float() * time_width_ratio).clamp(max=max_time_width).long()
        w = (torch.rand(b, num_time_masks, device=device) * (maxw[:, None] + 1).float()).long()
        t0 = (torch.rand(b, num_time_masks, device=device) * (lengths[:, None] - w).clamp(min=1).float()).long()
        ar = torch.arange(t, device=device)[None, None, :]
        tm = ((ar >= t0[..., None]) & (ar < (t0 + w)[..., None])).any(dim=1)  # (B, T)
        out = out.masked_fill(tm[:, :, None], 0.0)
    return out
