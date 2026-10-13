"""Probability cap for adaptive reference-phase sampling."""
import torch


def capped_probabilities(weights, cap=.28):
    if weights.ndim != 1 or weights.numel() * cap < 1:
        raise ValueError('Too few phase bins for probability cap')
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Phase weights must be finite and nonnegative')
    positive = weights.clamp(min=1e-8)
    fixed = torch.zeros_like(weights, dtype=torch.bool)
    for _ in range(weights.numel()):
        free_weights = positive.masked_fill(fixed, 0)
        remaining = (1 - fixed.sum() * cap).clamp(min=0)
        candidate = free_weights / free_weights.sum().clamp(min=1e-8) * remaining
        newly_fixed = (~fixed) & (candidate > cap)
        if not newly_fixed.any():
            return torch.where(fixed, torch.full_like(weights, cap), candidate)
        # Saturated bins must never receive redistributed probability again.
        fixed |= newly_fixed
    return torch.full_like(weights, cap)
