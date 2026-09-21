"""Geometry-verified evidence routing utilities.

The functions in this module deliberately operate on the geometric evidence
already produced by the renderer/warp stage.  They do not alter the input
scaffold.  Instead, they return a compact state that can be consumed by the
diffusion sampler to decide how strongly evidence should affect denoising.
"""

from __future__ import annotations

from typing import Dict

import torch


EVIDENCE_ABSTAIN = 0
EVIDENCE_ATTENUATE = 1
EVIDENCE_ADMIT = 2


def _as_map(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Normalize a map to ``[batch, 1, height, width]``."""
    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    elif tensor.ndim == 4 and tensor.shape[-1] == 1 and tensor.shape[1] != 1:
        tensor = tensor.permute(0, 3, 1, 2)
    if tensor.ndim != 4:
        raise ValueError(f"{name} must have shape [B, 1, H, W] or [B, H, W]")
    if tensor.shape[1] != 1:
        raise ValueError(f"{name} must have one channel, got {tensor.shape}")
    return tensor


def compute_geometry_evidence(
    warped_depths: torch.Tensor,
    warped_masks: torch.Tensor,
    target_depth: torch.Tensor,
    target_mask: torch.Tensor,
    depth_rel_tolerance: float = 0.08,
    depth_abs_tolerance: float = 0.02,
    conflict_threshold: float = 0.08,
    min_support_ratio: float = 0.02,
    admit_ratio_threshold: float = 0.20,
    max_conflict_ratio: float = 0.15,
    attenuated_weight: float = 0.35,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Classify warped multi-view geometry into admit/attenuate/abstain.

    Args:
        warped_depths: Per-reference target-camera depths with shape
            ``[K, B, 1, H, W]`` or ``[K, B, H, W]``.
        warped_masks: Visibility/support masks with the same leading shape.
        target_depth: The target-view scaffold depth ``[B, 1, H, W]``.
        target_mask: Valid target scaffold pixels ``[B, 1, H, W]``.

    The test combines three inexpensive signals: common support between a
    warped reference and the target scaffold, front-to-back visibility, and
    multi-reference depth agreement.  A conflict is kept separate from a
    missing support region, which is important for the abstention semantics.

    Returns:
        A dictionary containing soft geometry masks, diagnostic ratios, a
        per-reference routing gate, and an integer state per batch item:
        ``0=abstain``, ``1=attenuate``, ``2=admit``.
    """
    if not torch.is_tensor(warped_depths):
        warped_depths = torch.as_tensor(warped_depths)
    if not torch.is_tensor(warped_masks):
        warped_masks = torch.as_tensor(warped_masks)

    if warped_depths.ndim == 4:
        warped_depths = warped_depths.unsqueeze(2)
    if warped_masks.ndim == 4:
        warped_masks = warped_masks.unsqueeze(2)
    if warped_depths.ndim != 5 or warped_masks.ndim != 5:
        raise ValueError("warped maps must have shape [K, B, 1, H, W]")
    if warped_depths.shape[0] == 0:
        raise ValueError("at least one warped reference is required")
    if warped_depths.shape != warped_masks.shape:
        raise ValueError("warped_depths and warped_masks must have identical shapes")

    target_depth = _as_map(target_depth, "target_depth").float()
    target_mask = _as_map(target_mask, "target_mask").float()
    warped_depths = warped_depths.to(device=target_depth.device, dtype=torch.float32)
    warped_masks = warped_masks.to(device=target_depth.device, dtype=torch.float32)

    if warped_depths.shape[1:] != target_depth.shape:
        raise ValueError(
            "warped maps and target maps disagree: "
            f"{warped_depths.shape[1:]} vs {target_depth.shape}"
        )

    device = target_depth.device
    target_valid = (
        target_mask > 0.5
    ) & torch.isfinite(target_depth) & (target_depth > eps)
    warped_valid = (
        warped_masks > 0.5
    ) & torch.isfinite(warped_depths) & (warped_depths > eps)

    target_valid_k = target_valid.unsqueeze(0)
    common_support = warped_valid & target_valid_k

    # Sanitize invalid values before arithmetic. Multiplying a NaN by a
    # zero-valued validity mask would still produce NaN and could contaminate
    # the consensus field, even though the corresponding pixel is invalid.
    target_depth_safe = torch.where(
        target_valid, target_depth, torch.ones_like(target_depth)
    )
    warped_depth_safe = torch.where(
        warped_valid, warped_depths, torch.zeros_like(warped_depths)
    )
    target_depth_k = target_depth_safe.unsqueeze(0)

    relative_error = (warped_depth_safe - target_depth_k).abs() / target_depth_k.abs().clamp_min(eps)
    visibility_violation = common_support & (
        warped_depth_safe > target_depth_k + depth_abs_tolerance + depth_rel_tolerance * target_depth_k
    )

    tolerance = max(float(depth_rel_tolerance), eps)
    soft_consistency = torch.exp(-relative_error / tolerance) * common_support.float()
    hard_consistency = relative_error <= depth_rel_tolerance
    per_reference_admissible = common_support & ~visibility_violation & hard_consistency

    support_count = common_support.float().sum(dim=0)
    safe_depths = torch.where(common_support, warped_depth_safe, torch.zeros_like(warped_depth_safe))
    mean_depth = safe_depths.sum(dim=0) / support_count.clamp_min(1.0)
    mean_absolute_deviation = (
        (warped_depth_safe - mean_depth.unsqueeze(0)).abs() * common_support.float()
    ).sum(dim=0) / support_count.clamp_min(1.0)
    multi_view_conflict_score = mean_absolute_deviation / mean_depth.abs().clamp_min(eps)
    conflict_mask = (support_count >= 2.0) & (multi_view_conflict_score > conflict_threshold)

    # Keep the spatial score for every reference.  FMI uses this tensor to
    # form a local consensus, while CGA uses its spatially pooled version to
    # route frame-level messages.
    per_reference_evidence = (
        soft_consistency
        * per_reference_admissible.float()
        * (~conflict_mask).float().unsqueeze(0)
    )
    geometry_mask = per_reference_evidence.max(dim=0).values

    target_area = target_valid.float().sum(dim=(1, 2, 3)).clamp_min(1.0)
    support_ratio = (support_count > 0).float().sum(dim=(1, 2, 3)) / target_area
    admissible_ratio = geometry_mask.sum(dim=(1, 2, 3)) / target_area
    conflict_ratio = conflict_mask.float().sum(dim=(1, 2, 3)) / target_area
    free_space_ratio = visibility_violation.float().sum(dim=(2, 3, 4)) / target_area.unsqueeze(0)
    free_space_ratio = free_space_ratio.max(dim=0).values

    has_support = support_ratio >= min_support_ratio
    visibility_risk_ratio = torch.maximum(conflict_ratio, free_space_ratio)
    can_admit = (
        has_support
        & (admissible_ratio >= admit_ratio_threshold)
        & (visibility_risk_ratio <= max_conflict_ratio)
    )
    can_attenuate = has_support & (admissible_ratio > 0.0) & ~can_admit

    state = torch.full(
        (target_depth.shape[0],), EVIDENCE_ABSTAIN, dtype=torch.long, device=device
    )
    state[can_attenuate] = EVIDENCE_ATTENUATE
    state[can_admit] = EVIDENCE_ADMIT

    state_strength = torch.zeros_like(admissible_ratio)
    state_strength[state == EVIDENCE_ATTENUATE] = float(attenuated_weight)
    state_strength[state == EVIDENCE_ADMIT] = 1.0

    # ``geometry_mask`` is the verified, pre-state evidence field.  The
    # routing mask is the actual action consumed by FMI and is therefore
    # scaled by the frame-level state.  Expose a discrete per-region state as
    # well, so downstream diagnostics do not have to infer it from a soft
    # gate.
    routing_mask = geometry_mask * state_strength.view(-1, 1, 1, 1)
    region_state = torch.where(
        routing_mask > eps,
        state.view(-1, 1, 1, 1).expand_as(routing_mask),
        torch.zeros_like(routing_mask, dtype=torch.long),
    )

    per_reference_gate = per_reference_evidence.sum(dim=(2, 3, 4)) / target_area.unsqueeze(0)
    per_reference_gate = per_reference_gate.transpose(0, 1)
    per_reference_gate = per_reference_gate * state_strength.unsqueeze(1)
    frame_gate = per_reference_gate.max(dim=1).values if per_reference_gate.shape[1] else state_strength.new_zeros(state.shape)
    occlusion_mask = visibility_violation.float().max(dim=0).values

    return {
        "support_count": support_count,
        "support_ratio": support_ratio,
        "admissible_ratio": admissible_ratio,
        "conflict_ratio": conflict_ratio,
        "free_space_ratio": free_space_ratio,
        # Legacy free_space fields describe occlusion ordering, not empty space.
        "occlusion_ratio": free_space_ratio,
        "visibility_risk_ratio": visibility_risk_ratio,
        "conflict_mask": conflict_mask.float(),
        "free_space_mask": occlusion_mask,
        "occlusion_mask": occlusion_mask,
        "geometry_mask": geometry_mask,
        "routing_mask": routing_mask,
        "region_state": region_state,
        "per_reference_evidence": per_reference_evidence,
        "per_reference_gate": per_reference_gate,
        "frame_gate": frame_gate,
        "state_strength": state_strength,
        "state": state,
    }
