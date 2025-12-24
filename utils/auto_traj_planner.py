import itertools
from typing import List, Tuple

import numpy as np
import torch

from utils.pvd_utils import generate_traj_txt


def _clamp_within(value: float, limit: float) -> float:
    """Clamp value into [-limit, limit]."""
    return float(max(min(value, limit), -limit))


def _unique_in_order(values: List[float]) -> List[float]:
    seen = set()
    ordered = []
    for v in values:
        if v not in seen:
            ordered.append(v)
            seen.add(v)
    return ordered


def _neighbor_values(current: float, step: float, limit: float, window: int = 2) -> List[float]:
    """Generate nearby candidates around the current value."""
    if step <= 0:
        step = 1e-3
    deltas = [i * step for i in range(-window, window + 1)]
    values = [_clamp_within(current + d, limit) for d in deltas]
    return _unique_in_order(values)


def _score_viewmask(viewmask: torch.Tensor, mode: str) -> float:
    if viewmask is None:
        return float("inf")
    mode = (mode or "min_visible").lower()
    # Take the last frame mask for scoring.
    mask_frame = viewmask[-1]
    if torch.is_tensor(mask_frame):
        visible = float(mask_frame.sum().item())
    else:
        visible = float(np.asarray(mask_frame).sum())
    if mode == "max_visible":
        return -visible
    # default: min_visible
    return visible


def _evaluate_candidate(
    prev_vals: Tuple[float, float, float],
    candidate: Tuple[float, float, float],
    c2ws_anchor,
    H: int,
    W: int,
    focals,
    principal_points,
    pcd,
    imgs,
    masks,
    opts,
    viewcrafter,
) -> float:
    prev_phi, prev_theta, prev_r = prev_vals
    cand_phi, cand_theta, cand_r = candidate
    traj_phi = [prev_phi, cand_phi]
    traj_theta = [prev_theta, cand_theta]
    traj_r = [prev_r, cand_r]
    with torch.no_grad():
        camera_traj, num_views = generate_traj_txt(
            c2ws_anchor,
            H,
            W,
            focals,
            principal_points,
            traj_phi,
            traj_theta,
            traj_r,
            frame=2,
            device=viewcrafter.device,
        )
        _, viewmask, _ = viewcrafter.run_render(
            [pcd], [imgs], masks, H, W, camera_traj, num_views, nbv=True
        )

    mask_score = _score_viewmask(viewmask, getattr(opts, "planner_score", "min_visible"))
    smooth_lambda = float(getattr(opts, "planner_smooth_lambda", 0.0))
    smooth_penalty = smooth_lambda * (
        abs(cand_phi - prev_phi) + abs(cand_theta - prev_theta) + abs(cand_r - prev_r)
    )
    return mask_score + smooth_penalty


def _generate_candidates(
    current_phi: float,
    current_theta: float,
    current_r: float,
    phi_step: float,
    theta_step: float,
    r_step: float,
    phi_max: float,
    theta_max: float,
    r_max: float,
    max_candidates: int = 30,
) -> List[Tuple[float, float, float]]:
    phi_vals = _neighbor_values(current_phi, phi_step, phi_max)
    theta_vals = _neighbor_values(current_theta, theta_step, theta_max)
    r_vals = _neighbor_values(current_r, r_step, r_max)
    candidates = list(itertools.product(phi_vals, theta_vals, r_vals))
    # Prioritize candidates closer to the current position to keep moves smooth.
    def _candidate_key(vals: Tuple[float, float, float]) -> float:
        phi_v, theta_v, r_v = vals
        return (
            abs(phi_v - current_phi) / max(phi_step, 1e-3)
            + abs(theta_v - current_theta) / max(theta_step, 1e-3)
            + abs(r_v - current_r) / max(r_step, 1e-3)
        )

    candidates.sort(key=_candidate_key)
    return candidates[:max_candidates] if max_candidates > 0 else candidates


def write_traj_txt(path: str, phi_seq: List[float], theta_seq: List[float], r_seq: List[float]) -> None:
    with open(path, "w") as f:
        f.write(" ".join(f"{v:.6g}" for v in phi_seq) + "\n")
        f.write(" ".join(f"{v:.6g}" for v in theta_seq) + "\n")
        f.write(" ".join(f"{v:.6g}" for v in r_seq) + "\n")


def plan_traj_sequences(
    c2ws_anchor,
    pcd,
    imgs,
    masks,
    H,
    W,
    focals,
    principal_points,
    opts,
    viewcrafter,
):
    """
    Plan keyframe sequences for d_phi/d_theta/d_r starting from zero.

    Returns:
        phi_seq, theta_seq, r_seq: Lists of floats starting with 0.
    """
    keyframes = int(getattr(opts, "planner_keyframes", 7))
    keyframes = max(2, min(keyframes, 25))

    phi_max = float(getattr(opts, "planner_phi_max", 45.0))
    theta_max = float(getattr(opts, "planner_theta_max", 30.0))
    r_max = float(getattr(opts, "planner_r_max", 0.2))

    phi_step = float(getattr(opts, "planner_phi_step", 5.0))
    theta_step = float(getattr(opts, "planner_theta_step", 5.0))
    r_step = float(getattr(opts, "planner_r_step", 0.05))

    phi_seq: List[float] = [0.0]
    theta_seq: List[float] = [0.0]
    r_seq: List[float] = [0.0]

    for _ in range(1, keyframes):
        current = (phi_seq[-1], theta_seq[-1], r_seq[-1])
        candidates = _generate_candidates(
            current[0],
            current[1],
            current[2],
            phi_step,
            theta_step,
            r_step,
            phi_max,
            theta_max,
            r_max,
        )

        best_vals = None
        best_score = None
        for cand in candidates:
            score = _evaluate_candidate(
                current,
                cand,
                c2ws_anchor,
                H,
                W,
                focals,
                principal_points,
                pcd,
                imgs,
                masks,
                opts,
                viewcrafter,
            )
            if best_score is None or score < best_score:
                best_score = score
                best_vals = cand

        if best_vals is None:
            best_vals = current

        phi_seq.append(float(best_vals[0]))
        theta_seq.append(float(best_vals[1]))
        r_seq.append(float(best_vals[2]))

    loop_back = bool(getattr(opts, "planner_loop_back", True))
    if loop_back:
        if len(phi_seq) < 25 and (
            phi_seq[-1] != 0.0 or theta_seq[-1] != 0.0 or r_seq[-1] != 0.0
        ):
            phi_seq.append(0.0)
            theta_seq.append(0.0)
            r_seq.append(0.0)
        elif len(phi_seq) == 25:
            phi_seq[-1], theta_seq[-1], r_seq[-1] = 0.0, 0.0, 0.0

    return phi_seq, theta_seq, r_seq
