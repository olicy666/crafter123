"""Differentiable reconstruction losses that preserve observation provenance."""
import math

import torch
import torch.nn.functional as F

from .uncertainty_guidance import scale_residual


def auxiliary_losses(rendered, generated, references, scales):
    """Return evidence and completion losses for one [1,3,H,W] target.

    References contain detached RGB, support and sigma at evidence resolution.
    The completion mask is observation absence, not estimated generation quality.
    """
    if rendered.shape != generated.shape or rendered.ndim != 4 or rendered.shape[:2] != (1, 3):
        raise ValueError('rendered and generated must share shape [1,3,H,W]')
    if not torch.isfinite(rendered).all() or not torch.isfinite(generated).all():
        raise ValueError('reconstruction images must be finite')
    zero = rendered.sum() * 0
    if not references:
        return zero, (rendered - generated.detach()).abs().mean()
    size = references[0]['rgb'].shape[-2:]
    low = F.interpolate(rendered, size, mode='area')
    numerator = torch.zeros_like(low[:, :1])
    denominator = torch.zeros_like(low[:, :1])
    weights = []
    for ref in references:
        rgb, weight, sigma = [ref[k].detach().to(rendered) for k in ('rgb', 'support', 'sigma')]
        if rgb.shape != low.shape or weight.shape != low[:, :1].shape or sigma.shape != weight.shape:
            raise ValueError('reference RGB, support and scale shapes are inconsistent')
        if not torch.isfinite(rgb).all() or not torch.isfinite(weight).all() or (weight < 0).any() or (weight > 1).any():
            raise ValueError('reference RGB must be finite and support must be in [0,1]')
        residual, valid, _ = scale_residual(low - rgb, weight, sigma, scales)
        numerator = numerator + residual.abs().mean(1, keepdim=True) * valid
        denominator = denominator + valid
        weights.append(valid)
    gate = torch.stack(weights).amax(0)
    evidence = (gate * numerator / denominator.clamp_min(1e-8)).mean()
    support = F.interpolate(gate, rendered.shape[-2:], mode='nearest')
    missing = 1 - support
    # Full-image normalization preserves the absolute amount of missing support.
    completion = ((rendered - generated.detach()).abs().mean(1, keepdim=True) * missing).mean()
    return evidence, completion


def reconstruction_loss(real_render, real_rgb, target_render, target_rgb, references,
                        scales, ssim, completion_weight=.1):
    if not math.isfinite(completion_weight) or not 0 <= completion_weight <= 1:
        raise ValueError('completion_weight must be in [0,1]')
    real = .8 * (real_render - real_rgb.detach()).abs().mean() + .2 * (1 - ssim(real_render, real_rgb.detach()))
    evidence, completion = auxiliary_losses(target_render, target_rgb, references, scales)
    return real + evidence + completion_weight * completion, {
        'real': real.detach(), 'evidence': evidence.detach(), 'completion': completion.detach(),
    }


def camera_matrices(camera, image_size, device='cpu', znear=.01, zfar=100.):
    """Convert RDF column-vector cameras to transposed 3DGS matrices.

    Explicit principal-point offsets avoid silently centering off-axis cameras.
    """
    h, w = image_size
    k = torch.as_tensor(camera['K'], dtype=torch.float32, device=device).reshape(3, 3)
    r = torch.as_tensor(camera['R'], dtype=torch.float32, device=device).reshape(3, 3)
    t = torch.as_tensor(camera['t'], dtype=torch.float32, device=device).reshape(3)
    if not all(torch.isfinite(x).all() for x in (k, r, t)) or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError('camera must have finite parameters and positive focal lengths')
    if abs(float(k[0, 1])) > 1e-6 or abs(float(k[1, 0])) > 1e-6:
        raise ValueError('3DGS rasterizer requires zero-skew intrinsics')
    view = torch.eye(4, device=device)
    view[:3, :3], view[:3, 3] = r, t
    projection = torch.zeros(4, 4, device=device)
    projection[0, 0], projection[1, 1] = 2*k[0, 0]/w, 2*k[1, 1]/h
    # Rasterizer uses ((ndc + 1) * size - 1) / 2 to recover pixel centers.
    projection[0, 2], projection[1, 2] = (2*k[0, 2]+1)/w-1, (2*k[1, 2]+1)/h-1
    projection[2, 2], projection[2, 3] = zfar/(zfar-znear), -zfar*znear/(zfar-znear)
    projection[3, 2] = 1
    return view.T.contiguous(), (projection @ view).T.contiguous(), -r.T @ t


def seed_points(observations, stride=4):
    """Backproject original source depths only, in the recorded world frame."""
    points, colors = [], []
    for obs in observations:
        rgb, depth = obs['rgb'][0], obs['depth'].reshape(*obs['rgb'].shape[-2:])
        mask = obs['mask'].reshape_as(depth)
        h, w = depth.shape
        y, x = torch.meshgrid(torch.arange(0, h, stride), torch.arange(0, w, stride), indexing='ij')
        z = depth[y, x]
        valid = torch.isfinite(z) & (z > 0) & (mask[y, x] > .5)
        pixels = torch.stack((x[valid], y[valid], torch.ones_like(x[valid]))).float()
        k, r, t = [torch.as_tensor(obs['camera'][s]).reshape(shape).float() for s, shape in [('K',(3,3)),('R',(3,3)),('t',(3,1))]]
        xyz = r.T @ (torch.linalg.solve(k, pixels) * z[valid] - t)
        points.append(xyz.T)
        colors.append(rgb[:, y[valid], x[valid]].T)
    if not points or sum(len(p) for p in points) < 4:
        raise ValueError('at least four valid source-depth points are required')
    return torch.cat(points).numpy(), torch.cat(colors).numpy()
