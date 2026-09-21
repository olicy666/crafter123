"""Training-free scale selection from a local depth perturbation model.

Covariances describe linearized projection sensitivity, not calibrated errors.
All spatial scales are standard deviations in latent pixels.
"""
import math

import torch
import torch.nn.functional as F


def validate_scale_config(mode, relative_std, fixed_sigma, scales):
    if mode not in ('legacy', 'uncertainty', 'fixed'):
        raise ValueError('init_scale_mode must be legacy, uncertainty or fixed')
    values = (relative_std, fixed_sigma, *scales)
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('Uncertainty parameters must be finite and nonnegative')
    if len(scales) < 2 or scales[0] != 0 or any(a >= b for a, b in zip(scales, scales[1:])):
        raise ValueError('init_scales must start at zero and strictly increase')
    if fixed_sigma > scales[-1]:
        raise ValueError('init_fixed_sigma exceeds the largest represented scale')


def projection_covariance(points, depth_direction, intrinsics, depth_std):
    """Return [B,3,N] (xx,xy,yy) in target pixel squared units.

    points and depth_direction are target-camera X and dX/dd, [B,3,N].
    Nonfinite or negative standard deviations produce invalid covariance.
    """
    projected = intrinsics @ points
    derivative = intrinsics @ depth_direction
    denominator = projected[:, 2:3]
    valid = (denominator > 0) & torch.isfinite(denominator)
    valid = valid & torch.isfinite(depth_std) & (depth_std >= 0)
    safe = torch.where(valid, denominator, torch.ones_like(denominator))
    jacobian = (derivative[:, :2] * safe - projected[:, :2] * derivative[:, 2:3]) / safe.square()
    displacement = jacobian * depth_std
    x, y = displacement[:, :1], displacement[:, 1:2]
    covariance = torch.cat((x.square(), x * y, y.square()), dim=1)
    return torch.where(valid, covariance, torch.full_like(covariance, float('nan')))


def latent_sigma(covariance, support, size):
    """Support-normalized covariance resampling and conservative isotropic scale."""
    covariance, support = covariance.float(), support.float()
    valid = torch.isfinite(covariance).all(dim=1, keepdim=True)
    mask = support * valid
    numerator = F.interpolate(torch.where(valid, covariance, 0) * mask, size, mode='bilinear', align_corners=False)
    denominator = F.interpolate(mask, size, mode='bilinear', align_corners=False)
    cov = numerator / denominator.clamp_min(1e-8)
    dx, dy = size[1] / covariance.shape[-1], size[0] / covariance.shape[-2]
    xx, xy, yy = cov[:, :1] * dx**2, cov[:, 1:2] * dx * dy, cov[:, 2:3] * dy**2
    eigenvalue = (xx + yy + torch.sqrt((xx - yy).square() + 4 * xy.square())) / 2
    sigma = eigenvalue.clamp_min(0).sqrt()
    return torch.where(denominator > 1e-8, sigma, torch.full_like(sigma, float('nan')))


def _blur(value, sigma):
    if sigma == 0:
        return value
    radius = math.ceil(3 * sigma)
    x = torch.arange(-radius, radius + 1, device=value.device, dtype=value.dtype)
    kernel = torch.exp(-0.5 * (x / sigma).square())
    kernel = kernel / kernel.sum()
    channels = value.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    return F.conv2d(F.conv2d(value, horizontal, padding=(0, radius), groups=channels),
                    vertical, padding=(radius, 0), groups=channels)


def scale_residual(residual, support, sigma, scales):
    """Interpolate normalized Gaussian filters in variance; abstain beyond bank.

    Invalid/out-of-bank points contribute neither as donors nor recipients.
    This is a local isotropic approximation, not exact spatially varying diffusion.
    """
    eligible = torch.isfinite(sigma) & (sigma >= 0) & (sigma <= scales[-1])
    support = support.float() * eligible
    safe_sigma = torch.where(eligible, sigma, 0).float()
    variance = safe_sigma.square()
    output = torch.zeros_like(residual, dtype=torch.float32)
    scale_weights = []
    for i, scale in enumerate(scales):
        weight = torch.ones_like(variance)
        if i > 0:
            weight = weight.minimum((variance - scales[i-1]**2) / (scale**2 - scales[i-1]**2))
        if i + 1 < len(scales):
            weight = weight.minimum((scales[i+1]**2 - variance) / (scales[i+1]**2 - scale**2))
        weight = weight.clamp(0, 1) * eligible
        numerator = _blur(torch.where(support > 0, residual.float(), 0) * support, scale)
        filtered = numerator / _blur(support, scale).clamp_min(1e-8)
        output = output + weight * filtered
        scale_weights.append(weight)
    output = torch.where(support > 0, output, 0)
    return output.to(residual.dtype), support, torch.cat(scale_weights, dim=1)
