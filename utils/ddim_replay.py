"""Shared validated replay arguments for both DDIM conditioning variants."""
import torch

from utils.observation_loop import noise_at_schedule


def prepare_replay(kwargs, schedule, original_steps, timesteps, image):
    count = kwargs.pop('resume_steps', None)
    reference = kwargs.pop('replay_reference', None)
    mask = kwargs.pop('replay_mask', None)
    noise = kwargs.pop('replay_noise', None)
    if count is not None:
        if original_steps or timesteps is not None:
            raise ValueError('resume_steps requires the DDIM schedule without a timesteps override')
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= len(schedule):
            raise ValueError('resume_steps must be between 1 and the DDIM schedule length')
    provided = [v is not None for v in (reference, mask, noise)]
    if any(provided) and not all(provided):
        raise ValueError('replay_reference, replay_mask and replay_noise must be supplied together')
    if reference is not None:
        if reference.shape != image.shape or noise.shape != image.shape:
            raise ValueError('replay reference/noise must match the latent shape')
        if mask.shape != (image.shape[0], 1, *image.shape[2:]):
            raise ValueError('replay mask must have one channel and match batch/spatial dimensions')
        if not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
            raise ValueError('replay mask must be finite and binary')
        reference, mask, noise = (v.to(image) for v in (reference, mask, noise))
    return count, reference, mask, noise


def preserve_replay(model, image, reference, mask, noise, step):
    if reference is None:
        return image
    noised = noise_at_schedule(model, reference, noise, step).to(image)
    return torch.where(mask.bool(), noised, image)
