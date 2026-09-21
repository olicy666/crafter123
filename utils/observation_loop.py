"""Frozen-model observation feedback with fixed evidence and bounded DDIM replay."""
from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F

from utils.uncertainty_guidance import latent_sigma, scale_residual


@dataclass(frozen=True)
class LoopConfig:
    enabled: bool = False
    max_rounds: int = 2
    resample_steps: int = 10
    strength: float = 0.35
    min_improvement: float = 1e-4
    error_threshold: float = 0.01
    eval_size: int = 256
    scales: tuple = (0., 1., 2., 4., 8.)

    def __post_init__(self):
        for name in ('max_rounds', 'resample_steps', 'eval_size'):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'loop {name} must be a positive integer')
        for name in ('strength', 'min_improvement', 'error_threshold'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'loop {name} must be finite and nonnegative')
        if not 0 < self.strength <= 1 or self.min_improvement <= 0:
            raise ValueError('loop strength must be in (0,1]; min_improvement must be positive')
        if (len(self.scales) < 2 or self.scales[0] != 0
                or any(not math.isfinite(s) or s < 0 for s in self.scales)
                or any(a >= b for a, b in zip(self.scales, self.scales[1:]))):
            raise ValueError('loop scales must be finite, start at zero and strictly increase')

    @classmethod
    def from_options(cls, options):
        defaults = cls()
        return cls(**{name: getattr(options, 'loop_' + name, getattr(defaults, name))
                      for name in cls.__dataclass_fields__})


def flatten_video(video):
    b, c, t, h, w = video.shape
    return video.permute(2, 0, 1, 3, 4).reshape(t*b, c, h, w)


def unflatten_video(images, batch, frames):
    return images.reshape(frames, batch, *images.shape[1:]).permute(1, 2, 0, 3, 4)


def noise_at_schedule(model, clean, noise, step):
    """Forward noise at the exact replay timestep, including training rescaling."""
    t = torch.full((clean.shape[0],), int(step), device=clean.device, dtype=torch.long)
    if getattr(model, 'use_dynamic_rescale', False):
        clean = clean * model.scale_arr[int(step)].to(clean)
    return model.q_sample(x_start=clean, t=t, noise=noise)


class ObservationLoop:
    """Evidence never changes after construction; generated frames are not anchors.

    Scores are support-normalized RGB discrepancies at uncertainty-dependent
    scales, not a calibrated likelihood. Acceptance is per sequence and requires
    no supported frame to worsen, as well as a positive mean improvement.
    """
    @torch.no_grad()
    def __init__(self, engine, frames, conditioned, config, observation_frames=None,
                 apply_global_routing=True):
        self.engine, self.config = engine, config
        self.entries = {}
        self.records = {t: {'enabled': True, 'rounds': [], 'reason': 'no_observations'}
                        for t in range(len(frames))}
        pool = frames if observation_frames is None else observation_frames
        anchors = [(i, f) for i, f in enumerate(pool) if getattr(f, 'is_observation', False)]
        self.anchor_indices = [i for i, _ in anchors]
        self.anchor_cameras = [{k: torch.as_tensor(v).detach().cpu() for k, v in f.camera.items()}
                               for _, f in anchors]
        for target, frame in enumerate(frames):
            record = self.records[target]
            record.update(observation_indices=self.anchor_indices,
                          observation_cameras=self.anchor_cameras,
                          config=asdict(config),
                          metric='fixed_support_multiscale_rgb_l1')
            if target in conditioned or getattr(frame, 'is_observation', False):
                record['reason'] = 'protected_frame'
                continue
            if not anchors:
                continue
            h, w = engine._as_map(frame.depth, 'target depth').shape[-2:]
            ratio = min(1., config.eval_size / max(h, w))
            size = (max(1, round(h * ratio)), max(1, round(w * ratio)))
            warps = {}
            for local, (_, source) in enumerate(anchors):
                depth = engine._as_map(source.depth, 'source depth')
                mask = engine._as_map(source.mask, 'source mask')
                depth = torch.where(mask > .5, depth, 0)
                rgb, z, support, uncertainty = engine.warp_rgb_depth(
                    source.rgb, depth, source.camera, frame.camera, h, w,
                    source_depth_std=getattr(source, 'depth_std', None), return_uncertainty=True,
                )
                warps[local] = dict(warped_rgb=rgb, warped_depth=z, warped_mask=support, **uncertainty)
            # Use the existing geometry verifier even when FMI/CGA are disabled.
            from utils.evidence_routing import compute_geometry_evidence
            evidence = compute_geometry_evidence(
                torch.stack([v['warped_depth'] for v in warps.values()]),
                torch.stack([v['warped_mask'] for v in warps.values()]),
                engine._as_map(frame.depth, 'target depth'), engine._as_map(frame.mask, 'target mask'),
                depth_rel_tolerance=engine.depth_rel_tolerance,
                depth_abs_tolerance=engine.depth_abs_tolerance,
                conflict_threshold=engine.conflict_threshold,
                min_support_ratio=engine.min_support_ratio,
                admit_ratio_threshold=engine.admit_ratio_threshold,
                max_conflict_ratio=engine.max_conflict_ratio,
                attenuated_weight=engine.attenuated_weight,
            )
            maps = evidence['per_reference_evidence']
            if maps.ndim == 4:
                maps = maps.unsqueeze(2)
            if apply_global_routing:
                maps = maps * evidence['state_strength'].view(1, -1, 1, 1, 1)
            references = []
            for ref, info in warps.items():
                covariance = info['projection_covariance']
                support = maps[ref] * torch.isfinite(covariance).all(1, keepdim=True)
                sigma = latent_sigma(covariance, support, size)
                denominator = F.interpolate(support, size, mode='area')
                rgb = F.interpolate(info['warped_rgb'] * support, size, mode='area') / denominator.clamp_min(1e-8)
                eligible = torch.isfinite(sigma) & (sigma <= config.scales[-1])
                support = denominator * eligible
                references.append((rgb.detach(), support.detach(), sigma.detach()))
            if not any(bool((s > 1e-6).any()) for _, s, _ in references):
                record['reason'] = 'no_reliable_support'
                continue
            self.entries[target] = references
            record.update(reason='ready', eval_size=list(size), scales=list(config.scales),
                          support=torch.stack([s for _, s, _ in references]).cpu(),
                          sigma=torch.stack([s for _, _, s in references]).cpu(),
                          uncertainty_sources=[v['uncertainty_source'] for v in warps.values()])

    def evaluate(self, images, compute_scores=True):
        """Return fixed per-frame scores and current observation residuals."""
        batch, _, frames, h, w = images.shape
        scores = torch.zeros(batch, frames, device=images.device)
        present = torch.zeros(batch, frames, device=images.device, dtype=torch.bool)
        correction = torch.zeros_like(images)
        support_video = torch.zeros(batch, 1, frames, h, w, device=images.device)
        for target, references in self.entries.items():
            size = references[0][0].shape[-2:]
            current = F.interpolate((images[:, :, target].float() + 1) / 2, size, mode='area')
            numerator = torch.zeros(batch, device=images.device)
            denominator = torch.zeros_like(numerator)
            delta = torch.zeros_like(current)
            weights = torch.zeros_like(current[:, :1])
            for rgb, mask, sigma in references:
                # Each supported scale is scored separately: errors cannot
                # cancel across references or coarse/fine representations.
                residual = rgb - current
                filtered, valid, _ = scale_residual(residual, mask, sigma, self.config.scales)
                delta += filtered * valid
                weights += valid
                for floor in (self.config.scales if compute_scores else ()):
                    level_sigma = torch.maximum(sigma, torch.full_like(sigma, floor))
                    level, _, _ = scale_residual(residual, mask, level_sigma, self.config.scales)
                    numerator += (level.abs().mean(1, keepdim=True) * valid).sum((1, 2, 3))
                    denominator += valid.sum((1, 2, 3))
            scores[:, target] = numerator / denominator.clamp_min(1e-8)
            present[:, target] = denominator > 1e-6
            delta = delta / weights.clamp_min(1e-8)
            gate = torch.stack([s for _, s, _ in references]).amax(0)
            correction[:, :, target] = 2 * F.interpolate(delta * gate, (h, w), mode='bilinear', align_corners=False)
            support_video[:, :, target] = F.interpolate((gate > 1e-6).float(), (h, w), mode='nearest')
        return scores, present, correction, support_video

    @torch.no_grad()
    def run(self, model, sampler, samples, images, sample_kwargs, seed):
        config = self.config
        if not self.entries:
            return samples, images, self.records
        scores, present, correction, support = self.evaluate(images)
        batch, _, frames, _, _ = samples.shape
        active = present.any(1)
        stop_reasons = ['budget_exhausted' if bool(v) else 'no_reliable_support' for v in active]
        for target, record in self.records.items():
            if target in self.entries:
                record['initial_error'] = scores[:, target].detach().cpu()
        steps = min(config.resample_steps, len(sampler.ddim_timesteps))
        timestep = int(sampler.ddim_timesteps[steps - 1])
        for round_index in range(config.max_rounds):
            mean = (scores * present).sum(1) / present.sum(1).clamp_min(1)
            converged = active & (mean <= config.error_threshold)
            for b in converged.nonzero().flatten().tolist():
                stop_reasons[b] = 'error_threshold'
            active = active & ~converged
            if not bool(active.any()):
                break
            # Encode a difference so VAE reconstruction error alone is not a correction.
            corrected = (images + config.strength * correction).clamp(-1, 1)
            deltas = []
            for target in range(frames):
                if target not in self.entries:
                    deltas.append(torch.zeros_like(samples[:, :, target]))
                    continue
                encoded = self.engine._encode_guide_latent(
                    torch.cat((images[:, :, target], corrected[:, :, target]))
                )
                old, new = encoded.chunk(2)
                deltas.append(new - old)
            delta = torch.stack(deltas, dim=2)
            if delta.shape != samples.shape:
                raise ValueError('Loop VAE latent shape must match the sampled latent')
            latent_support = F.interpolate(flatten_video(support), samples.shape[-2:], mode='area')
            # Only fully supported latent cells are mutable; the rest follow
            # the accepted latent at every replay timestep.
            mutable = unflatten_video((latent_support >= 1 - 1e-6).to(samples), batch, frames)
            mutable *= active.view(batch, 1, 1, 1, 1)
            mutable_batch = mutable.flatten(1).any(1)
            for b in (active & ~mutable_batch).nonzero().flatten().tolist():
                stop_reasons[b] = 'no_supported_latent_cells'
            active &= mutable_batch
            if not bool(active.any()):
                break
            proposal = samples + mutable * delta
            devices = [samples.device.index or 0] if samples.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(seed) + 100003 + round_index)
                noise = torch.randn_like(samples)
                noised = noise_at_schedule(model, proposal, noise, timestep)
                replay_kwargs = dict(sample_kwargs)
                replay_kwargs.update(x_T=noised, resume_steps=steps,
                                     replay_reference=samples, replay_mask=1 - mutable, replay_noise=noise)
                candidate, _ = sampler.sample(**replay_kwargs)
            candidate = torch.where(mutable.bool(), candidate, samples)
            candidate_images = model.decode_first_stage(candidate)
            # Preserve exact pixel outputs outside support, including observed
            # and conditioned views, despite the decoder's receptive field.
            pixel_mutable = support * active.view(batch, 1, 1, 1, 1)
            candidate_images = torch.where(pixel_mutable.bool(), candidate_images, images)
            new_scores, _, new_correction, _ = self.evaluate(candidate_images)
            new_mean = (new_scores * present).sum(1) / present.sum(1).clamp_min(1)
            finite = torch.isfinite(candidate_images).flatten(1).all(1) & torch.isfinite(candidate).flatten(1).all(1)
            nonworse = ((new_scores <= scores) | ~present).all(1)
            accepted = active & finite & nonworse & (mean - new_mean >= config.min_improvement)
            for b in (active & ~accepted).nonzero().flatten().tolist():
                stop_reasons[b] = 'rejected_nonfinite' if not bool(finite[b]) else 'insufficient_improvement'
            for target, record in self.records.items():
                if target in self.entries:
                    record['rounds'].append(dict(
                        round=round_index + 1, active=active.cpu().clone(), accepted=accepted.cpu().clone(),
                        before=scores[:, target].cpu().clone(), candidate=new_scores[:, target].cpu().clone(),
                        ddim_steps=steps, start_timestep=timestep,
                    ))
            choose = accepted.view(batch, 1, 1, 1, 1)
            samples = torch.where(choose, candidate, samples)
            images = torch.where(choose, candidate_images, images)
            scores = torch.where(accepted[:, None], new_scores, scores)
            correction = torch.where(choose, new_correction, correction)
            active &= accepted
        for target, record in self.records.items():
            if target in self.entries:
                record.update(final_error=scores[:, target].detach().cpu(),
                              reason='finished', max_rounds=config.max_rounds,
                              stop_reasons=stop_reasons,
                              extra_ddim_steps=len(record['rounds']) * steps,
                              attempted_rounds=len(record['rounds']))
        return samples, images, self.records
