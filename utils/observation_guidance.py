"""Scale-aware original-observation correction within one DDIM trajectory."""
from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F

from utils.observation_loop import LoopConfig, ObservationLoop


@dataclass(frozen=True)
class ObservationGuidanceConfig:
    enabled: bool = False
    strength: float = .35
    corrections: int = 3
    eval_size: int = 256
    scales: tuple = (0., 1., 2., 4., 8.)

    def __post_init__(self):
        if not math.isfinite(self.strength) or not 0 <= self.strength <= 1:
            raise ValueError('observation guidance strength must be in [0, 1]')
        for name in ('corrections', 'eval_size'):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'observation guidance {name} must be a positive integer')
        LoopConfig(eval_size=self.eval_size, scales=self.scales)

    @classmethod
    def from_options(cls, options):
        defaults = cls()
        return cls(**{name: getattr(options, 'obs_' + name, getattr(defaults, name))
                      for name in cls.__dataclass_fields__})


class ObservationGuidance:
    """Fixed evidence; no reference ranking, global routing, or outer replay.

    The correction is a filtered RGB residual, not a proximal solve or an
    exact posterior update. VAE encoding uses its mode to avoid adding noise.
    """

    def __init__(self, engine, frames, conditioned, config, observation_frames=None):
        self.engine, self.config = engine, config
        self.evidence = ObservationLoop(
            engine, frames, conditioned,
            LoopConfig(enabled=True, eval_size=config.eval_size, scales=config.scales),
            observation_frames, apply_global_routing=False,
        )
        self.indices = set()
        self.records = self.evidence.records
        for record in self.records.values():
            record.pop('rounds', None)
            record.pop('metric', None)
            record.update(config=asdict(config), interventions=[],
                          mode='sampling_observation_guidance',
                          global_routing=False,
                          depth_rel_std=engine.init_depth_rel_std)

    def set_schedule(self, total_steps):
        # The final half of the actual schedule, including its last update.
        available = max(1, total_steps // 2)
        count = min(self.config.corrections, available)
        self.indices = ({0} if count == 1 else
                        {round(i * (available - 1) / (count - 1)) for i in range(count)})

    def should_correct(self, index):
        return (self.config.enabled and self.config.strength > 0
                and bool(self.evidence.entries) and index in self.indices)

    @torch.no_grad()
    def __call__(self, model, clean, index, timestep):
        if not self.should_correct(index):
            return clean
        images = model.decode_first_stage(clean)
        _, _, correction, support = self.evidence.evaluate(images, compute_scores=False)
        corrected = (images + self.config.strength * correction).clamp(-1, 1)
        result = clean.clone()
        for target in self.evidence.entries:
            # Keep zero-support pixels out of the VAE input difference, even
            # when upsampling has spread a neighboring residual into them.
            candidate = torch.where(support[:, :, target].bool(),
                                    corrected[:, :, target], images[:, :, target])
            old = self.engine._encode_guide_latent(images[:, :, target])
            new = self.engine._encode_guide_latent(candidate)
            if old.shape != clean[:, :, target].shape or new.shape != old.shape:
                raise ValueError('Observation guidance VAE latent shape mismatch')
            gate = F.interpolate(support[:, :, target], clean.shape[-2:], mode='area') > 0
            delta = torch.where(gate, new - old, 0).to(clean)
            proposal = clean[:, :, target] + delta
            finite = torch.isfinite(proposal).flatten(1).all(1)
            result[:, :, target] = torch.where(finite[:, None, None, None],
                                               proposal, clean[:, :, target])
            self.records[target]['interventions'].append(dict(
                index=int(index), timestep=int(timestep[0]),
                applied=finite.detach().cpu(),
                latent_delta_rms=delta.float().square().flatten(1).mean(1).sqrt().detach().cpu(),
            ))
        return result


def correct_clean_prediction(sampler, pred_x0, corrector, index, timestep):
    """Convert the native predicted x0 to VAE units before correcting it."""
    if corrector is None or not corrector.should_correct(index):
        return pred_x0
    model = sampler.model
    if model.use_dynamic_rescale:
        scale = sampler.ddim_scale_arr[index].to(pred_x0)
        if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
            raise ValueError('Dynamic latent scale must be finite and positive')
        clean = pred_x0 / scale
        corrected = corrector(model, clean, index, timestep)
        # Difference form preserves the original prediction for a zero update.
        return pred_x0 + (corrected - clean) * scale
    return corrector(model, pred_x0, index, timestep)
