"""Export separate real observations, generated hypotheses and fixed evidence."""
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from .observation_loop import LoopConfig, ObservationLoop


def _cpu(value):
    return torch.as_tensor(value).detach().float().cpu().clone()


def save_reconstruction_window(images, frames, observation_frames, engine, options, window_index):
    if images.ndim != 4 or images.shape[1] != 3 or not torch.isfinite(images).all():
        raise ValueError('reconstruction export needs finite [T,3,H,W] RGB in [0,1]')
    if (images < 0).any() or (images > 1).any():
        raise ValueError('reconstruction RGB must be in [0,1]')
    if not frames or len(frames) != len(images):
        raise ValueError('reconstruction export requires a camera record for each output')
    pool = frames if observation_frames is None else observation_frames
    originals = [f for f in pool if f.is_observation]
    if not originals:
        raise ValueError('reconstruction export requires original observations')
    if any(f.rgb.shape[0] != 1 for f in originals + list(frames)):
        raise ValueError('reconstruction export supports one scene per batch')
    scales = tuple(getattr(options, 'obs_scales', (0., 1., 2., 4., 8.)))
    evidence = ObservationLoop(engine, frames, set(),
        LoopConfig(eval_size=getattr(options, 'obs_eval_size', 256), scales=scales),
        originals, apply_global_routing=False)
    def camera(frame):
        return {key: _cpu(frame.camera[key]) for key in ('K', 'R', 't')}
    observations = [dict(rgb=_cpu(f.rgb), depth=_cpu(f.depth), mask=_cpu(f.mask), camera=camera(f)) for f in originals]
    targets = []
    for index, frame in enumerate(frames):
        if frame.is_observation:
            continue
        refs = [dict(rgb=_cpu(rgb), support=_cpu(weight), sigma=_cpu(sigma))
                for rgb, weight, sigma in evidence.entries.get(index, [])]
        targets.append(dict(rgb=_cpu(images[index:index+1]), camera=camera(frame), references=refs, frame_index=index))
    if not targets:
        raise ValueError('reconstruction export needs at least one generated target')
    payload = dict(schema_version=1, camera_convention='RDF_world_to_camera',
                   observations=observations, targets=targets, scales=scales,
                   window_index=int(window_index), seed=int(getattr(options, 'seed', 42)),
                   trajectory_from_evaluation_images=getattr(options, 'mode', '') == 'single_view_eval')
    directory = Path(options.save_dir) / 'reconstruction'
    directory.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=directory, prefix=f'window_{window_index:06d}_', suffix='.pt', delete=False) as output:
        path = Path(output.name)
        try:
            torch.save(payload, output)
        except Exception:
            path.unlink(missing_ok=True)
            raise
    return str(path)
