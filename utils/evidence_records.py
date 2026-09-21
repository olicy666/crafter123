"""Persist sampling diagnostics without retaining every window in memory."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import torch


_CONFIG_KEYS = (
    'obs_enabled', 'obs_strength', 'obs_corrections', 'obs_eval_size', 'obs_scales',
    'loop_enabled', 'loop_max_rounds', 'loop_resample_steps', 'loop_strength',
    'loop_min_improvement', 'loop_error_threshold', 'loop_eval_size', 'loop_scales',
    'seed', 'n_samples', 'ddim_steps', 'ddim_eta', 'height', 'width',
    'video_length', 'frame_stride', 'unconditional_guidance_scale',
    'cfg_img', 'guidance_rescale', 'timestep_spacing', 'multiple_cond_cfg',
    'use_evidence_routing', 'use_fmi', 'use_cga', 'use_freq_mix',
    'filter_type', 'freq_cutoff', 'low_freq_norm', 'noise_level',
    'init_scale_mode', 'init_depth_rel_std', 'init_fixed_sigma', 'init_scales',
    'num_recent', 'num_ref', 'overlap_threshold', 'depth_rel_tolerance',
    'depth_abs_tolerance', 'conflict_threshold', 'min_support_ratio',
    'admit_ratio_threshold', 'max_conflict_ratio', 'attenuated_weight',
    'abstain_gate', 'geo_early_scale', 'geo_late_scale',
)


def save_evidence_window(record, save_dir, window_index, options, frame_list=None):
    """Save one window with local camera indices and collision-safe filenames."""
    directory = Path(save_dir) / 'evidence'
    directory.mkdir(parents=True, exist_ok=True)
    cameras = []
    for frame in frame_list or []:
        cameras.append({
            key: value.detach().cpu().clone() if torch.is_tensor(value) else value
            for key, value in frame.camera.items()
        })
    payload = {
        'schema_version': 1,
        'window_index': window_index,
        'config': {
            key: getattr(options, key) for key in _CONFIG_KEYS
            if hasattr(options, key)
        },
        'cameras': cameras,
        'is_observation': [bool(getattr(frame, 'is_observation', False)) for frame in frame_list or []],
        'samples': record,
    }
    # A unique suffix preserves earlier runs in the same scene directory.
    with NamedTemporaryFile(
        dir=directory, prefix=f'window_{window_index:06d}_',
        suffix='.pt', delete=False,
    ) as output:
        path = Path(output.name)
        try:
            torch.save(payload, output)
        except Exception:
            path.unlink(missing_ok=True)
            raise
    return str(path)
