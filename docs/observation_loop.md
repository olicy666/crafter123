# Training-free observation feedback loop

## Enable

Append one flag to an existing working inference command:

```sh
--observation_loop
```

This activates uncertainty-aware initialization and the complete observation
feedback loop. It sets the effective `init_scale_mode` to `uncertainty`, which
is also saved in the run configuration. No new checkpoint, learned verifier,
optimizer, external service, or training is needed. Without the flag the loop
is disabled and the original initialization choice remains effective.

An explicit configuration is:

```sh
--observation_loop --loop_max_rounds 2 --loop_resample_steps 10 \
--loop_strength 0.35 --loop_min_improvement 0.0001 \
--loop_error_threshold 0.01 --loop_eval_size 256 --loop_scales 0 1 2 4 8
```

Keep the original image, checkpoint, trajectory, resolution and sampler flags.
Both ordinary CFG and multiple-condition CFG use the loop; `ddim_eta=0` and
stochastic DDIM are supported, as is the model's dynamic latent rescaling.
`--disable_fmi` and `--disable_cga` retain their respective meanings. The loop
has its own observation verifier, which remains active even when the original
initialization/attention evidence-routing ablation is disabled.

## Complete execution path

1. Run the original full DDIM generation with the configured initialization.
2. Build fixed observation evidence from **original input photographs**. Warp
   them to each generated view, retaining the visible z-buffer winner's depth
   and projection covariance. Verify depth and reference conflicts using the
   existing geometry verifier. Targets may use observations on either side of
   the trajectory, including original photographs outside the current sparse
   interpolation window.
3. Resize evidence to a bounded evaluation resolution with support-normalized
   RGB resampling. Convert projection covariance to evaluation pixel units.
   Invalid or out-of-bank uncertainty is excluded. These reference images,
   supports, uncertainty fields, scales and scoring denominators are fixed for
   every candidate in this loop.
4. Evaluate the current decoded output against each observed projection. At
   every bank scale, use the greater of that scale and the local uncertainty
   sigma to filter the RGB residual. Take absolute residuals separately before
   aggregating references and scales. This is a fixed multiscale RGB L1 proxy,
   not an exact marginalized likelihood or a learned perceptual metric.
5. Recompute the uncertainty-filtered signed residual from the current output.
   Combine references with fixed evidence weights and use it to propose an RGB
   correction. Encode the difference between the corrected image and current
   image and add that latent difference to the accepted latent. Each frame's
   pair is encoded separately to bound encoder batch memory. No model weights
   are updated and no gradients through the decoder are needed.
6. Forward-noise this proposal to the precise timestep at the end of the first
   `loop_resample_steps` entries of the **original ascending DDIM schedule**.
   Replay those entries in descending order with the real denoiser. The count
   is clipped to the available schedule length. Dynamic rescaling is applied
   before forward noising, matching the training convention.
7. Preserve conditioned and observed frames, unsupported regions, and inactive
   batch items. Only fully supported latent cells can change. At each replay
   timestep the protected latent follows the accepted latent's forward-noise
   trajectory with one fixed noise realization. The final latent is restored
   exactly there. After decoding, unsupported pixels are restored from the
   accepted RGB output, since the decoder can otherwise spread changes across
   a mask boundary. Consequently the delivered RGB can differ from a fresh
   decode of the accepted latent; subsequent rounds retain both states.
8. Re-evaluate the candidate on the same evidence. Accept independently for
   each batch sequence only if it is finite, no supported target frame's score
   worsens, and mean supported-frame error drops by `loop_min_improvement`.
   Otherwise discard the candidate and stop that batch item's extra rounds.
   Stop also at `loop_error_threshold`, no mutable latent support, or budget.

Accepted sequences can therefore improve their fixed proxy monotonically. This
does **not** prove true novel-view geometry improves, nor guarantee that a
candidate will be accepted. RGB differences can reflect lighting, specularity
or exposure as well as generation errors. The objective's coarse components
can still favor smooth images; compare against fixed smoothing at equal compute.

## Parameters and budget

| Flag | Default | Units / interpretation |
|:--|:--|:--|
| `--loop_max_rounds` | 2 | Maximum attempted extra rounds after initial generation |
| `--loop_resample_steps` | 10 | Actual DDIM steps per extra round |
| `--loop_strength` | 0.35 | RGB residual correction multiplier, in (0,1] |
| `--loop_min_improvement` | 0.0001 | Required absolute drop in mean observation proxy |
| `--loop_error_threshold` | 0.01 | Stop when mean observation proxy is this small |
| `--loop_eval_size` | 256 | Maximum evaluation image dimension, no upsampling |
| `--loop_scales` | 0 1 2 4 8 | Gaussian standard deviations in evaluation pixels |

Loop scales differ from initialization scales, which use latent pixels. A
projection sigma above the loop bank maximum abstains; it is not clamped down.
Depth covariance still uses `init_depth_rel_std` (default .05), or an explicitly
provided `FrameData.depth_std`. This is a perturbation model, not automatically
calibrated uncertainty.

At defaults with a 50-step base schedule, at most 20 extra DDIM steps are added.
Each step can require multiple denoiser evaluations for CFG. Masks restrict
where results change; they do not implement sparse UNet execution. The full
sequence/batch is replayed while any batch item is active. Decoding, encoding,
observation warps and metric computation add costs beyond the denoiser steps.
Replay RNG is seeded independently per sample/round and restores the caller's
RNG state. Repeated runs with identical inputs/settings are reproducible subject
to the project's usual backend determinism limitations.

## Observation provenance and programmatic use

`FrameData(..., is_observation=True)` declares an original photograph with its
matching camera, depth and valid mask. The default is false. Never mark a
rendering or generated conditioning image as an observation.

The standard ViewCrafter preparation marks original reference overrides; sparse
interpolation passes the full original observation pool to each window. Iterative
branches using generated reference images do not mark those images as observed.
A window with no original observations explicitly reports `no_observations` and
returns its initial generation. To use genuine external anchors programmatically,
pass an `observation_frames` list with cameras in the same coordinate system.

For the lower-level synthesis API, pass:

```python
from utils.observation_loop import LoopConfig

# Additional keyword arguments to the existing image_guided_synthesis call:
loop_config = LoopConfig(enabled=True)
# loop_config=loop_config, observation_frames=original_frames
```

The low-level loop flag controls feedback; set the supplied router's
`init_scale_mode='uncertainty'` when creating it if uncertainty initialization is
also desired. The CLI does this automatically. Enabling feedback without a
router/frame list raises an error instead of silently bypassing the feature.

## Records and testing

Existing evidence files retain their structure and add per-frame
`observation_loop` records: effective config, original observation indices and
cameras, fixed support and sigma, initial/final error, attempted rounds,
per-batch activity/acceptance, candidate error, exact restart timestep, extra
DDIM step count and stop reasons. `is_observation` is also stored alongside the
window's camera list. Indices in loop records refer to the passed observation
pool; its camera list is saved so external-window anchors can be identified.
No-obs/protected/no-support targets have explicit reasons and no round entries.

Run from the code root:

```sh
python -m pytest -q tests
```

Tests include both actual DDIM sampler implementations with an analytic denoiser,
dynamic rescaling, stochastic replay, ordinary/multiple CFG, exact step counts,
protected trajectories, batch-wise acceptance, rejection/rollback, fixed evidence,
early exit, invalid uncertainty, external anchors, evidence persistence, and the
complete synthesis call chain. They do not replace a CUDA/checkpoint run or
establish benchmark gains. Existing NumPy/PyTorch schedule interoperability may
emit deprecation warnings under NumPy 2; those are separate from loop behavior.
