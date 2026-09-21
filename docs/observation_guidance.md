# Sampling-time observation guidance

This opt-in GeoWitness mode uses fixed original-observation evidence to correct clean predictions within one DDIM trajectory. The frozen video model supplies a generative prior for novel-view synthesis.

Append `--observation_guidance` to an existing inference command. Do not combine it with `--observation_loop`. Omitting the new flag retains the legacy path.

```bash
python inference.py \
  --image_dir test/images/car.jpg \
  --ckpt_path ./checkpoints/model.ckpt \
  --model_path ./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  --mode single_view_autotraj \
  --height 320 --width 576 --video_length 16 \
  --ddim_steps 10 --device cuda:0 \
  --observation_guidance
```

Use paths and resolution compatible with your installed weights, as for the original entry point. The example follows `run.sh`; it is not a completed model run.

## Configuration

| Option | Default | Meaning |
|---|---|---|
| `--init_depth_rel_std` | 0.05 | Assumed depth standard deviation / depth when external uncertainty is unavailable; reused option name |
| `--obs_strength` | 0.35 | RGB residual update strength in [0, 1] |
| `--obs_corrections` | 3 | Maximum interventions, uniformly placed in the final half of the actual DDIM schedule, including the last update |
| `--obs_eval_size` | 256 | Maximum evaluation image dimension |
| `--obs_scales` | 0 1 2 4 8 | Gaussian standard deviations in evaluation pixels |

Defaults are implementation choices, not tuned results. Local depth/visibility/conflict tolerances and base sampler settings remain relevant. Global routing states, candidate ranking, FMI and CGA are bypassed, regardless of their legacy defaults. No outer replay or acceptance loop runs.

## Computation and compatibility

Only `FrameData.is_observation=True` supplies RGB anchors. An external observation pool supports windowed generation. Geometry, support and projection scales are fixed per sampling call. Zero or invalid support supplies no correction. The controller filters observation-minus-prediction residuals at the supported scale, fuses references, and encodes the RGB correction as a deterministic VAE mode difference. Completely unsupported latent cells and protected frames receive no direct correction.

The hook executes before DDIM dynamic rescaling, converting to VAE latent units and back. Both DDIM implementations and epsilon/v prediction are supported. Correction preserves the predicted epsilon direction. Quantized predictions, original DDPM stepping and replay are rejected when the controller is attached. A direct `p_sample_ddim` caller must configure the controller schedule first; normal `sample` calls do so automatically.

Per-frame evidence records contain `observation_guidance`, fixed support/scales, original source IDs and intervention steps. Nonfinite updates fall back per batch item and frame. Strength zero or no observation evidence skips VAE correction entirely. This does not necessarily equal a legacy run with FMI/CGA enabled: the new mode bypasses those modules.

The update is not an exact proximal solve and carries no global convergence or monotonic-error guarantee. Unsupported final RGB pixels may change through the decoder and subsequent denoising. Protected frames remain unmodified by this intervention; base sampling does not promise exact input reproduction.

## Validation

```bash
PYTHONPATH=. python -m pytest -q tests
```

Tests use analytic denoisers, a lightweight VAE and the actual geometry/DDIM code on CPU. They cover legacy regressions, dynamic rescaling, both samplers, epsilon/v prediction, deterministic/stochastic DDIM, observation provenance, bypassing legacy controls and no-op/failure cases. Real checkpoint inference, CUDA/low-precision behavior, memory cost and synthesis quality still require a model run. No published result is assigned to this new mode.
