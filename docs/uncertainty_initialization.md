# Projection-uncertainty-aware initialization

The complete observation-feedback extension is now available through
`--observation_loop`; see [its usage and algorithm](observation_loop.md).

This implementation adds training-free, spatially varying smoothing of the
existing evidence-guided initialization residual for rigid-scene novel-view
synthesis. The pretrained model, geometry verifier and attention controller are
unchanged. No optimizer or additional training loop is introduced.

## Running and controls

Append the following flags to your working `python inference.py` command,
retaining its checkpoint, image, camera trajectory and rendering options:

```sh
--init_scale_mode uncertainty --init_depth_rel_std 0.05 --init_scales 0 0.5 1 2 4 8
```

The default is `--init_scale_mode legacy`, which retains the original initialization.
For the fixed smoothing control use:

```sh
--init_scale_mode fixed --init_fixed_sigma 1 --init_scales 0 0.5 1 2 4 8
```

Scales are Gaussian standard deviations in **latent pixels**, not image pixels
or frequency cutoffs. They must start at zero and strictly increase. Negative
or nonfinite parameters are rejected. Fixed sigma must not exceed the bank.
The last represented scale is an abstention boundary, not a clipping threshold.

Keep the seed, sampler, existing frequency cutoff and normalization, references,
and geometry thresholds identical across comparisons. `--disable_fmi` bypasses
all initialization modes. `--disable_cga` retains initialization and disables
attention guidance. `--disable_evidence_routing` uses projected source masks in
place of verified evidence; this is an explicit ablation, not the full method.
Conditioned views and views without selected references retain their original
initial noise. Existing YAML model configuration does not require new entries;
these options enter through the inference CLI.

## Computation and interpretation

For source camera-axis depth d, target-camera position is X(d)=a*d+b, where
a=R_target*R_source^T*K_source^-1*[u,v,1]^T. Differentiating the perspective
projection gives J=d(project(X))/dd. The target image covariance is
Sigma=sigma_d^2*J*J^T, stored as (xx,xy,yy) in squared image pixels. The same
z-buffer winner provides RGB, target-camera depth, and covariance.

By default sigma_d=0.05*d is an **assumed depth perturbation model**. It is not
estimated from depth disagreement or claimed to be calibrated. A programmatic
caller can supply `FrameData(..., depth_std=...)`, a map matching depth's shape
and units; the standard inference pipeline currently uses the relative-depth
assumption. Invalid supplied standard deviations remain invalid and do not fall
back to zero or to the default assumption.

Covariance is resampled using valid evidence support, then transformed to latent
coordinates by diag(W_latent/W_image, H_latent/H_image). The square root of its
largest eigenvalue sets a conservative isotropic smoothing scale. For each
reference, the existing FMI guide-minus-noise residual is filtered with a
support-normalized Gaussian bank. Adjacent filters are linearly interpolated in
variance. Unsupported, invalid, and out-of-bank positions do not donate or receive
residuals. Per-reference evidence then weights the residual consensus, and the
maximum local evidence gates its addition to the unchanged shared base noise.

For a locally constant Gaussian displacement delta~N(0,Sigma),
E[exp(i*omega^T*delta)]=exp(-omega^T*Sigma*omega/2). This standard identity motivates
suppressing fine spatial components as projection sensitivity increases. The
implementation is a local isotropic approximation to this reasoning. Finite
Gaussian support, variance interpolation, spatially varying masks, and nonlinear
VAE encoding mean it is not an exact expected diffusion process or a theorem
about generated geometry. It adds smoothing to the original FMI residual; it
does not restore frequencies already removed by the original FMI filter.

The current covariance covers depth perturbations only. Pose errors, visibility
changes, correlated reference errors, and reference provenance are not modeled.
Pure rotation correctly produces zero depth-induced projection uncertainty;
that does not establish that its correspondence is otherwise reliable. The
support-normalized filter excludes invalid donors locally; it cannot undo
information mixing that already occurred in the VAE or the original FFT.

## Records and verification

With `--save_evidence true` (the existing default), each exported frame has an
`initialization` record containing mode, applied flag, and applied/skipped reason.
Processed frames include the effective latent injection mask. Adaptive frames
also contain the selected references' image covariance, uncertainty sources,
relative-depth assumption, latent sigma, excluded-scale mask, and scale bank.
Applied nonlegacy frames additionally save scale interpolation weights.
Latent sigma/weights flatten `[reference, batch]` in that order; covariance retains
`[reference, batch, 3, H, W]`. NaNs in covariance/sigma explicitly mark unavailable
uncertainty, while the effective injection mask is zero there. All tensors are
detached to CPU for export. These are additive fields in record schema version 1.
Dense covariance records increase storage usage; disable evidence saving when
those diagnostics are not needed.

Run the regression suite from the code root with:

```sh
python -m pytest -q tests
```

It checks analytical and finite-difference projection sensitivity, camera and
z-buffer behavior, covariance unit conversion, unsupported/invalid evidence,
spatial-scale response, legacy equivalence at zero scale, multiple references,
configuration, disabling, and record persistence. Small tensor integration tests
use an encoder stand-in. Full pretrained-model quality and runtime measurements
still require the project's GPU/checkpoint environment. No benchmark improvement
is implied by passing these implementation tests.
