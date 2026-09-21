# Generation-assisted reconstruction

The video prior produces supplemental images. A separate 3DGS optimization fits
the scene to real photographs, fixed projected evidence and generated completion.
The Gaussian scene is trainable; the video prior, VAE and evidence are frozen.

## Export

Add `--observation_guidance --export_reconstruction` to the existing inference
command. Each window writes a unique `.pt` under `save_dir/reconstruction/`.
For the two-input reconstruction protocol, use `--mode sparse_view_interp` and
an image directory containing only the two real training images. The trainer
rejects `single_view_eval` exports because that mode estimates trajectories from
evaluation RGB. Use separate held-out camera metadata for reconstruction metrics.
The export uses the first generated sample and preserves `[0,1]` RGB, RDF
world-to-camera `K,R,t`, original source depth, and per-reference RGB/support/scale.
Original photographs are exported from `is_observation` records, never from
decoded conditioned frames. Generated targets remain separately identified.

For the paired base-generation control, use the same command and seed with
`--obs_strength 0`. This preserves the backbone and bypasses correction without
re-enabling the older FMI/CGA or replay paths. Reconstruction evidence is still
constructed from the original photographs. Export reconstructs this fixed
evidence after sampling; it does not use generated RGB to verify geometry.

## Fit the scene

Install the official graphdeco-inria/gaussian-splatting dependencies and CUDA
extensions on the experiment machine. The following commands run from this
repository. Replace the example paths with actual paths:

```bash
python train_reconstruction.py --bundle /path/to/window.pt \
  --gaussian-repo /path/to/gaussian-splatting --output /path/to/fit \
  --variant evidence --iterations 2000 --completion-weight 0.1
```

By default, source depths initialize a coarse point cloud, sampled every four
pixels. `--init-points /path/to/points.npz` accepts `points` and `colors` arrays
of shape `[N,3]` in the same world frame, with RGB in `[0,1]`. Use exactly the same
initialization for all paired controls. Do not initialize from held-out RGB or
depth. The scene keeps its initial point count: no densification, splitting or
opacity reset. SH degree increases at iterations 1000 and 2000. Original 3DGS
Adam settings are used with opacity learning rate 0.05 and its 30,000-step
position learning-rate schedule. Exposure compensation is disabled.

Each iteration samples one real camera and, except in the real-only control,
one generated camera. The full loss is:

* Real: `0.8 * L1 + 0.2 * (1 - SSIM)` against the original photograph.
* Evidence: normalized Gaussian filtering of each rendered-minus-observed
  residual at the original projection scale; average absolute errors across
  references with support weights, retain maximum support as absolute strength,
  then average over the full processing image.
* Completion: full-resolution L1 to the generated image, weighted by `1-g`
  and by 0.1, averaged over the full image. Here `g` is nearest-neighbor-resized
  maximum effective observation support. It is not generation confidence.

Real and supplemental terms are averaged independently of the number of images.
No-support targets get completion only; full-support pixels get evidence only.
The same scale bank and effective support rules serve generation and fitting.

## Four controls

| Input bundle | Variant | Purpose |
| --- | --- | --- |
| Either, same original images | `real_only` | Real-image baseline |
| Zero-correction generation | `uniform` | Conventional loss on real and generated images |
| Corrected generation | `uniform` | Isolate completion-stage correction |
| Same corrected bundle | `evidence` | Isolate evidence-aware reconstruction |

The uniform variant assigns one conventional image loss to each sampled real
and generated view. Matching optimizer iterations does not imply identical
runtime: real-only uses one render, the other variants use two.

Optional `--evaluation /path/to/heldout.pt` loads `{'views': [{'rgb': tensor,
'camera': {'K': tensor, 'R': tensor, 't': tensor}}, ...]}` only after fitting.
RGB has shape `[1,3,H,W]`. It exports rendered PNGs, per-view PSNR and SSIM;
use the existing evaluation environment to compute LPIPS from these PNGs.
The caller must keep held-out views out of the export and point initialization.
Camera metadata alone may be supplied for rendering. A bundle contains one
coordinate frame; independently reconstructed windows must not be concatenated
without camera/geometry alignment.

## Validation scope

CPU tests check gradients, scale-dependent behavior, support limits, provenance,
duplicate-reference invariance, camera projection and bundle serialization.
The training entry point follows the public upstream renderer/model interfaces.
CUDA rasterizer execution and real-scene quality still require a GPU experiment.
Paper gray table values are simulated layout placeholders, not these tests or
measured training results.
