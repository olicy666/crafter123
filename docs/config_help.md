## Important configuration options for [inference.py](../inference.py):

Use `--observation_loop` to enable the complete training-free feedback method
(uncertainty initialization plus observed-image verification and bounded DDIM
replay). See [observation loop](observation_loop.md) for parameters, provenance,
acceptance rules, ablations and diagnostics.

For the new uncertainty-aware initialization and its ablation controls, see
[uncertainty initialization](uncertainty_initialization.md).

| Option | Default | Meaning |
|:--|:--|:--|
| `--init_scale_mode` | `legacy` | `legacy`, `uncertainty`, or `fixed` |
| `--init_depth_rel_std` | `0.05` | Assumed depth standard deviation divided by depth |
| `--init_fixed_sigma` | `1.0` | Fixed-control Gaussian std in latent pixels |
| `--init_scales` | `0 0.5 1 2 4 8` | Increasing Gaussian std bank in latent pixels |

### 1. General configs
| Configuration | default |   Explanation  | 
|:------------- |:----- | :------------- |
| `--image_dir` | './test/images/fruit.png' | Image file path |
| `--out_dir` | './output' | Output directory |
| `--device` | 'cuda:0' | The device to use |
| `--exp_name` | None | Experiment name, use image file name by default |
### 2. Point cloud render configs
#### The definition of world coordinate system and tips for adjusting point cloud render configs are illustrated in [render document](./render_help.md).
| Configuration | default |   Explanation  | 
|:------------- |:----- | :------------- |
| `--mode` | 'single_view_txt' | Supported modes include `single_view_txt`, `single_view_target`, `single_view_autotraj`, and the iterative/evaluation modes implemented in `inference.py` |
| `--traj_txt` | None | Required for 'single_view_txt' mode, a txt file that specify camera trajectory |
| `--elevation` | 5. | The elevation angle of the input image in degree. Estimate a rough value based on your visual judgment |
| `--center_scale` | 1. | Scale factor for the spherical radius (r). By default, r is set to the depth value of the center pixel (H//2, W//2) of the reference image |
| `--d_theta` | 10. | Required for 'single_view_target' mode, specify target theta angle as (theta + d_theta) |
| `--d_phi` | 30. | Required for 'single_view_target' mode, specify target phi angle as (phi + d_phi) |
| `--d_r` | -.2 | Required for 'single_view_target' mode, specify target radius as (r + r*dr) |
| `--bg_trd` | 0.2 | Range from [0,1). Required for 'sparse_view_interp' mode, higher values will produce a cleaner point cloud but may create holes in the background |
### 3. Diffusion configs
| Configuration | default |   Explanation  | 
|:------------- |:----- | :------------- |
| `--ckpt_path` | './checkpoints/model.ckpt' | Checkpoint path |
| `--config` | './configs/inference_pvd_1024.yaml' | Config (yaml) path |
| `--ddim_steps` | 50 | Steps of ddim if positive, otherwise use DDPM, reduce to 10 to speed up inference |
| `--ddim_eta` | 1.0 | Eta for ddim sampling (0.0 yields deterministic sampling) |
| `--bs` | 1 | Batch size for inference, should be one |
| `--height` | 576 | Image height, in pixel space |
| `--width` | 1024 | Image width, in pixel space |
| `--frame_stride` | 10 | Fixed |
| `--unconditional_guidance_scale` | 7.5 | Prompt classifier-free guidance |
| `--seed` | 123 | Seed for seed_everything |
| `--video_length` | 16 | Inference video length; it must not exceed the checkpoint temporal length |
| `--negative_prompt` | False | Unused |
| `--text_input` | True | Unused |
| `--prompt` | 'Rotating view of a scene' | Fixed |
| `--multiple_cond_cfg` | False | Use multi-condition cfg or not |
| `--cfg_img` | None | Guidance scale for image conditioning |
| `--timestep_spacing` | "uniform_trailing" | The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information. |
| `--guidance_rescale` | 0.7 | Guidance rescale in [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) |
| `--perframe_ae` | True | If we use per-frame AE decoding, set it to True to save GPU memory, especially for the model of 576x1024 |
| `--n_samples` | 1 | Num of samples per prompt |

### 4. Geometry-verified evidence routing
The router is an inference-time controller. It keeps the rendered scaffold,
input views, camera trajectory, and diffusion checkpoint fixed, and decides how
strongly projected evidence may enter latent initialization and temporal
self-attention.

| Configuration | default | Explanation |
|:------------- |:----- |:------------- |
| `--use_evidence_routing` / `--disable_evidence_routing` | enabled | Enable the admit/attenuate/abstain verifier; disabling it recovers the overlap-only path |
| `--num_recent` | 8 | Number of preceding frames considered as evidence candidates |
| `--num_ref` | 3 | Maximum retained candidates after overlap filtering |
| `--overlap_threshold` | 0.1 | Minimum scaffold overlap for candidate retention |
| `--depth_rel_tolerance` | 0.08 | Relative target-camera depth agreement tolerance |
| `--depth_abs_tolerance` | 0.02 | Absolute depth margin for front-to-back checks |
| `--conflict_threshold` | 0.08 | Relative multi-reference depth deviation treated as conflict |
| `--min_support_ratio` | 0.02 | Minimum target-area support before routing is allowed |
| `--admit_ratio_threshold` | 0.20 | Admissible target-area ratio required for full admission |
| `--max_conflict_ratio` | 0.15 | Maximum conflict/occlusion-ordering risk for full admission |
| `--attenuated_weight` | 0.35 | Strength applied to usable but non-admissible evidence |
| `--abstain_gate` | 0.02 | Temporal-attention floor for unverified candidate messages |
| `--disable_fmi` | not set (enabled) | Ablation flag that disables evidence-gated frequency-mixed initialization |
| `--disable_cga` | not set (enabled) | Ablation flag that disables consistency-gated attention |
| `--save_evidence` | True | Save per-window evidence under the scene output directory; set to `false` to disable |
| `--geo_early_scale` / `--geo_late_scale` | 1.0 / 0.35 | Temporal-attention routing strength at high/low denoising noise |

After `run_diffusion`, `ViewCrafter.last_evidence_record` contains the
CPU-safe per-frame evidence ratios, `routing_mask`, discrete `region_state`,
frame gates, and reference indices for downstream diagnostics.
By default, each sampling call also saves an independent
`evidence/window_<index>_<unique-suffix>.pt` file under `save_dir`.
`ViewCrafter.evidence_record_paths` lists these files without retaining all
windows' tensors in memory. Each file contains the window index, sampling and
routing settings, camera records in window-local order, and evidence indexed by
sample and local camera index. Unique suffixes prevent overwriting earlier runs
in the same scene directory. Setting `--save_evidence false` preserves only the
latest in-memory record.

`occlusion_ratio` and `occlusion_mask` describe candidates behind the target's
first visible surface. The former names `free_space_ratio` and
`free_space_mask` remain as value-equivalent compatibility aliases.

With `--use_freq_mix false`, FMI replaces FFT mixing with spatial average
pooling but retains the same guide timestep and shared-noise forward diffusion
as the FFT branch. This corrects the earlier ablation path, which also skipped
guide noising. Results from that earlier path are not directly interchangeable
with this ablation; the default FFT path is unchanged.

## Simplified sampling-time observation guidance

Use `--observation_guidance` for the fixed-evidence, scale-aware sampling path. It bypasses FMI/CGA and reference ranking, and cannot be combined with `--observation_loop`. The old default is unchanged. See [observation_guidance.md](observation_guidance.md) for the method, options, compatibility limits and CPU tests.
