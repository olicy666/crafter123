import os
import argparse
import math


def _nonnegative_finite(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError('must be finite and nonnegative')
    return number


class _ScaleBankAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) < 2 or values[0] != 0 or any(a >= b for a, b in zip(values, values[1:])):
            parser.error(f'{self.dest} must start at zero and strictly increase')
        setattr(namespace, self.dest, values)

def _parse_bool(value):
    """Parse boolean CLI values without treating ``"False"`` as true."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--export_reconstruction', action='store_true',
                        help='Export the first sample with real RGB, target cameras and scale-aware reconstruction evidence')
    parser.add_argument('--observation_guidance', dest='obs_enabled', action='store_true',
                        help='Use fixed scale-aware observation correction within DDIM; bypass FMI/CGA and outer replay')
    parser.add_argument('--obs_strength', type=_nonnegative_finite, default=.35)
    parser.add_argument('--obs_corrections', type=int, default=3,
                        help='Corrections in the final half of the actual DDIM schedule, including the last update')
    parser.add_argument('--obs_eval_size', type=int, default=256)
    parser.add_argument('--obs_scales', type=_nonnegative_finite, nargs='+', action=_ScaleBankAction,
                        default=[0., 1., 2., 4., 8.], help='Gaussian std bank in evaluation pixels')
    parser.add_argument('--observation_loop', dest='loop_enabled', action='store_true',
                        help='Enable the complete frozen-model observation feedback loop and uncertainty initialization')
    parser.add_argument('--loop_max_rounds', type=int, default=2)
    parser.add_argument('--loop_resample_steps', type=int, default=10,
                        help='Number of low-noise steps replayed from the original DDIM schedule per round')
    parser.add_argument('--loop_strength', type=_nonnegative_finite, default=.35)
    parser.add_argument('--loop_min_improvement', type=_nonnegative_finite, default=1e-4)
    parser.add_argument('--loop_error_threshold', type=_nonnegative_finite, default=.01)
    parser.add_argument('--loop_eval_size', type=int, default=256,
                        help='Maximum image dimension for the fixed observation metric')
    parser.add_argument('--loop_scales', type=_nonnegative_finite, nargs='+', action=_ScaleBankAction,
                        default=[0., 1., 2., 4., 8.], help='Gaussian std bank in evaluation pixels')

    ## general
    parser.add_argument('--image_dir', type=str, default='./test/images/fruit.png', help='Image file path')
    parser.add_argument('--out_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to use')
    parser.add_argument('--exp_name',  type=str, default=None, help='Experiment name, use image file name by default')

    ## renderer
    parser.add_argument('--mode',  type=str,  default='single_view_txt', help="Supported modes include 'single_view_txt', 'single_view_target', 'single_view_autotraj', and iterative evaluation modes")
    parser.add_argument('--traj_txt',  type=str, help="Required for 'single_view_txt' mode, a txt file that specify camera trajectory")
    parser.add_argument('--elevation',  type=float, default=5., help='The elevation angle of the input image in degree. Estimate a rough value based on your visual judgment' )
    parser.add_argument('--center_scale',  type=float, default=1., help='Range: (0, 2]. Scale factor for the spherical radius (r). By default, r is set to the depth value of the center pixel (H//2, W//2) of the reference image')
    parser.add_argument('--d_theta', nargs='+', type=int, default=10., help="Range: [-40, 40]. Required for 'single_view_target' mode, specify target theta angle as theta + d_theta")
    parser.add_argument('--d_phi', nargs='+', type=int, default=30., help="Range: [-45, 45]. Required for 'single_view_target' mode, specify target phi angle as phi + d_phi")
    parser.add_argument('--d_r', nargs='+', type=float, default=-.2, help="Range: [-.5, .5]. Required for 'single_view_target' mode, specify target radius as r + r*dr")
    parser.add_argument('--d_x', nargs='+', type=float, default=0., help="Range: [-200, 200]. Required for 'single_view_target' mode, '+' denotes pan right")
    parser.add_argument('--d_y', nargs='+', type=float, default=0., help="Range: [-100, .100]. Required for 'single_view_target' mode, '+' denotes pan up")
    parser.add_argument('--planner_keyframes', type=int, default=7, help='Number of keyframes to plan for auto trajectory (2-25)')
    parser.add_argument('--planner_phi_max', type=float, default=45., help='Max absolute d_phi (deg) for planner candidates')
    parser.add_argument('--planner_theta_max', type=float, default=30., help='Max absolute d_theta (deg) for planner candidates')
    parser.add_argument('--planner_r_max', type=float, default=0.2, help='Max absolute d_r ratio for planner candidates')
    parser.add_argument('--planner_phi_step', type=float, default=5., help='Step size for phi candidate search (deg)')
    parser.add_argument('--planner_theta_step', type=float, default=5., help='Step size for theta candidate search (deg)')
    parser.add_argument('--planner_r_step', type=float, default=0.05, help='Step size for radius ratio candidate search')
    parser.add_argument('--planner_score', type=str, default='min_visible', help="Planner scoring: 'min_visible' or 'max_visible'")
    parser.add_argument('--planner_smooth_lambda', type=float, default=0.1, help='Smoothness weight on adjacent keyframes for planner')
    parser.add_argument('--planner_loop_back', type=_parse_bool, default=True, help='Whether planner should return to origin at the last keyframe')
    parser.add_argument('--mask_image', type=_parse_bool, default=False, help='Required for mulitpule reference images and iterative mode')
    parser.add_argument('--mask_pc',  type=_parse_bool, default=True, help='Required for mulitpule reference images and iterative mode')
    parser.add_argument('--reduce_pc', type=_parse_bool, default=False, help='Required for mulitpule reference images and iterative mode')
    parser.add_argument('--bg_trd',  type=float, default=0., help='Required for mulitpule reference images and iterative mode, set to 0. is no mask')
    parser.add_argument('--dpt_trd',  type=float, default=1., help='Required for mulitpule reference images and iterative mode, limit the max depth by * dpt_trd')


    ## diffusion
    parser.add_argument("--ckpt_path", type=str, default='./checkpoints/model.ckpt', help="checkpoint path")
    parser.add_argument("--config", type=str, default='./configs/inference_pvd_1024.yaml', help="config (yaml) path")
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM, reduce to 10 to speed up inference")
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)")
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference, should be one")
    parser.add_argument("--height", type=int, default=576, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=1024, help="image width, in pixel space")
    parser.add_argument("--frame_stride", type=int, default=10, help="Fixed")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=7.5, help="prompt classifier-free guidance")
    parser.add_argument("--seed", type=int, default=123, help="seed for seed_everything")
    parser.add_argument("--video_length", type=int, default=16, help="inference video length; must not exceed the checkpoint temporal length")
    parser.add_argument("--negative_prompt", type=_parse_bool, default=False, help="unused")
    parser.add_argument("--text_input", type=_parse_bool, default=True, help="unused")
    parser.add_argument("--prompt", type=str, default='Rotating view of a scene', help="Fixed")
    parser.add_argument("--multiple_cond_cfg", action='store_true', default=False, help="use multi-condition cfg or not")
    parser.add_argument("--cfg_img", type=float, default=None, help="guidance scale for image conditioning")
    parser.add_argument("--timestep_spacing", type=str, default="uniform_trailing", help="The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.")
    parser.add_argument("--guidance_rescale", type=float, default=0.7, help="guidance rescale in [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891)")
    parser.add_argument("--perframe_ae", type=_parse_bool, default=True, help="if we use per-frame AE decoding, set it to True to save GPU memory, especially for the model of 576x1024")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt")

    ## dust3r
    parser.add_argument('--model_path', type=str, default='./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth', help='The path of the model')
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--schedule', type=str, default='linear')
    parser.add_argument('--niter', default=300)
    parser.add_argument('--lr', default=0.01)
    parser.add_argument('--min_conf_thr', default=3.0) # minimum=1.0, maximum=20

    ## Frequency-preserving latent parameters
    parser.add_argument('--use_freq_mix', type=_parse_bool, default=True, help='Enable frequency-mixed initialization for coarse layout')
    parser.add_argument('--filter_type', type=str, default='gaussian', help="Type of low-pass filter: 'gaussian' or 'ideal'")
    parser.add_argument('--freq_cutoff', type=float, default=0.25, help='Normalized frequency cutoff (0.0-1.0), smaller = more blur, default 0.25')
    parser.add_argument('--low_freq_norm', type=_parse_bool, default=True, help='Normalize low-frequency components (key for stability)')
    parser.add_argument('--noise_level', type=int, default=999, help='Noise level for the guide latent in the forward diffusion process')
    parser.add_argument('--init_scale_mode', choices=['legacy', 'uncertainty', 'fixed'], default='legacy',
                        help='Scale selection for geometry-guided initialization; legacy preserves the baseline')
    parser.add_argument('--init_depth_rel_std', type=_nonnegative_finite, default=0.05,
                        help='Assumed depth std/depth when FrameData.depth_std is absent; not calibrated uncertainty')
    parser.add_argument('--init_fixed_sigma', type=_nonnegative_finite, default=1.0, help='Fixed-control std in latent pixels')
    parser.add_argument('--init_scales', type=_nonnegative_finite, nargs='+', action=_ScaleBankAction,
                        default=[0., 0.5, 1., 2., 4., 8.],
                        help='Increasing Gaussian std bank in latent pixels, starting at zero')

    ## Candidate evidence pool
    parser.add_argument('--num_recent', type=int, default=8,
                        help='Number of preceding frames considered as evidence candidates')
    parser.add_argument('--num_ref', type=int, default=3,
                        help='Maximum number of candidates retained after overlap filtering')
    parser.add_argument('--overlap_threshold', type=float, default=0.1,
                        help='Minimum scaffold overlap used for candidate selection')

    ## Geometry-verified evidence routing inside the diffusion process
    parser.add_argument('--save_evidence', type=_parse_bool, default=True,
                        help='Save per-window geometric evidence and sampling settings under save_dir/evidence')
    parser.add_argument('--use_evidence_routing', dest='use_evidence_routing', action='store_true', default=True,
                        help='Route only geometrically admissible evidence through FMI and temporal attention')
    parser.add_argument('--disable_evidence_routing', dest='use_evidence_routing', action='store_false',
                        help='Disable evidence routing and recover the legacy overlap-based guidance path')
    parser.add_argument('--disable_fmi', dest='use_fmi', action='store_false', default=True,
                        help='Disable evidence-gated frequency-mixed initialization for ablation')
    parser.add_argument('--disable_cga', dest='use_cga', action='store_false', default=True,
                        help='Disable consistency-gated temporal attention for ablation')
    parser.add_argument('--depth_rel_tolerance', type=float, default=0.08,
                        help='Relative target-camera depth tolerance for local geometric agreement')
    parser.add_argument('--depth_abs_tolerance', type=float, default=0.02,
                        help='Absolute target-camera depth tolerance for visibility ordering')
    parser.add_argument('--conflict_threshold', type=float, default=0.08,
                        help='Relative multi-reference depth deviation treated as conflict')
    parser.add_argument('--min_support_ratio', type=float, default=0.02,
                        help='Minimum target-area support required before evidence can be routed')
    parser.add_argument('--admit_ratio_threshold', type=float, default=0.20,
                        help='Admissible target-area ratio required for full evidence admission')
    parser.add_argument('--max_conflict_ratio', type=float, default=0.15,
                        help='Maximum conflicting target-area ratio for full admission')
    parser.add_argument('--attenuated_weight', type=float, default=0.35,
                        help='Routing strength for evidence that is usable but not fully admissible')
    parser.add_argument('--abstain_gate', type=float, default=0.02,
                        help='Attention gate floor for unverified temporal messages')
    parser.add_argument('--geo_early_scale', type=float, default=1.0,
                        help='Geometry routing strength at the high-noise/early denoising stage')
    parser.add_argument('--geo_late_scale', type=float, default=0.35,
                        help='Geometry routing strength at the low-noise/late denoising stage')

    return parser
