import importlib
import numpy as np
import cv2
import torch
import torch.distributed as dist
from collections import OrderedDict
import os
from lvdm.models.samplers.ddim import DDIMSampler
from lvdm.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond
from einops import rearrange, repeat

def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def check_istarget(name, para_list):
    """ 
    name: full name of source para
    para_list: partial name of target para 
    """
    istarget=False
    for para in para_list:
        if para in name:
            return True
    return istarget


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_npz_from_dir(data_dir):
    data = [np.load(os.path.join(data_dir, data_name))['arr_0'] for data_name in os.listdir(data_dir)]
    data = np.concatenate(data, axis=0)
    return data


def load_npz_from_paths(data_paths):
    data = [np.load(data_path)['arr_0'] for data_path in data_paths]
    data = np.concatenate(data, axis=0)
    return data   


def resize_numpy_image(image, max_resolution=512 * 512, resize_short_edge=None):
    h, w = image.shape[:2]
    if resize_short_edge is not None:
        k = resize_short_edge / min(h, w)
    else:
        k = max_resolution / (h * w)
        k = k**0.5
    h = int(np.round(h * k / 64)) * 64
    w = int(np.round(w * k / 64)) * 64
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image


def setup_dist(args):
    if dist.is_initialized():
        return
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://'
    )

def load_model_checkpoint(model, ckpt):
    state_dict = torch.load(ckpt, map_location="cpu")
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
        try:
            model.load_state_dict(state_dict, strict=True)
        except:
            ## rename the keys for 256x256 model
            new_pl_sd = OrderedDict()
            for k,v in state_dict.items():
                new_pl_sd[k] = v

            for k in list(new_pl_sd.keys()):
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding")
                    new_pl_sd[new_key] = new_pl_sd[k]
                    del new_pl_sd[k]
            model.load_state_dict(new_pl_sd, strict=True)
    else:
        # deepspeed
        new_pl_sd = OrderedDict()
        for key in state_dict['module'].keys():
            new_pl_sd[key[16:]]=state_dict['module'][key]
        model.load_state_dict(new_pl_sd)
    print('>>> model checkpoint loaded.')
    return model

def get_latent_z(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z

from .frame_data import FrameData

@torch.no_grad()
def image_guided_synthesis(model, prompts, videos, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1., \
                        unconditional_guidance_scale=1.0, cfg_img=None, fs=None, text_input=False, multiple_cond_cfg=False, timestep_spacing='uniform', guidance_rescale=0.0, condition_index=None, warp_guidance=None, frame_list=None, return_evidence=False, seed=42, loop_config=None, observation_frames=None, observation_guidance_config=None, **kwargs):
    from utils.observation_loop import LoopConfig, ObservationLoop
    loop_config = LoopConfig() if loop_config is None else loop_config
    from utils.observation_guidance import ObservationGuidanceConfig, ObservationGuidance
    obs_config = observation_guidance_config or ObservationGuidanceConfig()
    if obs_config.enabled and loop_config.enabled:
        raise ValueError('Choose sampling observation guidance or the legacy observation loop')
    if loop_config.enabled and (warp_guidance is None or not frame_list):
        raise ValueError('observation_loop requires a geometry router and frame_list')
    if loop_config.enabled and ddim_steps < 1:
        raise ValueError('observation_loop requires positive ddim_steps')
    if len(noise_shape) != 5:
        raise ValueError(f"noise_shape must be [B, C, T, H, W], got {noise_shape}")
    if obs_config.enabled and (warp_guidance is None or frame_list is None
                               or len(frame_list) != noise_shape[2] or ddim_steps < 1):
        raise ValueError('Observation guidance requires geometry, one FrameData per view, and positive ddim_steps')
    if not torch.is_tensor(videos) or videos.ndim != 5:
        raise ValueError(
            "videos must be a tensor with shape [B, 3, T, H, W], "
            f"got {type(videos).__name__} with shape "
            f"{getattr(videos, 'shape', None)}"
        )
    if videos.shape[0] != noise_shape[0]:
        raise ValueError(
            "videos and noise_shape must have the same batch size, "
            f"got {videos.shape[0]} and {noise_shape[0]}"
        )
    if videos.shape[2] != noise_shape[2]:
        raise ValueError(
            "videos and noise_shape must have the same temporal length, "
            f"got {videos.shape[2]} and {noise_shape[2]}"
        )
    if videos.shape[1] != 3:
        raise ValueError(
            "videos must contain RGB channels in dimension 1, "
            f"got {videos.shape[1]}"
        )
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if condition_index is None or len(condition_index) == 0:
        raise ValueError("condition_index must contain at least one frame index")
    condition_index = [int(index) for index in condition_index]
    if any(index < 0 or index >= noise_shape[2] for index in condition_index):
        raise ValueError(
            "condition_index must refer to frames inside noise_shape[2], "
            f"got {condition_index} for {noise_shape[2]} frames"
        )

    ddim_sampler = DDIMSampler(model) if not multiple_cond_cfg else DDIMSampler_multicond(model)
    batch_size = noise_shape[0]
    if fs is None:
        fs = torch.zeros(batch_size, dtype=torch.long, device=model.device)
    elif torch.is_tensor(fs):
        fs = fs.to(device=model.device, dtype=torch.long).reshape(-1)
        if fs.numel() == 1:
            fs = fs.expand(batch_size)
        elif fs.numel() != batch_size:
            raise ValueError(f"fs must contain one value or one value per batch item, got {fs.numel()}")
    else:
        fs = torch.full((batch_size,), int(fs), dtype=torch.long, device=model.device)

    if not text_input or prompts is None:
        prompts = [""]*batch_size
    elif isinstance(prompts, str):
        prompts = [prompts]
    else:
        prompts = list(prompts)
        if len(prompts) == 1 and batch_size > 1:
            prompts = prompts * batch_size
    if len(prompts) != batch_size:
        raise ValueError(
            f"prompts must contain one string or one string per batch item, "
            f"got {len(prompts)} for batch size {batch_size}"
        )
    assert condition_index is not None, "Error: condition index is None!"

    img = videos[:,:,condition_index[0]] #bchw
    img_emb = model.embedder(img) ## blc
    img_emb = model.image_proj_model(img_emb)

    cond_emb = model.get_learned_conditioning(prompts)
    cond = {"c_crossattn": [torch.cat([cond_emb,img_emb], dim=1)]}
    if model.model.conditioning_key == 'hybrid':
        z = get_latent_z(model, videos) # b c t h w
        # if loop or interp:
        #     img_cat_cond = torch.zeros_like(z)
        #     img_cat_cond[:,:,0,:,:] = z[:,:,0,:,:]
        #     img_cat_cond[:,:,-1,:,:] = z[:,:,-1,:,:]
        # else:
        img_cat_cond = z
        cond["c_concat"] = [img_cat_cond] # b c 1 h w
    
    if unconditional_guidance_scale != 1.0:
        if model.uncond_type == "empty_seq":
            prompts = batch_size * [""]
            uc_emb = model.get_learned_conditioning(prompts)
        elif model.uncond_type == "zero_embed":
            uc_emb = torch.zeros_like(cond_emb)
        else:
            raise ValueError(
                "multiple-condition CFG requires model.uncond_type to be "
                "'empty_seq' or 'zero_embed'"
            )
        uc_img_emb = model.embedder(torch.zeros_like(img)) ## b l c
        uc_img_emb = model.image_proj_model(uc_img_emb)
        uc = {"c_crossattn": [torch.cat([uc_emb,uc_img_emb],dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc["c_concat"] = [img_cat_cond]
    else:
        uc = None

    # The multi-condition sampler always evaluates a third branch whenever
    # classifier-free guidance is active.  Construct it independently of the
    # image guidance scale; otherwise ``cfg_img=None`` or ``cfg_img=1`` would
    # leave the sampler with a missing conditioning object.
    effective_cfg_img = (
        unconditional_guidance_scale if cfg_img is None else cfg_img
    )
    if multiple_cond_cfg and unconditional_guidance_scale != 1.0:
        uc_2 = {"c_crossattn": [torch.cat([uc_emb,img_emb],dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc_2["c_concat"] = [img_cat_cond]
        kwargs.update({"unconditional_conditioning_img_nonetext": uc_2})
    else:
        kwargs.update({"unconditional_conditioning_img_nonetext": None})

    z0 = None
    cond_mask = None

    batch_variants = []
    evidence_record = {}
    for sample_idx in range(n_samples):

        if z0 is not None:
            cond_z0 = z0.clone()
            kwargs.update({"clean_cond": True})
        else:
            cond_z0 = None
        if ddim_sampler is not None:
            condition_set = set(condition_index or [])
            
            # Use a shared seed so the evidence-controlled latent trajectory
            # remains comparable across the generated frames.
            base_seed = 42 if seed is None else int(seed)
            noise_seed = base_seed + sample_idx  # Different seed per sample, but consistent across frames
            # DDIM's stochastic updates draw from the global torch RNG rather
            # than the local generator used for x_T.  Reset it here so paired
            # Base/router runs with the same seed differ only in the intended
            # evidence-routing intervention.
            torch.manual_seed(noise_seed)
            generator = torch.Generator(device=model.device).manual_seed(noise_seed)
            x_T = torch.randn(
                tuple(noise_shape), generator=generator, device=model.device
            )
            geo_bias = None
            selection_history = {}

            controller = None
            if obs_config.enabled:
                warp_guidance.set_diffusion_model(model)
                controller = ObservationGuidance(warp_guidance, frame_list, condition_set,
                                                 obs_config, observation_frames)

            if not obs_config.enabled and warp_guidance is not None and frame_list is not None and len(frame_list) > 0:
                # Set diffusion model reference for q_sample operations
                warp_guidance.set_diffusion_model(model)
                warp_guidance.shared_noise_seed = noise_seed

                n_frames = min(noise_shape[2], len(frame_list))

                for t in range(n_frames):
                    ref_indices, per_ref_warp = warp_guidance.select_reference_frames(t, frame_list)
                    evidence_state = warp_guidance.build_evidence_state(
                        t, frame_list, ref_indices, per_ref_warp
                    )
                    selection_history[t] = {
                        'ref_indices': ref_indices,
                        'per_ref_warp': per_ref_warp,
                        'evidence_state': evidence_state,
                    }
                    if t in condition_set or not ref_indices:
                        evidence_state['initialization'] = {
                            'mode': getattr(warp_guidance, 'init_scale_mode', 'legacy'),
                            'applied': False,
                            'reason': 'conditioned_frame' if t in condition_set else 'no_references',
                        }
                        evidence_record.setdefault(sample_idx, {})[t] = \
                            warp_guidance.export_evidence_state(evidence_state)
                        continue

                    updated_noise = warp_guidance.initialize_noise_with_fmi(
                        t,
                        x_T[:, :, t],
                        frame_list,
                        ref_indices,
                        per_ref_warp,
                        evidence_state=evidence_state,
                    )
                    x_T[:, :, t] = updated_noise
                    evidence_record.setdefault(sample_idx, {})[t] = \
                        warp_guidance.export_evidence_state(evidence_state)

                geo_bias = warp_guidance.build_cga_bias(
                    selection_history=selection_history,
                    total_frames=noise_shape[2],
                    batch_size=noise_shape[0]
                )
                if geo_bias is not None:
                    geo_bias['num_diffusion_steps'] = int(getattr(model, 'num_timesteps', 1000))

            sampler_kwargs = dict(kwargs)
            if controller is not None:
                if sampler_kwargs.get('geo_bias') is not None:
                    raise ValueError('Sampling observation guidance cannot be combined with CGA')
                sampler_kwargs['clean_prediction_corrector'] = controller
            if geo_bias is not None:
                sampler_kwargs.update({"geo_bias": geo_bias})

            sample_arguments = dict(sampler_kwargs)
            sample_arguments.update(
                S=ddim_steps, conditioning=cond, batch_size=batch_size, shape=noise_shape[1:],
                verbose=False, unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=uc, eta=ddim_eta, cfg_img=effective_cfg_img,
                mask=cond_mask, x0=cond_z0, x_T=x_T, fs=fs,
                timestep_spacing=timestep_spacing, guidance_rescale=guidance_rescale,
            )
            samples, _ = ddim_sampler.sample(**sample_arguments)
            if controller is not None:
                for frame_index, record in controller.records.items():
                    evidence_record.setdefault(sample_idx, {}).setdefault(frame_index, {})['observation_guidance'] = record

        ## reconstruct from latent to pixel space
        batch_images = model.decode_first_stage(samples)
        if loop_config.enabled:
            controller = ObservationLoop(warp_guidance, frame_list[:noise_shape[2]],
                                         condition_set, loop_config, observation_frames)
            samples, batch_images, loop_records = controller.run(
                model, ddim_sampler, samples, batch_images, sample_arguments, noise_seed,
            )
            for frame_index, record in loop_records.items():
                evidence_record.setdefault(sample_idx, {}).setdefault(frame_index, {})['observation_loop'] = record
        batch_variants.append(batch_images)
    ## variants, batch, c, t, h, w
    batch_variants = torch.stack(batch_variants)
    result = batch_variants.permute(1, 0, 2, 3, 4, 5)
    if return_evidence:
        return result, evidence_record
    return result
