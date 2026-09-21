from types import SimpleNamespace

import pytest
import torch

from utils.frame_data import FrameData
from utils.observation_loop import LoopConfig, ObservationLoop, noise_at_schedule
from utils.warp_guidance import EvidenceRoutingEngine
from lvdm.models.samplers.ddim import DDIMSampler
from lvdm.models.samplers.ddim_multiplecond import DDIMSampler as MultiSampler


class TinyDiffusion:
    """Analytic denoiser used with the actual DDIM update equations."""
    def __init__(self, dynamic=False, prediction=.5):
        self.device = torch.device('cpu')
        self.num_timesteps = 20
        self.betas = torch.linspace(.0001, .02, self.num_timesteps)
        self.alphas_cumprod = (1 - self.betas).cumprod(0)
        self.alphas_cumprod_prev = torch.cat((torch.ones(1), self.alphas_cumprod[:-1]))
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod).sqrt()
        self.scale_arr = torch.linspace(1., .7, 20)
        self.use_dynamic_rescale = dynamic
        self.parameterization = 'eps'
        self.prediction = prediction
        self.scale_factor = 1.
        self.calls = []
        self.model = SimpleNamespace(conditioning_key='crossattn')
        self.uncond_type = 'empty_seq'

    def q_sample(self, x_start, t, noise):
        alpha = self.alphas_cumprod[t].view(-1, *([1] * (x_start.ndim - 1)))
        return alpha.sqrt()*x_start + (1-alpha).sqrt()*noise

    def apply_model(self, x, t, c, **kwargs):
        assert not any(k.startswith('replay_') or k == 'resume_steps' for k in kwargs)
        self.calls.append((int(t[0]), x.clone()))
        alpha = self.alphas_cumprod[t].view(-1, *([1] * (x.ndim - 1)))
        clean = torch.full_like(x, self.prediction)
        if self.use_dynamic_rescale:
            clean *= self.scale_arr[t].view(-1, *([1] * (x.ndim - 1)))
        epsilon = (x-alpha.sqrt()*clean)/(1-alpha).sqrt()
        return (epsilon - (1-alpha).sqrt()*x) / alpha.sqrt() if self.parameterization == 'v' else epsilon

    def predict_start_from_z_and_v(self, x, t, v):
        alpha = self.alphas_cumprod[t].view(-1, *([1] * (x.ndim - 1)))
        return alpha.sqrt()*x - (1-alpha).sqrt()*v

    def predict_eps_from_z_and_v(self, x, t, v):
        alpha = self.alphas_cumprod[t].view(-1, *([1] * (x.ndim - 1)))
        return alpha.sqrt()*v + (1-alpha).sqrt()*x

    def decode_first_stage(self, x):
        return x

    def embedder(self, x):
        return torch.zeros(x.shape[0], 1, 4)

    def image_proj_model(self, x):
        return x

    def get_learned_conditioning(self, prompts):
        return torch.zeros(len(prompts), 1, 4)


def fixture(batch=1, dynamic=False, prediction=.5, observed=True, config=None):
    model = TinyDiffusion(dynamic, prediction)
    engine = EvidenceRoutingEngine(device='cpu', vae_encoder=SimpleNamespace(
        encode=lambda x: SimpleNamespace(mode=lambda: x)), init_scale_mode='uncertainty')
    engine.set_diffusion_model(model)
    depth = torch.ones(batch, 1, 8, 8)
    rgb = torch.full((batch, 3, 8, 8), .75)
    camera = dict(K=torch.eye(3), R=torch.eye(3), t=torch.zeros(3, 1))
    frames = [FrameData(rgb, rgb, depth, depth, camera, is_observation=observed),
              FrameData(rgb, rgb, depth, depth, camera)]
    config = config or LoopConfig(enabled=True, resample_steps=2, eval_size=8, scales=(0., 1., 2.))
    controller = ObservationLoop(engine, frames, {0}, config)
    samples = torch.full((batch, 3, 2, 8, 8), -.5)
    samples[:, :, 0] = .5
    return model, engine, frames, controller, samples


@pytest.mark.parametrize('sampler_class', [DDIMSampler, MultiSampler])
@pytest.mark.parametrize('dynamic', [False, True])
@pytest.mark.parametrize('eta', [0., 1.])
@pytest.mark.parametrize('parameterization', ['eps', 'v'])
def test_real_ddim_replay_exact_steps_and_reference_trajectory(sampler_class, dynamic, eta, parameterization):
    model = TinyDiffusion(dynamic)
    model.parameterization = parameterization
    sampler = sampler_class(model)
    sampler.make_schedule(4, ddim_eta=eta, verbose=False)
    reference = torch.zeros(1, 3, 2, 4, 4)
    reference[:, :, 0] = -.2
    mask = torch.zeros(1, 1, 2, 4, 4)
    mask[:, :, 0] = 1
    noise = torch.randn_like(reference)
    start = noise_at_schedule(model, reference, noise, sampler.ddim_timesteps[1])
    result, _ = sampler.sample(S=4, batch_size=1, shape=reference.shape[1:], x_T=start,
                              resume_steps=2, eta=eta, replay_reference=reference,
                              replay_mask=mask, replay_noise=noise, verbose=False,
                              unconditional_conditioning_img_nonetext=None)
    assert [t for t, _ in model.calls] == [6, 1]
    for step, x in model.calls:
        expected = noise_at_schedule(model, reference, noise, step)
        torch.testing.assert_close(x[:, :, 0], expected[:, :, 0])
    assert torch.equal(result[:, :, 0], reference[:, :, 0])


@pytest.mark.parametrize('sampler_class', [DDIMSampler, MultiSampler])
def test_closed_loop_accepts_real_sampler_improvement(sampler_class):
    model, _, _, controller, samples = fixture()
    sampler = sampler_class(model)
    sampler.make_schedule(4, verbose=False)
    arguments = dict(S=4, batch_size=1, shape=samples.shape[1:], eta=0., verbose=False,
                     unconditional_conditioning_img_nonetext=None)
    result, images, records = controller.run(model, sampler, samples, samples.clone(), arguments, 42)
    assert records[1]['rounds'][0]['accepted'].item()
    assert records[1]['final_error'].item() < records[1]['initial_error'].item()
    assert torch.equal(images[:, :, 0], samples[:, :, 0])
    assert torch.equal(result[:, :, 0], samples[:, :, 0])
    assert records[1]['attempted_rounds'] <= 2


@pytest.mark.parametrize('prediction', [-1., float('nan')])
def test_rejection_restores_exact_previous_result(prediction):
    model, _, _, controller, samples = fixture(prediction=prediction)
    sampler = DDIMSampler(model)
    sampler.make_schedule(4, verbose=False)
    result, images, records = controller.run(model, sampler, samples, samples.clone(),
        dict(S=4, batch_size=1, shape=samples.shape[1:], eta=0., verbose=False), 1)
    assert torch.equal(result, samples) and torch.equal(images, samples)
    assert not records[1]['rounds'][0]['accepted'].item()
    assert records[1]['attempted_rounds'] == 1


def test_generated_reference_is_not_an_observation():
    model, _, _, controller, samples = fixture(observed=False)
    result, images, records = controller.run(model, None, samples, samples, {}, 1)
    assert not controller.entries and not model.calls
    assert result is samples and images is samples
    assert records[1]['reason'] == 'no_observations'


def test_window_can_use_external_observation_and_excludes_invalid_uncertainty():
    _, engine, frames, _, _ = fixture()
    config = LoopConfig(enabled=True, eval_size=8)
    controller = ObservationLoop(engine, [frames[1]], set(), config, observation_frames=[frames[0]])
    assert 0 in controller.entries
    frames[0].depth_std = torch.full_like(frames[0].depth, float('nan'))
    invalid = ObservationLoop(engine, [frames[1]], set(), config, observation_frames=[frames[0]])
    assert not invalid.entries
    assert invalid.records[0]['reason'] == 'no_reliable_support'


def test_fixed_metric_detects_error_and_recomputes_correction_without_changing_support():
    _, _, _, controller, samples = fixture()
    initial, present, delta, support = controller.evaluate(samples)
    improved = samples.clone()
    improved[:, :, 1] = .3
    later, _, new_delta, new_support = controller.evaluate(improved)
    assert (later[present] < initial[present]).all()
    assert new_delta.abs().sum() < delta.abs().sum()
    assert torch.equal(support, new_support)


def test_threshold_exits_without_model_calls_and_rng_is_restored():
    model, _, _, controller, samples = fixture()
    sampler = DDIMSampler(model)
    sampler.make_schedule(4, verbose=False)
    matching = torch.full_like(samples, .5)
    _, _, records = controller.run(model, sampler, matching, matching, {}, 5)
    assert not model.calls and records[1]['stop_reasons'] == ['error_threshold']
    rng = torch.random.get_rng_state().clone()
    controller.run(model, sampler, samples, samples,
                   dict(S=4, batch_size=1, shape=samples.shape[1:], eta=1., verbose=False), 5)
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize('kwargs', [dict(max_rounds=0), dict(resample_steps=-1),
    dict(strength=2), dict(min_improvement=0), dict(eval_size=0), dict(scales=(0., 2., 1.))])
def test_loop_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        LoopConfig(**kwargs)


@pytest.mark.parametrize('count', [0, -1, 5, True])
def test_invalid_resume_count(count):
    model = TinyDiffusion()
    with pytest.raises(ValueError, match='resume_steps'):
        DDIMSampler(model).sample(S=4, batch_size=1, shape=(3, 2, 4, 4),
                                 resume_steps=count, verbose=False)


@pytest.mark.parametrize('multicond', [False, True])
def test_synthesis_with_actual_sampler_exports_loop_and_cfg(multicond):
    from utils.diffusion_utils import image_guided_synthesis
    model, engine, frames, controller, samples = fixture()
    output, records = image_guided_synthesis(
        model, [''], samples, samples.shape, ddim_steps=4, ddim_eta=0.,
        condition_index=[0], warp_guidance=engine, frame_list=frames,
        multiple_cond_cfg=multicond, unconditional_guidance_scale=2.,
        return_evidence=True, loop_config=controller.config,
    )
    assert output.shape == (1, 1, 3, 2, 8, 8)
    record = records[0][1]['observation_loop']
    assert record['enabled'] and record['reason'] == 'finished'
    assert record['observation_indices'] == [0]
    assert record['final_error'].item() <= record['initial_error'].item()


def test_batch_independent_acceptance_and_exact_hole_preservation():
    model, engine, frames, _, samples = fixture(batch=2)
    frames[0].mask[:, :, :3] = 0
    controller = ObservationLoop(engine, frames, {0}, LoopConfig(enabled=True, resample_steps=2, eval_size=8))
    # First sample is already correct; only the second is allowed to change.
    samples[0] = .5
    sampler = DDIMSampler(model)
    sampler.make_schedule(4, verbose=False)
    result, images, records = controller.run(model, sampler, samples, samples.clone(),
        dict(S=4, batch_size=2, shape=samples.shape[1:], eta=0., verbose=False), 2)
    assert torch.equal(result[0], samples[0]) and torch.equal(images[0], samples[0])
    assert torch.equal(images[1, :, 1, :3], samples[1, :, 1, :3])
    assert torch.equal(result[1, :, 1, :3], samples[1, :, 1, :3])
    assert records[1]['rounds'][0]['accepted'].tolist() == [False, True]
    assert records[1]['stop_reasons'][0] == 'error_threshold'


def test_loop_evidence_roundtrip(tmp_path):
    from utils.evidence_records import save_evidence_window
    model, _, frames, controller, samples = fixture()
    sampler = DDIMSampler(model)
    sampler.make_schedule(4, verbose=False)
    _, _, records = controller.run(model, sampler, samples, samples.clone(),
        dict(S=4, batch_size=1, shape=samples.shape[1:], eta=0., verbose=False), 2)
    record = {0: {i: {'observation_loop': value} for i, value in records.items()}}
    path = save_evidence_window(record, tmp_path, 0, SimpleNamespace(loop_enabled=True), frames)
    restored = torch.load(path, weights_only=True)
    loop = restored['samples'][0][1]['observation_loop']
    assert loop['rounds'][0]['accepted'].item()
    assert loop['config']['strength'] == .35
    assert loop['extra_ddim_steps'] > 0
    assert restored['is_observation'] == [True, False]


def test_loop_disabled_is_identical_to_legacy_invocation():
    from utils.diffusion_utils import image_guided_synthesis
    model, engine, frames, _, samples = fixture()
    arguments = dict(ddim_steps=4, ddim_eta=0., condition_index=[0],
                     warp_guidance=engine, frame_list=frames, return_evidence=True, seed=19)
    original, _ = image_guided_synthesis(model, [''], samples, samples.shape, **arguments)
    disabled, records = image_guided_synthesis(model, [''], samples, samples.shape,
                                               loop_config=LoopConfig(enabled=False), **arguments)
    assert torch.equal(original, disabled)
    assert all('observation_loop' not in record for record in records[0].values())


def test_no_mutable_latent_cells_exits_without_proposal():
    model, engine, frames, _, samples = fixture()
    frames[0].mask.zero_()
    frames[0].mask[:, :, 1, 1] = 1
    # Permit the tiny observed area through the geometry admission threshold.
    engine.min_support_ratio = 0
    controller = ObservationLoop(engine, frames, {0}, LoopConfig(enabled=True, eval_size=8))
    assert controller.entries
    small = torch.zeros(1, 3, 2, 1, 1)
    engine.vae = SimpleNamespace(encode=lambda x: SimpleNamespace(
        mode=lambda: torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))))
    sampler = DDIMSampler(model)
    sampler.make_schedule(4, verbose=False)
    _, _, records = controller.run(model, sampler, small, samples, {}, 2)
    assert not model.calls
    assert records[1]['stop_reasons'] == ['no_supported_latent_cells']
