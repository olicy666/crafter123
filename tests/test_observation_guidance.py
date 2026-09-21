"""Exercise the simplified path with actual geometry and DDIM, without weights."""
import pytest
import torch

from configs.infer_config import get_parser
from utils.observation_guidance import ObservationGuidance, ObservationGuidanceConfig
from utils.observation_loop import LoopConfig
from utils.diffusion_utils import image_guided_synthesis
from test_observation_loop import fixture, TinyDiffusion, DDIMSampler, MultiSampler


def make_controller(strength=.35, observed=True):
    model, engine, frames, _, samples = fixture(observed=observed)
    config = ObservationGuidanceConfig(enabled=True, strength=strength, eval_size=8,
                                      scales=(0., 1., 2.), corrections=2)
    controller = ObservationGuidance(engine, frames, {0}, config)
    controller.set_schedule(4)
    return model, engine, frames, samples, controller


def test_fixed_evidence_correction_preserves_conditioned_frame_and_reduces_rgb_error():
    model, _, _, samples, controller = make_controller()
    before, _, _, _ = controller.evidence.evaluate(samples)
    support = controller.evidence.entries[1][0][1].clone()
    corrected = controller(model, samples, 0, torch.tensor([1]))
    after, _, _, _ = controller.evidence.evaluate(corrected)
    assert after[0, 1] < before[0, 1]
    assert torch.equal(corrected[:, :, 0], samples[:, :, 0])
    assert torch.equal(support, controller.evidence.entries[1][0][1])
    assert len(controller.records[1]['interventions']) == 1


def test_local_support_survives_global_abstention():
    _, engine, frames, _, _ = fixture()
    engine.min_support_ratio = 2.
    controller = ObservationGuidance(engine, frames, {0}, ObservationGuidanceConfig(enabled=True))
    assert 1 in controller.evidence.entries


@pytest.mark.parametrize('strength,observed', [(0., True), (.35, False)])
def test_zero_strength_or_no_observation_is_exact_noop(strength, observed):
    model, _, _, samples, controller = make_controller(strength, observed)
    model.decode_first_stage = lambda _: pytest.fail('No-op must not decode')
    assert controller(model, samples, 0, torch.tensor([1])) is samples


def test_unsupported_region_is_not_corrected():
    model, _, _, samples, controller = make_controller()
    rgb, support, sigma = controller.evidence.entries[1][0]
    support[:, :, :, 4:] = 0
    corrected = controller(model, samples, 0, torch.tensor([1]))
    assert torch.equal(corrected[:, :, 1, :, 4:], samples[:, :, 1, :, 4:])
    assert not torch.equal(corrected[:, :, 1, :, :4], samples[:, :, 1, :, :4])


def test_nonfinite_encoder_update_is_rejected_per_frame():
    model, engine, _, samples, controller = make_controller()
    engine._encode_guide_latent = lambda image: torch.full_like(image, float('nan'))
    result = controller(model, samples, 0, torch.tensor([1]))
    assert torch.equal(result, samples)
    assert not controller.records[1]['interventions'][0]['applied'].any()


@pytest.mark.parametrize('sampler_class', [DDIMSampler, MultiSampler])
@pytest.mark.parametrize('dynamic', [False, True])
@pytest.mark.parametrize('parameterization', ['eps', 'v'])
@pytest.mark.parametrize('eta', [0., 1.])
def test_sampler_hook_units_schedule_and_denoiser_kwargs(sampler_class, dynamic, parameterization, eta):
    model = TinyDiffusion(dynamic=dynamic, prediction=.5)
    model.parameterization = parameterization
    original_apply = model.apply_model

    def apply(*args, **kwargs):
        assert 'clean_prediction_corrector' not in kwargs
        return original_apply(*args, **kwargs)

    model.apply_model = apply

    class Corrector:
        def set_schedule(self, steps):
            assert steps == 4
            self.calls = []

        def should_correct(self, index):
            return index in (0, 1)

        def __call__(self, model, clean, index, timestep):
            torch.testing.assert_close(clean, torch.full_like(clean, .5), atol=2e-6, rtol=0)
            self.calls.append(index)
            return clean + .1

    corrector = Corrector()
    sampler = sampler_class(model)
    sample, _ = sampler.sample(S=4, batch_size=1, shape=(3, 2, 4, 4), eta=eta,
                               clean_prediction_corrector=corrector, verbose=False,
                               unconditional_conditioning_img_nonetext=None)
    assert corrector.calls == [1, 0]
    assert len(model.calls) == 4 and torch.isfinite(sample).all()


@pytest.mark.parametrize('multiple', [False, True])
@pytest.mark.parametrize('dynamic', [False, True])
def test_synthesis_bypasses_legacy_modules_and_exports_records(multiple, dynamic):
    model, engine, frames, samples, controller = make_controller()
    model.use_dynamic_rescale = dynamic
    model.prediction = -.5
    def forbidden(*args, **kwargs):
        pytest.fail('Simplified mode must bypass FMI/CGA and reference ranking')
    engine.select_reference_frames = forbidden
    engine.initialize_noise_with_fmi = forbidden
    engine.build_cga_bias = forbidden
    output, records = image_guided_synthesis(
        model, [''], samples, list(samples.shape), condition_index=[0], ddim_steps=4,
        ddim_eta=0., warp_guidance=engine, frame_list=frames, return_evidence=True,
        observation_guidance_config=controller.config, multiple_cond_cfg=multiple,
    )
    assert output.shape == (1, 1, 3, 2, 8, 8)
    assert len(model.calls) == 4
    assert len(records[0][1]['observation_guidance']['interventions']) == 2
    assert records[0][0]['observation_guidance']['reason'] == 'protected_frame'


def test_configs_and_exclusive_modes():
    options = get_parser().parse_args(['--observation_guidance', '--obs_corrections', '2'])
    config = ObservationGuidanceConfig.from_options(options)
    assert config.enabled and config.corrections == 2
    assert not ObservationGuidanceConfig.from_options(get_parser().parse_args([])).enabled
    for changes in ({'strength': 1.1}, {'corrections': 0}, {'eval_size': 0}, {'scales': (1., 2.)}):
        with pytest.raises(ValueError):
            ObservationGuidanceConfig(**changes)
    with pytest.raises(ValueError, match='Choose'):
        image_guided_synthesis(None, None, None, [1, 3, 2, 4, 4],
                               loop_config=LoopConfig(enabled=True),
                               observation_guidance_config=config)


@pytest.mark.parametrize('sampler_class', [DDIMSampler, MultiSampler])
def test_empty_controller_preserves_base_sampler_bitwise(sampler_class):
    model, _, _, samples, controller = make_controller(observed=False)
    sampler = sampler_class(model)
    start = torch.randn_like(samples)
    arguments = dict(S=4, batch_size=1, shape=samples.shape[1:], x_T=start,
                     eta=1., verbose=False, unconditional_conditioning_img_nonetext=None)
    torch.manual_seed(41)
    base, _ = sampler.sample(**arguments)
    torch.manual_seed(41)
    guided, _ = sampler.sample(**arguments, clean_prediction_corrector=controller)
    assert torch.equal(base, guided)


def test_batch_correction_and_external_original_pool():
    model, engine, frames, _, samples = fixture(batch=2)
    controller = ObservationGuidance(engine, [frames[1]], set(),
                                    ObservationGuidanceConfig(enabled=True, eval_size=8),
                                    observation_frames=[frames[0]])
    controller.set_schedule(1)
    clean = samples[:, :, 1:]
    result = controller(model, clean, 0, torch.tensor([1, 1]))
    assert result.shape == clean.shape
    assert (result > clean).all()
    assert controller.records[0]['observation_indices'] == [0]


@pytest.mark.parametrize('sampler_class', [DDIMSampler, MultiSampler])
def test_quantized_sampling_rejects_new_controller(sampler_class):
    model, _, _, samples, controller = make_controller()
    with pytest.raises(ValueError, match='unquantized DDIM'):
        sampler_class(model).sample(S=4, batch_size=1, shape=samples.shape[1:],
                                    clean_prediction_corrector=controller, quantize_x0=True)


def test_saved_evidence_identifies_new_mode(tmp_path):
    from utils.evidence_records import save_evidence_window
    options = get_parser().parse_args(['--observation_guidance'])
    path = save_evidence_window({}, tmp_path, 0, options)
    saved = torch.load(path, weights_only=True)
    assert saved['config']['obs_enabled'] is True
    assert saved['config']['obs_corrections'] == 3
