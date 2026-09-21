from types import SimpleNamespace

import pytest
import torch

from utils.uncertainty_guidance import (
    projection_covariance, latent_sigma, scale_residual, validate_scale_config,
)
from utils.warp_guidance import EvidenceRoutingEngine
from utils.frame_data import FrameData


def test_projection_matches_finite_difference_and_translation_formula():
    rays = torch.tensor([[[0.2], [0.1], [1.]]], dtype=torch.float64)
    depth = 2.
    translation = torch.tensor([[[0.4], [0.2], [0.]]], dtype=torch.float64)
    K = torch.diag(torch.tensor([100., 80., 1.], dtype=torch.float64))[None]
    std = torch.tensor([[[0.1]]], dtype=torch.float64)
    covariance = projection_covariance(rays * depth + translation, rays, K, std)

    def project(d):
        p = K @ (rays * d + translation)
        return p[:, :2] / p[:, 2:]

    j = (project(depth + 1e-5) - project(depth - 1e-5)) / 2e-5
    expected = torch.cat((j[:, :1]**2, j[:, :1]*j[:, 1:], j[:, 1:]**2), 1) * std**2
    torch.testing.assert_close(covariance, expected)
    assert covariance[0, 0, 0].item() == pytest.approx((100 * .4 / depth**2 * .1)**2)
    rotation = torch.tensor([[[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]], dtype=torch.float64)
    # Use a rotation preserving positive target depth.
    rotation = rotation.transpose(1, 2)
    rotated = rotation @ rays
    zero = projection_covariance(rotated * depth, rotated, K, std)
    torch.testing.assert_close(zero, torch.zeros_like(zero), atol=1e-20, rtol=0)


def test_covariance_resizes_in_pixel_units_without_nan_pollution():
    cov = torch.zeros(1, 3, 8, 16)
    cov[:, 0] = 16
    cov[:, 2] = 4
    mask = torch.ones(1, 1, 8, 16)
    assert torch.allclose(latent_sigma(cov, mask, (4, 4)), torch.ones(1, 1, 4, 4))
    cov[:, :, :4] = float('nan')
    sigma = latent_sigma(cov, mask, (4, 4))
    assert torch.isnan(sigma[:, :, :2]).all()
    assert torch.isfinite(sigma[:, :, 2:]).all()


def test_scale_filter_preserves_supported_constant_and_rejects_invalid():
    residual = torch.full((1, 3, 9, 9), float('nan'))
    support = torch.zeros(1, 1, 9, 9)
    support[:, :, 3:6, 3:6] = .35
    residual[:, :, 3:6, 3:6] = 2
    sigma = torch.ones_like(support)
    out, weights, bank = scale_residual(residual, support, sigma, (0., 1., 2.))
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[:, :, 3:6, 3:6], torch.full((1, 3, 3, 3), 2.))
    assert (out * (support == 0)).count_nonzero() == 0
    torch.testing.assert_close(bank.sum(1, keepdim=True), torch.ones_like(sigma))
    sigma.fill_(3)
    out, weights, bank = scale_residual(residual, support, sigma, (0., 1., 2.))
    assert out.count_nonzero() == weights.count_nonzero() == bank.count_nonzero() == 0


def test_increasing_scale_suppresses_fine_variation_and_interpolates_variance():
    x = ((torch.arange(17)[None, :] + torch.arange(17)[:, None]) % 2).float() * 2 - 1
    x = x[None, None]
    support = torch.ones_like(x)
    fine, _, _ = scale_residual(x, support, support * 0, (0., 1., 2.))
    coarse, _, _ = scale_residual(x, support, support, (0., 1., 2.))
    middle, _, weights = scale_residual(x, support, support * (0.5**0.5), (0., 1., 2.))
    assert coarse[:, :, 3:-3, 3:-3].abs().mean() < fine.abs().mean() * .01
    torch.testing.assert_close(middle, (fine + coarse) / 2)
    torch.testing.assert_close(weights[:, :2], torch.full_like(weights[:, :2], .5))


def test_zbuffer_keeps_covariance_from_visible_source():
    engine = EvidenceRoutingEngine(device='cpu', init_scale_mode='uncertainty')
    camera = {'K': torch.eye(3), 'R': torch.eye(3), 't': torch.zeros(3, 1)}
    target = dict(camera, t=torch.tensor([[2.], [0.], [0.]]))
    depth = torch.tensor([[[[2., 4.]]]])
    std = torch.tensor([[[[.2, .8]]]])
    rgb = torch.tensor([[[[1., 0.]], [[0., 1.]], [[0., 0.]]]])
    # Both round to target pixel x=1: 0+2/2=1 and 1+2/4=1.5 -> 2.
    # Move second ray to x=.4 so its projection rounds to one as well.
    source = dict(camera, K=torch.diag(torch.tensor([2.5, 1., 1.])))
    color, z, mask, info = engine.warp_rgb_depth(rgb, depth, source, target, 1, 3,
                                               source_depth_std=std, return_uncertainty=True)
    assert z[0, 0, 0, 1] == 2
    assert color[0, 0, 0, 1] == 1
    assert info['projection_covariance'][0, 0, 0, 1].item() == pytest.approx(.01)
    assert info['uncertainty_source'] == 'provided_depth_std'
    assert len(engine.warp_rgb_depth(rgb, depth, source, target, 1, 3)) == 3


def make_engine(mode):
    calls = []

    def encode(rgb):
        calls.append(True)
        return SimpleNamespace(mode=lambda: rgb)

    engine = EvidenceRoutingEngine(device='cpu', vae_encoder=SimpleNamespace(encode=encode),
                                   init_scale_mode=mode, low_freq_norm=False)
    return engine, calls


@pytest.mark.parametrize('mode', ['legacy', 'fixed', 'uncertainty'])
def test_complete_reference_to_initialization_and_cpu_record(mode):
    torch.manual_seed(12)
    engine, calls = make_engine(mode)
    camera = {'K': torch.eye(3), 'R': torch.eye(3), 't': torch.zeros(3, 1)}
    rgb = torch.rand(1, 3, 8, 8)
    depth = torch.ones(1, 1, 8, 8)
    frame = FrameData(rgb, rgb, depth, depth, camera)
    refs, warps = engine.select_reference_frames(1, [frame, frame])
    evidence = engine.build_evidence_state(1, [frame, frame], refs, warps)
    noise = torch.randn_like(rgb)
    result = engine.initialize_noise_with_fmi(1, noise, [frame, frame], refs, warps, evidence_state=evidence)
    assert torch.isfinite(result).all() and not torch.equal(result, noise)
    assert len(calls) == 1
    record = engine.export_evidence_state(evidence)['initialization']
    assert record['mode'] == mode and record['applied']
    assert record['effective_init_mask'].device.type == 'cpu'
    if mode == 'uncertainty':
        assert record['uncertainty_sources'] == ['relative_depth_assumption']
        torch.testing.assert_close(record['latent_sigma'], torch.zeros_like(record['latent_sigma']))


@pytest.mark.parametrize('missing', [True, False])
def test_uncertainty_abstains_before_encoding_missing_or_out_of_bank(missing):
    engine, calls = make_engine('uncertainty')
    noise = torch.randn(1, 3, 4, 4)
    info = {'warped_rgb': torch.ones_like(noise), 'warped_mask': torch.ones(1, 1, 4, 4)}
    if not missing:
        info['projection_covariance'] = torch.full((1, 3, 4, 4), 10000.)
    evidence = {'routing_enabled': False}
    result = engine.initialize_noise_with_fmi(1, noise, [], [0], {0: info}, evidence_state=evidence)
    assert torch.equal(result, noise) and not calls
    assert not evidence['initialization']['applied']


@pytest.mark.parametrize('relative, fixed, scales', [
    (-.1, 1., (0., 1.)), (float('nan'), 1., (0., 1.)),
    (.1, 2., (0., 1.)), (.1, 1., (0., 1., .5)), (.1, 1., (1., 2.)),
])
def test_invalid_configuration_fails_explicitly(relative, fixed, scales):
    with pytest.raises(ValueError):
        validate_scale_config('uncertainty', relative, fixed, scales)


@pytest.mark.parametrize('mode', ['legacy', 'fixed', 'uncertainty'])
def test_disabled_fmi_never_encodes_or_changes_noise(mode):
    engine, calls = make_engine(mode)
    engine.use_fmi = False
    noise = torch.randn(1, 3, 4, 4)
    evidence = {}
    assert torch.equal(engine.initialize_noise_with_fmi(1, noise, [], [0], {}, evidence_state=evidence), noise)
    assert not calls and evidence['initialization']['reason'] == 'fmi_disabled'


@pytest.mark.parametrize('use_freq_mix', [True, False])
def test_zero_scale_matches_legacy_with_multiple_references_and_soft_support(use_freq_mix):
    torch.manual_seed(19)
    legacy, _ = make_engine('legacy')
    adaptive, _ = make_engine('uncertainty')
    fixed, _ = make_engine('fixed')
    fixed.init_fixed_sigma = 0
    noise = torch.randn(2, 3, 8, 8)
    mask = torch.rand(2, 1, 8, 8)
    mask[:, :, :2] = 0
    warps = {i: {'warped_rgb': torch.rand_like(noise), 'warped_mask': mask,
                 'projection_covariance': torch.zeros(2, 3, 8, 8)} for i in range(2)}
    outputs = []
    for engine in (legacy, adaptive, fixed):
        engine.use_freq_mix = use_freq_mix
        outputs.append(engine.initialize_noise_with_fmi(2, noise, [], [0, 1], warps))
    for output in outputs[1:]:
        torch.testing.assert_close(output, outputs[0])
        assert torch.equal(output[:, :, :2], noise[:, :, :2])


def test_bad_external_depth_std_is_not_replaced_by_zero_uncertainty():
    engine, calls = make_engine('uncertainty')
    camera = {'K': torch.eye(3), 'R': torch.eye(3), 't': torch.zeros(3, 1)}
    rgb, depth = torch.ones(1, 3, 4, 4), torch.ones(1, 1, 4, 4)
    frame = FrameData(rgb, rgb, depth, depth, camera, depth_std=torch.full_like(depth, -1))
    refs, warps = engine.select_reference_frames(1, [frame, frame])
    evidence = engine.build_evidence_state(1, [frame, frame], refs, warps)
    noise = torch.randn_like(rgb)
    output = engine.initialize_noise_with_fmi(1, noise, [frame, frame], refs, warps, evidence_state=evidence)
    assert torch.equal(output, noise) and not calls


def test_vectorized_zbuffer_matches_scalar_oracle_with_ties_and_invalid_depth():
    engine = EvidenceRoutingEngine(device='cpu')
    camera = dict(K=torch.eye(3), R=torch.eye(3), t=torch.zeros(3, 1))
    source = dict(camera, K=torch.diag(torch.tensor([4., 1., 1.])))
    depth = torch.tensor([[[[2., 2., 1., float('nan'), -1., 3., 2., 1.]]],
                          [[[0., 3., 2., 1., 1., 2., 3., 4.]]]])
    rgb = torch.arange(48).reshape(2, 3, 1, 8).float() / 48
    out, z, mask = engine.warp_rgb_depth(rgb, depth, source, camera, 1, 4)
    expected_rgb = torch.zeros_like(out)
    expected_z = torch.zeros_like(z)
    for b in range(2):
        for x in range(8):
            d = depth[b, 0, 0, x]
            tx = round(x / 4)
            if torch.isfinite(d) and d > 0 and (expected_z[b, 0, 0, tx] == 0 or d < expected_z[b, 0, 0, tx]):
                expected_z[b, 0, 0, tx] = d
                expected_rgb[b, :, 0, tx] = rgb[b, :, 0, x]
    assert torch.equal(z, expected_z)
    assert torch.equal(out, expected_rgb)
    assert torch.equal(mask, (expected_z > 0).float())
