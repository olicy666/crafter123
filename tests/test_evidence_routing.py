import pytest

torch = pytest.importorskip("torch")

from utils.evidence_routing import (
    EVIDENCE_ABSTAIN,
    EVIDENCE_ATTENUATE,
    EVIDENCE_ADMIT,
    compute_geometry_evidence,
)


def _maps(*depth_values, height=4, width=4):
    depths = torch.tensor(depth_values, dtype=torch.float32).view(len(depth_values), 1, 1, 1, 1)
    depths = depths.expand(-1, 1, 1, height, width).clone()
    masks = torch.ones_like(depths)
    target_depth = torch.ones(1, 1, height, width)
    target_mask = torch.ones_like(target_depth)
    return depths, masks, target_depth, target_mask


def test_consistent_evidence_is_admitted():
    warped_depths, warped_masks, target_depth, target_mask = _maps(1.0, 1.005)
    result = compute_geometry_evidence(
        warped_depths, warped_masks, target_depth, target_mask
    )

    assert result["state"].item() == EVIDENCE_ADMIT
    assert result["conflict_ratio"].item() == 0.0
    assert result["geometry_mask"].mean().item() > 0.9
    assert result["routing_mask"].mean().item() > 0.9
    assert torch.all(result["region_state"] == EVIDENCE_ADMIT)
    assert result["frame_gate"].item() > 0.9
    assert result["per_reference_evidence"].shape == warped_depths.shape


def test_missing_support_abstains():
    warped_depths, warped_masks, target_depth, target_mask = _maps(1.0)
    warped_masks.zero_()
    result = compute_geometry_evidence(
        warped_depths, warped_masks, target_depth, target_mask
    )

    assert result["state"].item() == EVIDENCE_ABSTAIN
    assert result["support_ratio"].item() == 0.0
    assert result["frame_gate"].item() == 0.0
    assert result["routing_mask"].sum().item() == 0.0
    assert result["region_state"].sum().item() == 0


def test_partial_support_is_attenuated():
    warped_depths, warped_masks, target_depth, target_mask = _maps(1.0)
    warped_masks.zero_()
    warped_masks[0, 0, 0, 0, 0] = 1.0
    result = compute_geometry_evidence(
        warped_depths, warped_masks, target_depth, target_mask
    )

    assert result["state"].item() == EVIDENCE_ATTENUATE
    assert result["support_ratio"].item() > 0.02
    assert result["admissible_ratio"].item() < 0.20
    assert 0.0 < result["frame_gate"].item() < 1.0
    assert result["routing_mask"].mean().item() < result["geometry_mask"].mean().item()
    routed = result["routing_mask"] > 0
    assert routed.any()
    assert torch.all(result["region_state"][routed] == EVIDENCE_ATTENUATE)
    assert torch.any(result["region_state"] == EVIDENCE_ABSTAIN)


def test_conflicting_evidence_is_not_admitted():
    warped_depths, warped_masks, target_depth, target_mask = _maps(1.0, 1.5)
    result = compute_geometry_evidence(
        warped_depths,
        warped_masks,
        target_depth,
        target_mask,
        depth_rel_tolerance=0.1,
        conflict_threshold=0.1,
    )

    assert result["conflict_ratio"].item() > 0.9
    assert result["visibility_risk_ratio"].item() > 0.9
    assert result["geometry_mask"].mean().item() == 0.0
    assert result["routing_mask"].mean().item() == 0.0
    assert result["region_state"].sum().item() == 0
    assert result["state"].item() == EVIDENCE_ABSTAIN


def test_invalid_depth_values_do_not_contaminate_evidence():
    warped_depths, warped_masks, target_depth, target_mask = _maps(1.0, 1.0)
    warped_depths[0, 0, 0, 0, 0] = float("nan")
    target_depth[0, 0, 0, 0] = float("inf")

    result = compute_geometry_evidence(
        warped_depths, warped_masks, target_depth, target_mask
    )

    for key in ("geometry_mask", "per_reference_evidence", "frame_gate"):
        assert torch.isfinite(result[key]).all()


def test_array_like_inputs_are_normalized():
    warped_depths, warped_masks, target_depth, target_mask = _maps(1.0)
    result = compute_geometry_evidence(
        warped_depths.tolist(),
        warped_masks.tolist(),
        target_depth.tolist(),
        target_mask.tolist(),
    )

    assert result["state"].item() == EVIDENCE_ADMIT
    assert result["per_reference_evidence"].shape == warped_depths.shape


def test_temporal_gate_cannot_leak_into_spatial_attention():
    from lvdm.modules.attention import CrossAttention

    frame_bias = torch.full((1, 16, 16), -10.0)
    geo_bias = {
        "temporal_attention_only": True,
        "frame_bias": frame_bias,
        "temporal_length": 16,
    }
    sim = torch.zeros(1, 16, 16)

    spatial_attention = CrossAttention(query_dim=4, heads=1, dim_head=4)
    temporal_attention = CrossAttention(
        query_dim=4,
        heads=1,
        dim_head=4,
        temporal_length=16,
        temporal_attention=True,
    )

    spatial_result = spatial_attention._apply_geo_bias(
        sim, geo_bias, heads=1, spatial_self_attn=True
    )
    temporal_result = temporal_attention._apply_geo_bias(
        sim, geo_bias, heads=1, spatial_self_attn=True
    )

    assert torch.equal(spatial_result, sim)
    assert torch.equal(temporal_result, sim + frame_bias)


def test_pytorch3d_extrinsics_are_converted_to_rdf():
    from utils.warp_guidance import EvidenceRoutingEngine

    # An identity RDF camera is represented in PyTorch3D by flipping the
    # right/down axes into its left/up camera convention.
    p3d_rotation = torch.diag(torch.tensor([-1.0, -1.0, 1.0])).unsqueeze(0)
    p3d_translation = torch.zeros(1, 3, 1)
    rotation, translation = EvidenceRoutingEngine._pytorch3d_to_rdf(
        p3d_rotation, p3d_translation
    )

    assert torch.allclose(rotation, torch.eye(3).unsqueeze(0))
    assert torch.equal(translation, p3d_translation)


def test_pytorch3d_intrinsics_accept_scalar_and_single_axis_focal_lengths():
    from utils.warp_guidance import EvidenceRoutingEngine

    engine = EvidenceRoutingEngine(device='cpu')

    class Camera:
        K = None
        focal_length = torch.tensor([[2.0], [3.0]])
        principal_point = torch.tensor([[4.0], [5.0]])

    K = engine._pytorch3d_intrinsics(Camera(), batch_size=2)

    assert K.shape == (2, 3, 3)
    assert torch.equal(K[:, 0, 0], torch.tensor([2.0, 3.0]))
    assert torch.equal(K[:, 1, 1], torch.tensor([2.0, 3.0]))
    assert torch.equal(K[:, 0, 2], torch.tensor([4.0, 5.0]))
    assert torch.equal(K[:, 1, 2], torch.tensor([4.0, 5.0]))

    Camera.focal_length = torch.tensor(2.5)
    Camera.principal_point = torch.tensor([4.0, 5.0])
    scalar_K = engine._pytorch3d_intrinsics(Camera(), batch_size=2)
    assert torch.equal(scalar_K[:, 0, 0], torch.tensor([2.5, 2.5]))
    assert torch.equal(scalar_K[:, 1, 1], torch.tensor([2.5, 2.5]))


def test_frame_gates_become_log_biases_only_for_candidate_pairs():
    from utils.warp_guidance import EvidenceRoutingEngine

    engine = EvidenceRoutingEngine(device='cpu', abstain_gate=0.05)
    history = {
        2: {
            'ref_indices': [0],
            'per_ref_warp': {0: {}, 1: {}},
            'evidence_state': {
                'routing_enabled': True,
                'pair_gates': {0: torch.tensor([1.0])},
            },
        }
    }

    result = engine.build_cga_bias(history, total_frames=4, batch_size=1)
    bias = result['frame_bias'][0]

    assert torch.equal(bias[2, 0], torch.tensor(0.0))
    assert torch.allclose(bias[2, 1], torch.log(torch.tensor(0.05)))
    assert torch.equal(bias[2, 3], torch.tensor(0.0))


def test_nearest_valid_depth_ignores_empty_zbuffer_slots():
    from utils.warp_guidance import nearest_valid_depth

    zbuf = torch.tensor(
        [[[[4.0, -1.0, 7.0], [-1.0, -1.0, -1.0]]]],
        dtype=torch.float32,
    )
    depth = nearest_valid_depth(zbuf)

    assert torch.equal(depth, torch.tensor([[[4.0, 0.0]]]))


def test_occlusion_names_preserve_legacy_values():
    result = compute_geometry_evidence(*_maps(1.5))
    assert torch.equal(result['occlusion_ratio'], result['free_space_ratio'])
    assert torch.equal(result['occlusion_mask'], result['free_space_mask'])
    from utils.warp_guidance import EvidenceRoutingEngine
    exported = EvidenceRoutingEngine.export_evidence_state(result)
    assert torch.equal(exported['occlusion_ratio'], exported['free_space_ratio'])
    assert torch.equal(exported['occlusion_mask'], exported['free_space_mask'])


def _fmi_fixture(use_freq_mix):
    from types import SimpleNamespace
    from utils.warp_guidance import EvidenceRoutingEngine

    calls = []

    def encode(rgb):
        return SimpleNamespace(mode=lambda: rgb)

    def q_sample(x_start, t, noise):
        calls.append((x_start.clone(), t.clone(), noise.clone()))
        return 0.5 * x_start + 0.5 * noise

    engine = EvidenceRoutingEngine(
        vae_encoder=SimpleNamespace(encode=encode), device='cpu',
        use_freq_mix=use_freq_mix, low_freq_norm=False,
    )
    engine.diffusion_model = SimpleNamespace(
        num_timesteps=1000, scale_factor=1.0, q_sample=q_sample,
    )
    noise = torch.zeros(1, 3, 4, 4)
    mask = torch.ones(1, 1, 4, 4)
    warps = {0: {'warped_rgb': torch.ones_like(noise), 'warped_mask': mask}}
    evidence = {
        'routing_enabled': True, 'reference_indices': [0],
        'per_reference_evidence': mask.unsqueeze(0),
        'state_strength': torch.ones(1),
    }
    return engine, noise, warps, evidence, calls


@pytest.mark.parametrize('use_freq_mix', [True, False])
def test_fmi_abstention_returns_original_noise(use_freq_mix):
    engine, noise, warps, evidence, calls = _fmi_fixture(use_freq_mix)
    evidence['state_strength'].zero_()
    result = engine.initialize_noise_with_fmi(
        1, noise, [], [0], warps, evidence_state=evidence,
    )
    assert torch.equal(result, noise)
    assert not calls


@pytest.mark.parametrize('use_freq_mix', [True, False])
def test_fmi_attenuation_scales_residual_and_preserves_unrouted_pixels(use_freq_mix):
    engine, noise, warps, evidence, calls = _fmi_fixture(use_freq_mix)
    evidence['per_reference_evidence'][..., :2] = 0
    full = engine.initialize_noise_with_fmi(
        1, noise, [], [0], warps, evidence_state=evidence,
    )
    evidence['state_strength'].fill_(0.35)
    attenuated = engine.initialize_noise_with_fmi(
        1, noise, [], [0], warps, evidence_state=evidence,
    )
    assert torch.count_nonzero(full - noise) > 0
    assert torch.allclose(attenuated - noise, 0.35 * (full - noise))
    assert torch.equal(attenuated[..., :2], noise[..., :2])
    assert len(calls) == 2


@pytest.mark.parametrize('use_freq_mix', [True, False])
def test_fmi_both_mixers_use_the_same_noised_guide(use_freq_mix):
    from utils.warp_guidance import freq_mix_2d
    import torch.nn.functional as F

    engine, noise, warps, evidence, calls = _fmi_fixture(use_freq_mix)
    actual = engine.initialize_noise_with_fmi(
        1, noise, [], [0], warps, evidence_state=evidence,
    )
    assert len(calls) == 1
    guide, timestep, shared_noise = calls[0]
    assert torch.equal(timestep, torch.tensor([999]))
    assert torch.equal(shared_noise, noise)
    noised_guide = 0.5 * guide + 0.5 * noise
    if use_freq_mix:
        expected = freq_mix_2d(
            noised_guide, noise, engine.get_freq_filter(noise.shape),
            low_freq_norm=False, norm_factor=1.0,
        )
    else:
        expected = (
            F.avg_pool2d(noised_guide, 3, 1, 1)
            + noise - F.avg_pool2d(noise, 3, 1, 1)
        )
    assert torch.allclose(actual, expected)
