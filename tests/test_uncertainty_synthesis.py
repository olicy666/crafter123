"""Exercise the real synthesis call chain with lightweight model/sampler doubles."""
from types import SimpleNamespace

import pytest
import torch

from utils import diffusion_utils
from utils.frame_data import FrameData
from utils.warp_guidance import EvidenceRoutingEngine


@pytest.mark.parametrize('mode', ['legacy', 'fixed', 'uncertainty'])
def test_synthesis_exports_post_initialization_and_preserves_skipped_views(monkeypatch, mode):
    sampled_noise = []

    class Sampler:
        def __init__(self, model):
            pass

        def sample(self, **kwargs):
            sampled_noise.append(kwargs['x_T'].clone())
            return kwargs['x_T'], {}

    monkeypatch.setattr(diffusion_utils, 'DDIMSampler', Sampler)
    model = SimpleNamespace(
        device=torch.device('cpu'), num_timesteps=1000, scale_factor=1.,
        model=SimpleNamespace(conditioning_key='crossattn'),
        embedder=lambda x: torch.zeros(x.shape[0], 1, 4),
        image_proj_model=lambda x: x,
        get_learned_conditioning=lambda prompts: torch.zeros(len(prompts), 1, 4),
        decode_first_stage=lambda x: x,
        q_sample=lambda x_start, t, noise: .5*x_start + .5*noise,
    )
    encoder = SimpleNamespace(encode=lambda x: SimpleNamespace(mode=lambda: x))
    engine = EvidenceRoutingEngine(vae_encoder=encoder, device='cpu', init_scale_mode=mode,
                                   low_freq_norm=False)
    camera = {'K': torch.eye(3), 'R': torch.eye(3), 't': torch.zeros(3, 1)}
    depth = torch.ones(1, 1, 4, 4)
    rgb = torch.ones(1, 3, 4, 4)
    frames = [FrameData(rgb, rgb, depth, depth, camera) for _ in range(3)]
    shape = (1, 3, 3, 4, 4)
    output, records = diffusion_utils.image_guided_synthesis(
        model, [''], torch.ones(shape), shape, n_samples=2, condition_index=[1],
        warp_guidance=engine, frame_list=frames, return_evidence=True, seed=21,
    )
    assert output.shape == (1, 2, 3, 3, 4, 4)
    for sample in range(2):
        original = torch.randn(shape, generator=torch.Generator().manual_seed(21 + sample))
        torch.testing.assert_close(sampled_noise[sample][:, :, :2], original[:, :, :2])
        assert not torch.equal(sampled_noise[sample][:, :, 2], original[:, :, 2])
        assert records[sample][0]['initialization']['reason'] == 'no_references'
        assert records[sample][1]['initialization']['reason'] == 'conditioned_frame'
        assert records[sample][2]['initialization']['applied']
        assert records[sample][2]['initialization']['mode'] == mode
        if mode != 'legacy':
            assert 'scale_weights' in records[sample][2]['initialization']
