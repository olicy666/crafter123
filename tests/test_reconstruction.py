from types import SimpleNamespace

import pytest
import torch

from utils.reconstruction import auxiliary_losses, camera_matrices, reconstruction_loss, seed_points
from utils.reconstruction_data import save_reconstruction_window
from test_observation_loop import fixture


def reference(weight=1., sigma=0., rgb=0.):
    return dict(rgb=torch.full((1,3,8,8), rgb), support=torch.full((1,1,8,8), weight),
                sigma=torch.full((1,1,8,8), sigma))


def test_full_support_excludes_generated_texture_and_keeps_real_gradient():
    pred = torch.ones(1,3,8,8, requires_grad=True)
    generated = torch.full_like(pred, .5, requires_grad=True)
    evidence, completion = auxiliary_losses(pred, generated, [reference()], (0.,1.,2.))
    assert evidence.item() == 1 and completion.item() == 0
    (evidence + completion).backward()
    assert (pred.grad > 0).all()
    assert generated.grad is None


def test_no_support_uses_completion_without_fabricating_evidence():
    pred = torch.ones(1,3,8,8, requires_grad=True)
    evidence, completion = auxiliary_losses(pred, torch.zeros_like(pred), [], (0.,1.,2.))
    assert evidence.item() == 0 and completion.item() == 1
    completion.backward()
    assert torch.isfinite(pred.grad).all()


def test_weak_support_preserves_absolute_strength_and_complement():
    pred = torch.ones(1,3,8,8)
    evidence, completion = auxiliary_losses(pred, torch.zeros_like(pred), [reference(.25)], (0.,1.,2.))
    assert evidence.item() == .25 and completion.item() == .75


def test_reference_duplication_does_not_multiply_evidence():
    pred = torch.ones(1,3,8,8)
    one = auxiliary_losses(pred, torch.zeros_like(pred), [reference(.4)], (0.,1.,2.))
    two = auxiliary_losses(pred, torch.zeros_like(pred), [reference(.4),reference(.4)], (0.,1.,2.))
    for a,b in zip(one,two):
        torch.testing.assert_close(a,b)


def test_incompatible_reference_residuals_do_not_cancel():
    pred = torch.full((1,3,8,8), .5)
    evidence, _ = auxiliary_losses(pred, pred, [reference(rgb=0.), reference(rgb=1.)], (0.,1.,2.))
    assert evidence.item() == .5


def test_coarse_scale_attenuates_high_frequency_alignment_loss():
    board = ((torch.arange(8)[:,None]+torch.arange(8)[None,:])%2).float()
    pred = board.expand(1,3,8,8).clone().requires_grad_()
    fine, _ = auxiliary_losses(pred, pred.detach(), [reference(rgb=.5)], (0.,1.,2.))
    coarse, _ = auxiliary_losses(pred, pred.detach(), [reference(sigma=2.,rgb=.5)], (0.,1.,2.))
    assert coarse < fine*.1
    coarse.backward()
    assert torch.isfinite(pred.grad).all()


def test_camera_projection_preserves_off_center_intrinsics_and_rotation():
    camera = dict(K=torch.tensor([[20.,0.,7.],[0.,22.,5.],[0.,0.,1.]]),
                  R=torch.tensor([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]), t=torch.tensor([.1,.2,3.]))
    view, full, center = camera_matrices(camera, (16,20))
    world = torch.tensor([[.2,.4,.5,1.]])
    clip = world @ full
    ndc = clip[:,:2]/clip[:,3:4]
    pixels = ((ndc+1)*torch.tensor([20.,16.])-1)/2
    projected = camera['K']@(camera['R']@world[0,:3]+camera['t'])
    torch.testing.assert_close(pixels[0], projected[:2]/projected[2])
    torch.testing.assert_close(camera['R']@center+camera['t'], torch.zeros(3))


def test_bundle_roundtrip_preserves_provenance_camera_and_evidence(tmp_path):
    _, engine, frames, _, _ = fixture()
    options = SimpleNamespace(save_dir=str(tmp_path), obs_scales=(0.,1.,2.), obs_eval_size=8, seed=42)
    images = torch.zeros(len(frames),3,8,8)
    path = save_reconstruction_window(images, frames, None, engine, options, 0)
    data = torch.load(path, weights_only=True)
    assert not data['trajectory_from_evaluation_images']
    assert len(data['observations']) == 1 and len(data['targets']) == len(frames)-1
    assert data['targets'][0]['references']
    torch.testing.assert_close(data['observations'][0]['rgb'], frames[0].rgb.cpu())
    assert not torch.equal(data['observations'][0]['rgb'], data['targets'][0]['rgb'])
    points, colors = seed_points(data['observations'], stride=2)
    assert points.shape == colors.shape and points.shape[1] == 3


def test_loss_can_optimize_shared_render_parameters_on_cpu():
    parameter = torch.nn.Parameter(torch.full((1,3,8,8), .8))
    optimizer = torch.optim.SGD([parameter], lr=2.)
    initial = parameter.detach().abs().mean()
    for _ in range(12):
        loss, _ = reconstruction_loss(parameter, torch.zeros_like(parameter), parameter,
            torch.ones_like(parameter), [reference()], (0.,1.,2.), lambda a,b: 1-(a-b).square().mean())
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    assert parameter.detach().abs().mean() < initial


def test_out_of_bank_scales_cannot_supply_evidence():
    pred = torch.ones(1,3,8,8)
    evidence, completion = auxiliary_losses(pred, torch.zeros_like(pred), [reference(sigma=9.)], (0.,1.,2.))
    assert evidence.item() == 0 and completion.item() == 1
