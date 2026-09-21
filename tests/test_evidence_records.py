from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')

from utils.evidence_records import save_evidence_window


def test_window_records_do_not_overwrite_and_retain_camera_mapping(tmp_path):
    options = SimpleNamespace(seed=42, use_fmi=True, unrelated_secret='omit')
    frames = [SimpleNamespace(camera={'K': torch.eye(3), 'R': torch.eye(3)})]
    first_record = {0: {0: {'routing_mask': torch.ones(1, 1, 2, 2)}}}
    second_record = {0: {0: {'routing_mask': torch.zeros(1, 1, 2, 2)}}}
    first = save_evidence_window(first_record, tmp_path, 0, options, frames)
    second = save_evidence_window(second_record, tmp_path, 0, options, frames)
    assert first != second
    a = torch.load(first, map_location='cpu', weights_only=True)
    b = torch.load(second, map_location='cpu', weights_only=True)
    assert a['window_index'] == b['window_index'] == 0
    assert a['config'] == {'seed': 42, 'use_fmi': True}
    assert torch.equal(a['cameras'][0]['K'], torch.eye(3))
    assert a['samples'][0][0]['routing_mask'].sum().item() == 4
    assert b['samples'][0][0]['routing_mask'].sum().item() == 0


def test_failed_record_write_removes_only_its_partial_file(tmp_path, monkeypatch):
    first = save_evidence_window({}, tmp_path, 0, SimpleNamespace())

    def fail(*args, **kwargs):
        raise OSError('write failed')

    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(OSError, match='write failed'):
        save_evidence_window({}, tmp_path, 1, SimpleNamespace())
    assert [str(path) for path in (tmp_path / 'evidence').iterdir()] == [first]


def test_scale_configuration_and_diagnostics_survive_save(tmp_path):
    options = SimpleNamespace(init_scale_mode='uncertainty', init_depth_rel_std=.05,
                              init_fixed_sigma=1., init_scales=[0., 1., 2.])
    record = {0: {1: {'initialization': {'mode': 'uncertainty', 'applied': True,
                                      'latent_sigma': torch.ones(1, 1, 2, 2)}}}}
    path = save_evidence_window(record, tmp_path, 0, options)
    restored = torch.load(path, weights_only=True)
    assert restored['config'] == vars(options)
    assert restored['samples'][0][1]['initialization']['applied']
    torch.testing.assert_close(restored['samples'][0][1]['initialization']['latent_sigma'],
                               torch.ones(1, 1, 2, 2))
