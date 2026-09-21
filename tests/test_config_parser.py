from configs.infer_config import get_parser
import pytest


def test_boolean_cli_values_are_parsed_semantically():
    args = get_parser().parse_args([
        '--planner_loop_back', 'false',
        '--mask_image', '0',
        '--mask_pc', 'off',
        '--reduce_pc', 'no',
        '--negative_prompt', 'true',
        '--text_input', 'yes',
        '--perframe_ae', '1',
        '--use_freq_mix', 'false',
    ])

    assert args.planner_loop_back is False
    assert args.mask_image is False
    assert args.mask_pc is False
    assert args.reduce_pc is False
    assert args.negative_prompt is True
    assert args.text_input is True
    assert args.perframe_ae is True
    assert args.use_freq_mix is False


def test_default_video_length_matches_bundled_checkpoint():
    args = get_parser().parse_args([])
    assert args.video_length == 16


def test_evidence_saving_can_be_disabled():
    assert get_parser().parse_args([]).save_evidence is True
    assert get_parser().parse_args(['--save_evidence', 'false']).save_evidence is False


def test_uncertainty_cli_and_legacy_default():
    assert get_parser().parse_args([]).init_scale_mode == 'legacy'
    args = get_parser().parse_args(['--init_scale_mode', 'uncertainty', '--init_depth_rel_std', '.1',
                                  '--init_scales', '0', '1', '2'])
    assert args.init_depth_rel_std == .1 and args.init_scales == [0., 1., 2.]


@pytest.mark.parametrize('arguments', [
    ['--init_depth_rel_std', 'nan'], ['--init_fixed_sigma', '-1'],
    ['--init_scales', '0', '2', '1'], ['--init_scales', '1', '2'],
    ['--init_scales', '0', 'inf'],
])
def test_invalid_uncertainty_cli(arguments):
    with pytest.raises(SystemExit):
        get_parser().parse_args(arguments)


def test_loop_one_flag_and_options():
    from utils.observation_loop import LoopConfig
    assert not LoopConfig.from_options(get_parser().parse_args([])).enabled
    config = LoopConfig.from_options(get_parser().parse_args([
        '--observation_loop', '--loop_max_rounds', '3', '--loop_resample_steps', '5',
        '--loop_scales', '0', '1', '4',
    ]))
    assert config.enabled and config.max_rounds == 3 and config.resample_steps == 5
    assert config.scales == [0., 1., 4.]
