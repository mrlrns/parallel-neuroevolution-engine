"""Tests for config.py: defaults, YAML/CLI merge order, per-script argparsers."""

import pytest

from config import Config, build_argparser, charger_yaml, load_config


def test_defaults_match_dataclass():
    cfg = Config()
    assert cfg.seed == 0
    assert cfg.sub_step == 10
    assert cfg.frame_nb == 200
    assert cfg.chemin_champion is None


def test_charger_yaml_missing_file_returns_empty(tmp_path):
    assert charger_yaml(str(tmp_path / "does_not_exist.yaml")) == {}


def test_charger_yaml_none_returns_empty():
    assert charger_yaml(None) == {}


def test_charger_yaml_reads_values(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("seed: 42\nlearning_rate: 0.123\n")
    data = charger_yaml(str(p))
    assert data == {"seed": 42, "learning_rate": 0.123}


@pytest.mark.parametrize("script", ["train", "train2", "visualize", "test", "test2"])
def test_build_argparser_known_scripts(script):
    parser = build_argparser(script)
    args = parser.parse_args([])
    cfg = load_config(args)
    assert isinstance(cfg, Config)


def test_build_argparser_unknown_script_raises():
    with pytest.raises(ValueError):
        build_argparser("not_a_real_script")


def test_priority_dataclass_lt_yaml_lt_cli(tmp_path):
    """CLI overrides YAML overrides dataclass defaults."""
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("learning_rate: 0.5\nseed: 7\n")

    parser = build_argparser("train2")
    # Only override learning_rate on the CLI; seed should come from YAML.
    args = parser.parse_args(["--config", str(yaml_path), "--learning-rate", "0.9"])
    cfg = load_config(args)

    assert cfg.learning_rate == 0.9  # CLI wins over YAML
    assert cfg.seed == 7  # YAML wins over dataclass default (0)
    assert cfg.nb_episodes == Config().nb_episodes  # untouched -> dataclass default


def test_load_config_ignores_unknown_yaml_keys(tmp_path):
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("seed: 3\nchamp_qui_nexiste_pas: 123\n")
    parser = build_argparser("train")
    args = parser.parse_args(["--config", str(yaml_path)])
    cfg = load_config(args)
    assert cfg.seed == 3
    assert not hasattr(cfg, "champ_qui_nexiste_pas")


def test_chemin_champion_required_scripts_expose_flag():
    for script in ["train2", "visualize", "test", "test2"]:
        parser = build_argparser(script)
        args = parser.parse_args(["--chemin-champion", "foo.pt"])
        cfg = load_config(args)
        assert cfg.chemin_champion == "foo.pt"
