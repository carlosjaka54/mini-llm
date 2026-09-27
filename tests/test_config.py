import pytest

from mini_llm.config import Config
from mini_llm.utils import CONFIGS_DIR


@pytest.mark.parametrize("name, expected_m", [("cpu.yaml", 4.2), ("gpu.yaml", 13.8)])
def test_profiles_load(name, expected_m):
    cfg = Config.from_yaml(CONFIGS_DIR / name)
    assert isinstance(cfg.train.learning_rate, float)
    assert abs(cfg.approx_params / 1e6 - expected_m) < 0.2


def test_invalid_heads():
    with pytest.raises(ValueError):
        Config.from_dict({"model": {"n_embd": 250, "n_head": 4}})


def test_unknown_key():
    with pytest.raises(ValueError):
        Config.from_dict({"train": {"learnin_rate": 1e-3}})


def test_roundtrip(tmp_path):
    cfg = Config.from_yaml(CONFIGS_DIR / "cpu.yaml")
    cfg.save(tmp_path / "cfg.yaml")
    assert Config.from_yaml(tmp_path / "cfg.yaml") == cfg
