import math
from itertools import pairwise

import numpy as np
import pytest
import torch

from mini_llm.checkpoint import load_checkpoint, load_model, save_checkpoint
from mini_llm.config import Config
from mini_llm.model import GPT, GPTConfig
from mini_llm.train import configure_optimizer, get_lr, train
from mini_llm.utils import get_logger

# Secuencia repetitiva 0,1,...,49,0,1,...: el siguiente token es predecible
DATA = np.tile(np.arange(50, dtype=np.uint16), 400)


def tiny_config(**train_overrides) -> Config:
    train_cfg = {"batch_size": 8, "max_iters": 60, "warmup_iters": 5, "learning_rate": 3e-3,
                 "min_lr": 3e-4, "eval_interval": 20, "eval_iters": 3,
                 "checkpoint_interval": 20, "log_interval": 20}
    train_cfg.update(train_overrides)
    return Config.from_dict({
        "experiment_name": "test",
        "device": "cpu",
        "data": {"vocab_size": 300, "block_size": 16},
        "model": {"n_layer": 2, "n_head": 2, "n_embd": 32, "dropout": 0.0},
        "train": train_cfg,
    })


def test_lr_schedule():
    t = tiny_config().train
    assert get_lr(0, t) == pytest.approx(t.learning_rate / t.warmup_iters)
    assert get_lr(t.warmup_iters - 1, t) == pytest.approx(t.learning_rate)
    assert get_lr(t.max_iters, t) == pytest.approx(t.min_lr)
    after_warmup = [get_lr(i, t) for i in range(t.warmup_iters, t.max_iters + 1)]
    assert all(a >= b for a, b in pairwise(after_warmup))


def test_optimizer_covers_every_parameter_once():
    model = GPT(GPTConfig(300, 16, 2, 2, 32))
    opt = configure_optimizer(model, 0.1, 1e-3)
    decay, no_decay = opt.param_groups
    assert sum(p.numel() for g in opt.param_groups for p in g["params"]) == model.num_parameters()["total"]
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"]) and no_decay["weight_decay"] == 0.0


def test_checkpoint_roundtrip(tmp_path):
    cfg = tiny_config()
    torch.manual_seed(0)
    model = GPT(GPTConfig.from_config(cfg)).eval()
    opt = configure_optimizer(model, 0.1, 1e-3)
    save_checkpoint(tmp_path / "m.pt", model, opt, cfg, 7, 1.23)

    loaded, ck = load_model(tmp_path / "m.pt")
    idx = torch.randint(0, 300, (1, 16))
    assert torch.allclose(model(idx)[0], loaded(idx)[0])
    assert ck["iter"] == 7 and ck["best_val_loss"] == pytest.approx(1.23)
    assert "optimizer" in ck
    assert loaded.lm_head.weight is loaded.wte.weight  # el weight tying sobrevive


def test_training_learns_and_writes_artifacts(tmp_path):
    run = tmp_path / "run"
    result = train(tiny_config(), DATA, DATA[:2000], run, get_logger("t1"), vocab_size=300)
    for name in ["best.pt", "last.pt", "metrics.csv", "loss.png", "config.yaml"]:
        assert (run / name).exists(), name
    assert result["best_val_loss"] < math.log(300) - 2  # aprendió el patrón


def test_resume_continues_from_last_checkpoint(tmp_path):
    run = tmp_path / "run"
    train(tiny_config(max_iters=40), DATA, DATA[:2000], run, get_logger("t2"), vocab_size=300)
    assert load_checkpoint(run / "last.pt")["iter"] == 40
    train(tiny_config(max_iters=60), DATA, DATA[:2000], run, get_logger("t3"), vocab_size=300,
          resume=True)
    assert load_checkpoint(run / "last.pt")["iter"] == 60
