import math

import numpy as np
import pytest
import torch

from mini_llm.evaluate import bigram_loss, distinct_n, evaluate_model, unigram_loss
from mini_llm.model import GPT, GPTConfig

PATTERN = np.tile(np.arange(10, dtype=np.uint16), 500)  # 0,1,...,9,0,1,...


def test_unigram_on_uniform_tokens_is_about_log_10():
    assert unigram_loss(PATTERN, PATTERN, vocab_size=10) == pytest.approx(math.log(10), abs=0.01)


def test_bigram_beats_unigram_on_predictable_sequence():
    uni = unigram_loss(PATTERN, PATTERN, vocab_size=20)
    bi = bigram_loss(PATTERN, PATTERN, vocab_size=20)
    assert bi < 0.1 < uni  # con el token anterior, el siguiente es casi seguro


def test_distinct_n():
    assert distinct_n(["a b a b"], 2) == pytest.approx(2 / 3)
    assert distinct_n(["uno dos tres"], 2) == 1.0


def test_evaluate_model_untrained():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=50, block_size=8, n_layer=1, n_head=2, n_embd=16)
    model = GPT(cfg)
    data = np.random.default_rng(0).integers(0, 50, size=801).astype(np.uint16)
    r = evaluate_model(model, data, batch_size=16)
    assert r["tokens"] == 100 * 8                      # 100 ventanas completas
    assert abs(r["loss"] - math.log(50)) < 0.3         # sin entrenar ~ azar
    assert 0 <= r["top1_accuracy"] <= r["top5_accuracy"] <= 1
    assert len(r["loss_by_position"]) == 8
