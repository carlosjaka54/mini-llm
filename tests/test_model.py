import math

import numpy as np
import pytest
import torch

from mini_llm.dataset import get_batch
from mini_llm.model import GPT, GPTConfig

CFG = GPTConfig(vocab_size=97, block_size=16, n_layer=2, n_head=4, n_embd=32, dropout=0.0)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return GPT(CFG).eval()


def test_parameter_count_matches_formula(model):
    V, T, C, L = CFG.vocab_size, CFG.block_size, CFG.n_embd, CFG.n_layer
    # por bloque: atención 4C² + MLP 8C² + 2 LayerNorm (C cada una, sin bias)
    expected = V * C + T * C + L * (12 * C * C + 2 * C) + C
    assert model.num_parameters()["total"] == expected


def test_weight_tying(model):
    assert model.lm_head.weight is model.wte.weight


def test_forward_shapes_and_initial_loss(model):
    idx = torch.randint(0, CFG.vocab_size, (3, CFG.block_size))
    # Objetivos independientes de la entrada. Si usáramos targets=idx, el weight tying
    # hace que el modelo recién creado favorezca "repetir el token actual" y la pérdida
    # sale menor que ln(V). En texto real el siguiente token casi nunca es el mismo.
    targets = torch.randint(0, CFG.vocab_size, (3, CFG.block_size))
    logits, loss = model(idx, targets)
    assert logits.shape == (3, CFG.block_size, CFG.vocab_size)
    assert abs(loss.item() - math.log(CFG.vocab_size)) < 0.3  # ~ azar al inicio


def test_loss_accepts_non_contiguous_targets(model):
    idx = torch.randint(0, CFG.vocab_size, (2, CFG.block_size + 1))
    x, y = idx[:, :-1], idx[:, 1:]  # los "cortes" no son contiguos en memoria
    assert not y.is_contiguous()
    _, loss = model(x, y)
    assert torch.isfinite(loss)


def test_rejects_too_long_sequences(model):
    with pytest.raises(ValueError):
        model(torch.zeros(1, CFG.block_size + 1, dtype=torch.long))


def test_causality_future_tokens_do_not_matter(model):
    idx = torch.randint(0, CFG.vocab_size, (1, CFG.block_size))
    changed = idx.clone()
    changed[0, 10:] = (changed[0, 10:] + 1) % CFG.vocab_size  # alteramos el "futuro"
    a, _ = model(idx)
    b, _ = model(changed)
    assert torch.allclose(a[0, :10], b[0, :10], atol=1e-5)  # el pasado no cambia
    assert not torch.allclose(a[0, 10:], b[0, 10:])


def test_manual_attention_equals_sdpa(model):
    idx = torch.randint(0, CFG.vocab_size, (2, CFG.block_size))
    fast, _ = model(idx)
    for block in model.blocks:
        block.attn.use_sdpa = False
    slow, _ = model(idx)
    assert torch.allclose(fast, slow, atol=1e-5)


def test_generate_length_and_greedy_is_deterministic(model):
    prompt = torch.tensor([[1, 2, 3]])
    out1 = model.generate(prompt, max_new_tokens=25, temperature=0)  # supera block_size
    out2 = model.generate(prompt, max_new_tokens=25, temperature=0)
    assert out1.shape == (1, 28)
    assert torch.equal(out1, out2)
    assert int(out1.max()) < CFG.vocab_size


def test_generate_stops_at_eot(model):
    prompt = torch.tensor([[1, 2, 3]])
    first = model.generate(prompt, max_new_tokens=1, temperature=0)[0, -1].item()
    out = model.generate(prompt, max_new_tokens=10, temperature=0, eot_id=first)
    assert out.shape == (1, 4)


def test_can_overfit_a_single_batch():
    """Prueba de cordura clásica: si no puede memorizar un lote, algo está roto."""
    torch.manual_seed(0)
    m = GPT(CFG).train()
    idx = torch.randint(0, CFG.vocab_size, (4, CFG.block_size))
    x, y = idx[:, :-1], idx[:, 1:]
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    first = None
    for _ in range(60):
        _, loss = m(x, y)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.5


def test_get_batch_targets_are_shifted_inputs():
    data = np.arange(1000, dtype=np.uint16)
    x, y = get_batch(data, batch_size=8, block_size=16, generator=torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (8, 16)
    assert x.dtype == torch.int64
    assert torch.equal(y, x + 1)
