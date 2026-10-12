import torch

from mini_llm.model import GPT, GPTConfig
from mini_llm.sampling import (
    SamplingParams,
    apply_repetition_penalty,
    filter_top_k,
    filter_top_p,
    next_token_distribution,
)

LOGITS = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))


def test_top_k_keeps_k_best():
    kept = torch.isfinite(filter_top_k(LOGITS, 2))
    assert kept.tolist() == [[True, True, False, False]]


def test_top_p_adapts_to_cumulative_probability():
    assert torch.isfinite(filter_top_p(LOGITS, 0.5)).tolist() == [[True, False, False, False]]
    assert torch.isfinite(filter_top_p(LOGITS, 0.8)).tolist() == [[True, True, False, False]]
    assert torch.isfinite(filter_top_p(LOGITS, 0.9)).tolist() == [[True, True, True, False]]
    assert torch.isfinite(filter_top_p(LOGITS, 1.0)).all()


def test_repetition_penalty_lowers_seen_tokens():
    logits = torch.tensor([[2.0, -1.0, 2.0]])
    out = apply_repetition_penalty(logits, torch.tensor([[0, 1]]), 2.0)
    assert out.tolist() == [[1.0, -2.0, 2.0]]  # el token 2 no apareció: no cambia
    assert torch.equal(apply_repetition_penalty(logits, torch.tensor([[0]]), 1.0), logits)


def test_distribution_is_valid_and_temperature_zero_is_greedy():
    ctx = torch.tensor([[3]])
    probs = next_token_distribution(LOGITS, ctx, SamplingParams(temperature=0.7, top_k=3, top_p=0.95))
    assert abs(probs.sum().item() - 1) < 1e-6 and probs[0, 3] == 0
    greedy = next_token_distribution(LOGITS, ctx, SamplingParams(temperature=0))
    assert greedy.tolist() == [[1.0, 0.0, 0.0, 0.0]]


def test_generate_steps_returns_model_probabilities_and_stops():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=50, block_size=16, n_layer=1, n_head=2, n_embd=16)).eval()
    steps = list(model.generate_steps(torch.tensor([[1, 2]]), 8, SamplingParams(temperature=0)))
    assert len(steps) == 8
    for token, raw in steps:
        assert raw.shape == (1, 50) and abs(raw.sum().item() - 1) < 1e-5
        assert int(token) == int(raw.argmax())  # temperatura 0 = el más probable
    first = int(steps[0][0])
    assert list(model.generate_steps(torch.tensor([[1, 2]]), 8, SamplingParams(temperature=0),
                                     stop_ids={first})) == []


def test_repetition_penalty_reduces_repeats_in_greedy_generation():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=30, block_size=32, n_layer=1, n_head=2, n_embd=16)).eval()
    prompt = torch.tensor([[1, 2, 3]])
    plain = model.generate(prompt, 25, temperature=0)[0, 3:].tolist()
    penalized = model.generate(prompt, 25, temperature=0, repetition_penalty=3.0)[0, 3:].tolist()
    assert len(set(penalized)) >= len(set(plain))
