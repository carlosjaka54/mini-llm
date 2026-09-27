import io

import pytest
import torch

from mini_llm.generate import GenSettings, StreamDecoder, next_token_table, stream_generate
from mini_llm.model import GPT, GPTConfig
from mini_llm.tokenizer import train_tokenizer

CORPUS = [
    "Había una vez un niño llamado Tomás que tenía un perro pequeño.",
    "¡Qué día tan bonito! dijo la niña. ¿Quieres jugar conmigo?",
] * 20


@pytest.fixture(scope="module")
def tok():
    return train_tokenizer(CORPUS, vocab_size=300, min_frequency=1)


@pytest.fixture(scope="module")
def model(tok):
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=tok.get_vocab_size(), block_size=16, n_layer=1, n_head=2, n_embd=16)
    return GPT(cfg).eval()


def test_stream_decoder_rebuilds_multibyte_text(tok):
    text = "¿Dónde está el pingüino? ¡Ñandú!"
    dec = StreamDecoder(tok)
    streamed = "".join(dec.push(i) for i in tok.encode(text).ids)
    assert streamed == text
    assert "\ufffd" not in streamed


def test_generate_iter_matches_generate(model):
    prompt = torch.tensor([[1, 2, 3]])
    full = model.generate(prompt, 20, temperature=0)
    streamed = torch.cat([prompt, *model.generate_iter(prompt, 20, temperature=0)], dim=1)
    assert torch.equal(full, streamed)


def test_stream_generate_writes_what_it_returns(model, tok):
    buf = io.StringIO()
    text = stream_generate(model, tok, "Había", GenSettings(max_new_tokens=15, seed=1), out=buf)
    assert text.startswith("Había")
    assert buf.getvalue() == text + "\n"


def test_seed_makes_generation_reproducible(model, tok):
    s = GenSettings(max_new_tokens=15, seed=123, temperature=1.0)
    a = stream_generate(model, tok, "Había", s, out=io.StringIO())
    b = stream_generate(model, tok, "Había", s, out=io.StringIO())
    assert a == b


def test_next_token_table(model, tok):
    rows = next_token_table(model, tok, "Había una", k=5)
    probs = [p for _, p in rows]
    assert len(rows) == 5
    assert probs == sorted(probs, reverse=True)
    assert sum(probs) <= 1 + 1e-6
