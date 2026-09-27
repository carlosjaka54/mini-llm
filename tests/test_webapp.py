import pytest
import torch
from fastapi.testclient import TestClient

from mini_llm.model import GPT, GPTConfig
from mini_llm.tokenizer import train_tokenizer
from mini_llm.webapp import create_app

CORPUS = [
    "Había una vez un niño llamado Tomás que tenía un perro pequeño.",
    "¡Qué día tan bonito! dijo la niña. ¿Quieres jugar conmigo?",
] * 20
N_LAYER, N_HEAD = 2, 2


@pytest.fixture(scope="module")
def client():
    tok = train_tokenizer(CORPUS, vocab_size=300, min_frequency=1)
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=tok.get_vocab_size(), block_size=32, n_layer=N_LAYER, n_head=N_HEAD,
                    n_embd=16)
    model, untrained = GPT(cfg).eval(), GPT(cfg).eval()
    return TestClient(create_app(model, untrained, tok, {"experiment": "test"}))


def test_index_and_info(client):
    assert "mini-LLM" in client.get("/").text
    assert client.get("/api/info").json() == {"experiment": "test"}


def test_tokenize_rebuilds_text_even_with_multibyte_chars(client):
    text = "¿Dónde está el pingüino? ¡Ñandú!"
    groups = client.post("/api/tokenize", json={"text": text}).json()
    assert "".join(g["piece"] for g in groups) == text
    assert all(len(g["ids"]) >= 1 for g in groups)


def test_probs_sorted(client):
    rows = client.post("/api/probs", json={"text": "Había una", "k": 5}).json()
    probs = [r["prob"] for r in rows]
    assert len(rows) == 5 and probs == sorted(probs, reverse=True)


def test_generate_reproducible_and_untrained(client):
    body = {"prompt": "Había", "max_new_tokens": 12, "seed": 3, "temperature": 1.0}
    assert client.post("/api/generate", json=body).text == client.post("/api/generate", json=body).text
    assert client.post("/api/generate", json={**body, "trained": False}).status_code == 200


def test_attention_is_causal_and_normalized(client):
    r = client.post("/api/attention", json={"text": "Había una vez un niño"}).json()
    T = len(r["tokens"])
    assert len(r["layers"]) == N_LAYER and len(r["layers"][0]) == N_HEAD
    for layer in r["layers"]:
        for head in layer:
            for i in range(T):
                assert abs(sum(head[i]) - 1) < 1e-3              # cada fila suma 1
                assert all(head[i][j] == 0 for j in range(i + 1, T))  # no mira al futuro


def test_rejects_invalid_parameters(client):
    assert client.post("/api/generate", json={"temperature": 9}).status_code == 422
