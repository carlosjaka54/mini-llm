import json
import socket

import pytest
import torch
from fastapi.testclient import TestClient

from mini_llm.model import GPT, GPTConfig
from mini_llm.sft import add_chat_tokens
from mini_llm.tokenizer import train_tokenizer
from mini_llm.webapp import create_app, find_free_port

CORPUS = [
    "Había una vez un niño llamado Tomás que tenía un perro pequeño.",
    "¡Qué día tan bonito! dijo la niña. ¿Quieres jugar conmigo?",
    "El gato y el perro jugaban en el parque con la pelota roja.",
] * 20
N_LAYER, N_HEAD = 2, 2


def _events(response) -> list[dict]:
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


@pytest.fixture(scope="module")
def tok():
    return train_tokenizer(CORPUS, vocab_size=300, min_frequency=1)


@pytest.fixture(scope="module")
def client(tok):
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=tok.get_vocab_size(), block_size=32, n_layer=N_LAYER, n_head=N_HEAD,
                    n_embd=16)
    model, untrained = GPT(cfg).eval(), GPT(cfg).eval()
    return TestClient(create_app(model, untrained, tok, {"experiment": "test"}))


def test_index_and_info(client):
    assert "mini-LLM" in client.get("/").text
    info = client.get("/api/info").json()
    assert info["experiment"] == "test" and info["chat"] == {"available": False}


def test_generate_streams_tokens_with_confidence(client):
    body = {"prompt": "Había", "max_new_tokens": 12, "seed": 3, "temperature": 1.0}
    r = client.post("/api/generate", json=body)
    assert r.headers["content-type"].startswith("application/x-ndjson")
    events = _events(r)
    tokens, done = events[:-1], events[-1]
    assert done["type"] == "done" and 1 <= done["tokens"] <= 12
    assert all(e["type"] == "token" and 0 <= e["p"] <= 1 and e["rank"] >= 1 for e in tokens)
    assert all(len(e["alts"]) == 4 for e in tokens)
    texts = [  # misma semilla, mismo texto
        "".join(e["text"] for e in _events(client.post("/api/generate", json=body))[:-1])
        for _ in range(2)
    ]
    assert texts[0] == texts[1]


def test_generate_accepts_new_sampling_options(client):
    body = {"prompt": "Había", "max_new_tokens": 10, "seed": 1, "top_p": 0.9,
            "repetition_penalty": 1.3, "trained": False}
    assert _events(client.post("/api/generate", json=body))[-1]["type"] == "done"


def test_tokenize_rebuilds_text_even_with_multibyte_chars(client):
    text = "¿Dónde está el pingüino? ¡Ñandú!"
    groups = client.post("/api/tokenize", json={"text": text}).json()
    assert "".join(g["piece"] for g in groups) == text


def test_probs_sorted(client):
    rows = client.post("/api/probs", json={"text": "Había una", "k": 5}).json()
    probs = [r["prob"] for r in rows]
    assert len(rows) == 5 and probs == sorted(probs, reverse=True)


def test_score_rebuilds_text_and_gives_probabilities(client):
    text = "El gato y el perro jugaban."
    r = client.post("/api/score", json={"text": text}).json()
    assert "".join(t["text"] for t in r["tokens"]) == text
    assert all(0 <= t["p"] <= 1 for t in r["tokens"]) and r["perplexity"] > 0


def test_attention_is_causal_and_normalized(client):
    r = client.post("/api/attention", json={"text": "Había una vez un niño"}).json()
    T = len(r["tokens"])
    assert len(r["layers"]) == N_LAYER and len(r["layers"][0]) == N_HEAD
    for layer in r["layers"]:
        for head in layer:
            for i in range(T):
                assert abs(sum(head[i]) - 1) < 1e-3                    # cada fila suma 1
                assert all(head[i][j] == 0 for j in range(i + 1, T))  # no mira al futuro


def test_neighbors_and_embedding_map(client):
    r = client.post("/api/neighbors", json={"word": "perro", "k": 5}).json()
    assert r["word"] == "perro" and len(r["neighbors"]) <= 5
    sims = [n["similarity"] for n in r["neighbors"]]
    assert sims == sorted(sims, reverse=True) and all(n["text"] != "perro" for n in r["neighbors"])
    m = client.post("/api/embedding_map", json={"words": ["gato", "perro", "niño", "parque"]}).json()
    assert len(m["points"]) == 4 and len(m["explained_variance"]) == 2
    assert client.post("/api/embedding_map", json={"words": ["gato"]}).status_code == 422


def test_chat_requires_trained_model_but_base_mode_works(client):
    assert client.post("/api/chat", json={"message": "Hola"}).status_code == 409
    r = client.post("/api/chat", json={"message": "Hola", "mode": "base", "seed": 1, "max_new_tokens": 5})
    assert r.status_code == 200 and _events(r)[-1]["type"] == "done"


def test_chat_with_sft_model(tok):
    chat_tok = train_tokenizer(CORPUS, vocab_size=300, min_frequency=1)
    add_chat_tokens(chat_tok)
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=chat_tok.get_vocab_size(), block_size=32, n_layer=1, n_head=2, n_embd=16)
    model = GPT(cfg).eval()
    app = create_app(model, model, chat_tok, {}, chat={"model": model, "tokenizer": chat_tok})
    r = TestClient(app).post("/api/chat", json={"message": "Cuéntame un cuento", "seed": 2, "max_new_tokens": 8})
    assert r.status_code == 200 and _events(r)[-1]["type"] == "done"


def test_rejects_invalid_parameters(client):
    assert client.post("/api/generate", json={"temperature": 9}).status_code == 422
    assert client.post("/api/generate", json={"top_p": 0}).status_code == 422


def test_find_free_port_skips_busy_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        assert find_free_port("127.0.0.1", port) != port
