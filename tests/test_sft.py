import random

import torch

from mini_llm.config import Config
from mini_llm.model import GPT, GPTConfig
from mini_llm.sft import (
    ASSISTANT,
    IGNORE,
    USER,
    Example,
    add_chat_tokens,
    build_tensors,
    chat_prompt_ids,
    collect_examples,
    encode_example,
    expand_vocab,
    fit_response,
    make_instruction,
    reply,
    sft_train,
)
from mini_llm.tokenizer import EOT, train_tokenizer
from mini_llm.utils import get_logger

STORIES = [
    "Había una vez un perro llamado Spot. Spot jugaba en el parque con su pelota roja.",
    "Había una vez un pequeño conejo. El conejo saltaba en el bosque y era feliz.",
    "Una niña llamada Lily tenía una muñeca. Lily aprendió a compartir con sus amigos.",
] * 10


def _chat_tokenizer():
    tok = train_tokenizer(STORIES, vocab_size=300, min_frequency=1)
    add_chat_tokens(tok)
    return tok


def test_make_instruction_with_named_character():
    instr, key = make_instruction(STORIES[0], random.Random(0))
    assert "un perro llamado Spot" in instr and key == "Spot"


def test_make_instruction_from_opening_and_skips_unknown():
    instr, key = make_instruction(STORIES[1], random.Random(0))
    assert "un pequeño conejo" in instr and key == "conejo"
    assert make_instruction("Hoy llovió mucho en la ciudad.", random.Random(0)) is None


def test_special_tokens_are_single_tokens():
    tok = _chat_tokenizer()
    ids = chat_prompt_ids(tok, "Hola")
    assert ids[0] == tok.token_to_id(USER) and ids[-1] == tok.token_to_id(ASSISTANT)


def test_loss_only_on_the_response():
    tok = _chat_tokenizer()
    ex = Example("Cuéntame un cuento.", "Había una vez un perro.", "perro")
    ids, start = encode_example(tok, ex)
    X, Y = build_tensors(tok, [ex], block_size=64)
    assert X.shape == Y.shape == (1, 64)
    assert (Y[0, : start - 1] == IGNORE).all()                    # la petición no se aprende
    assert Y[0, start - 1].item() == ids[start]                   # primer token de la respuesta
    assert Y[0, len(ids) - 2].item() == tok.token_to_id(EOT)      # termina con fin de texto
    assert (Y[0, len(ids) - 1 :] == IGNORE).all()                 # el relleno no se aprende


def test_fit_response_cuts_at_sentence_end():
    tok = _chat_tokenizer()
    story = "Había una vez un perro llamado Spot. Spot jugaba en el parque. Era feliz."
    full = len(tok.encode(story).ids)
    assert fit_response(tok, story, full) == story              # cabe: no se toca
    cut = fit_response(tok, story, full - 5)
    assert story.startswith(cut) and cut.endswith(".") and len(tok.encode(cut).ids) <= full - 5


def test_expand_vocab_keeps_old_behaviour():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=100, block_size=16, n_layer=1, n_head=2, n_embd=16)).eval()
    idx = torch.randint(0, 100, (1, 8))
    before = model(idx)[0]
    expand_vocab(model, 102)
    after = model(idx)[0]
    assert after.shape[-1] == 102 and model.config.vocab_size == 102
    assert torch.allclose(before, after[..., :100], atol=1e-6)    # lo aprendido no cambia
    assert model.lm_head.weight is model.wte.weight


def test_sft_smoke(tmp_path):
    # Con un tokenizador entrenado en solo 3 frases, las peticiones ocupan muchos tokens:
    # usamos un contexto de 128 para que las conversaciones quepan completas.
    tok = _chat_tokenizer()
    cfg = Config.from_dict({
        "device": "cpu",
        "data": {"vocab_size": 300, "block_size": 128},
        "model": {"n_layer": 1, "n_head": 2, "n_embd": 32, "dropout": 0.0},
        "sft": {"batch_size": 4, "max_iters": 20, "warmup_iters": 2, "eval_interval": 10,
                "eval_iters": 2, "log_interval": 10, "learning_rate": 3e-3},
    })
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=tok.get_vocab_size(), block_size=128, n_layer=1, n_head=2, n_embd=32))
    examples = collect_examples(tok, STORIES, random.Random(0), 30, 128)
    assert examples
    X, Y = build_tensors(tok, examples, 128)
    result = sft_train(cfg, model, {"X": X, "Y": Y, "Xv": X, "Yv": Y}, tmp_path, get_logger("sft-test"),
                       tok, sample_request="Cuéntame un cuento sobre un perro llamado Spot.")
    assert (tmp_path / "best.pt").exists() and not result["interrupted"]
    assert isinstance(reply(model, tok, "Cuéntame un cuento.", max_new_tokens=10), str)
