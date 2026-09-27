import numpy as np

from mini_llm import prepare
from mini_llm.tokenizer import EOT, eot_id, iter_token_ids, train_tokenizer

CORPUS = [
    "Había una vez un niño llamado Tomás que tenía un perro pequeño.",
    "¡Qué día tan bonito! dijo la niña. ¿Quieres jugar conmigo?",
    "El gato marrón saltó sobre la mesa y comió un poco de pan.",
] * 20


def _tok():
    return train_tokenizer(CORPUS, vocab_size=300, min_frequency=1)


def test_roundtrip_spanish():
    tok = _tok()
    text = "¿Dónde está el pingüino? ¡Mañana iré con Ñandú!"
    assert tok.decode(tok.encode(text).ids) == text  # sin pérdida, incluso con palabras nuevas


def test_special_token_and_vocab():
    tok = _tok()
    assert eot_id(tok) == 0 and tok.id_to_token(0) == EOT
    assert tok.get_vocab_size() <= 300


def test_iter_token_ids_appends_eot():
    tok = _tok()
    docs = list(iter_token_ids(tok, ["hola", "adiós"]))
    assert all(d[-1] == eot_id(tok) for d in docs)


def test_encode_to_bin_roundtrip(tmp_path):
    tok = _tok()
    texts = ["Había una vez un gato.", "El perro ladró."]
    n = prepare.encode_to_bin(tok, texts, tmp_path / "x.bin")
    arr = np.fromfile(tmp_path / "x.bin", dtype=np.uint16)
    assert arr.size == n
    expected = [i for ids in iter_token_ids(tok, texts) for i in ids]
    assert arr.tolist() == expected


def test_clean_split_removes_duplicates_and_leakage():
    story = "Había una vez un gato que vivía en una casa muy grande y bonita."
    seen: set[str] = set()
    train, st = prepare.clean_split([story, story.upper()], 5, 100, seen, "duplicado")
    val, sv = prepare.clean_split([story], 5, 100, seen, "repetido en train")
    assert len(train) == 1 and st["duplicado"] == 1
    assert val == [] and sv["repetido en train"] == 1
