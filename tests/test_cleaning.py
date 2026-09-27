from mini_llm.cleaning import clean_document, fix_mojibake, normalize_text, text_fingerprint

STORY = "Había una vez un gato pequeño que vivía en una casa grande con su familia feliz. " * 2


def test_fix_mojibake():
    assert fix_mojibake("Â¿QuÃ© pasÃ³?") == "¿Qué pasó?"
    assert fix_mojibake("¿Qué pasó?") == "¿Qué pasó?"  # texto correcto no se toca


def test_normalize_quotes_dashes_markdown_and_spaces():
    raw = "\u00abHola\u00bb,   dijo **Ana**\u2026  \n\n\n\u2014S\u00ed\u2019"
    assert normalize_text(raw) == "\"Hola\", dijo Ana...\n\n-Sí'"


def test_normalize_unicode_nfc():
    decomposed = "e\u0301"  # "e" + tilde combinada
    assert normalize_text(decomposed) == "é"


def test_filters():
    assert clean_document(STORY, 5, 800)[1] == "ok"
    assert clean_document("Hola.", 5, 800) == (None, "muy corto")
    assert clean_document(STORY * 50, 5, 100) == (None, "muy largo")
    assert clean_document("这是一个很长的中文故事 " * 10, 5, 800) == (None, "caracteres extraños")
    assert clean_document("   ", 5, 800) == (None, "vacío")


def test_fingerprint_ignores_case_and_spaces():
    assert text_fingerprint("Hola  Mundo") == text_fingerprint("hola mundo")
