from mini_llm.rerank import character_names, repeated_trigrams, requested, score_story


def test_requested_reads_names_and_topics():
    assert requested("¿Me cuentas un cuento de una niña llamada Sofía?") == (["Sofía"], ["niña"])
    assert requested("Cuéntame un cuento sobre un perro llamado Max.") == (["Max"], ["perro"])
    assert requested("quiero un cuento del lobo") == ([], ["lobo"])


def test_character_names_ignores_sentence_starts():
    text = "Un día, Sofía vio a su amigo Leo. Pero Leo estaba triste. Ella dijo: hola."
    assert character_names(text) == ["Sofía", "Leo"]


def test_repetition():
    assert repeated_trigrams("el perro corre y el gato salta") == 0
    assert repeated_trigrams("él amaba su hueso. él amaba su hueso.") > 0.3


def test_scorer_prefers_the_story_that_keeps_its_protagonist():
    msg = "¿Me cuentas un cuento de una niña llamada Sofía?"
    good = "Una niña llamada Sofía tenía un perro llamado Spot. Sofía y Spot jugaban. Sofía sonrió."
    bad = "Una niña llamada Sofía tenía un perro llamado Spot. Luego dijo Lucy: hola. Y vino Tom. Lucy sonrió."
    assert score_story(msg, good, 5.0).score > score_story(msg, bad, 5.0).score
    assert score_story(msg, bad, 5.0).new_names == ["Spot", "Lucy", "Tom"]


def test_scorer_penalizes_ignoring_the_request():
    msg = "quiero un cuento del lobo"
    lion = "Una vez, una niña vio un león. El león era grande."
    wolf = "Una vez, una niña vio un lobo. El lobo era grande."
    assert score_story(msg, wolf, 5.0).score > score_story(msg, lion, 5.0).score
    assert score_story(msg, lion, 5.0).missing == ["lobo"]
