import io
import json

from mini_llm import data
from mini_llm.config import Config
from mini_llm.utils import get_logger

LOG = get_logger("test")


def _texts(buf: io.StringIO) -> list[str]:
    return [json.loads(line)["text"] for line in buf.getvalue().splitlines()]


def _cfg(**data_overrides) -> Config:
    return Config.from_dict({"data": data_overrides})


def test_separate_splits_skip_empty_and_respect_limits():
    source = {"train": ["  hola  ", "", "mundo", "extra"], "test": ["v1", "v2", "v3"]}

    def fake(data_cfg, split, skip=0):
        yield from source[split][skip:]

    files = {"train": io.StringIO(), "val": io.StringIO()}
    written = data.download(_cfg(), {"train": 2, "val": 1}, files, LOG, stream_fn=fake)
    assert written == {"train": 2, "val": 1}
    assert _texts(files["train"]) == ["hola", "mundo"]
    assert _texts(files["val"]) == ["v1"]


def test_resumes_after_network_error(monkeypatch):
    monkeypatch.setattr(data.time, "sleep", lambda s: None)
    source = ["a", "", "b", "c", "d", "e"]
    calls = {"n": 0}

    def flaky(data_cfg, split, skip=0):
        calls["n"] += 1
        for i, text in enumerate(source[skip:], start=skip):
            if calls["n"] == 1 and i == 3:
                raise ConnectionError("red caída")
            yield text

    cfg = _cfg(train_split="train", val_split="otro")
    files = {"train": io.StringIO(), "val": io.StringIO()}
    data.download(cfg, {"train": 4, "val": 0}, files, LOG, stream_fn=flaky)
    assert _texts(files["train"]) == ["a", "b", "c", "d"]  # sin duplicados ni pérdidas


def test_single_split_is_divided_without_overlap():
    docs = [f"doc {i}" for i in range(200)]

    def fake(data_cfg, split, skip=0):
        yield from docs[skip:]

    cfg = _cfg(train_split="train", val_split="train")
    files = {"train": io.StringIO(), "val": io.StringIO()}
    written = data.download(cfg, {"train": 90, "val": 10}, files, LOG, stream_fn=fake)
    train, val = _texts(files["train"]), _texts(files["val"])
    assert written == {"train": 90, "val": 10}
    assert not set(train) & set(val)
    assert val[0] != "doc 0"  # la validación se reparte a lo largo del recorrido


def test_topic_filter_counts_whole_words():
    cfg = _cfg(topic_keywords=["datos", "algoritmo", "software"], min_topic_terms=2, min_topic_hits=3)
    f = data.TopicFilter(cfg.data)
    assert f.accepts("El algoritmo procesa datos. Un software usa algoritmos y datos.")
    assert not f.accepts("Los candidatos votaron en la elección municipal.")  # "datos" != "candidatos"
    assert f.score("Algoritmos, algoritmo y DATOS") == (2, 3)


def test_topic_filter_applied_during_download():
    docs = ["receta de cocina", "el software y el algoritmo del software", "fútbol", "algoritmo de software"]

    def fake(data_cfg, split, skip=0):
        yield from docs[skip:]

    cfg = _cfg(train_split="a", val_split="b", topic_keywords=["software", "algoritmo"],
               min_topic_terms=2, min_topic_hits=2)
    files = {"train": io.StringIO(), "val": io.StringIO()}
    data.download(cfg, {"train": 2, "val": 0}, files, LOG, stream_fn=fake)
    assert _texts(files["train"]) == [docs[1], docs[3]]


def test_file_stats(tmp_path):
    out = tmp_path / "val.jsonl"
    out.write_text('{"text": "uno dos"}\n{"text": "tres"}\n', encoding="utf-8")
    stats = data.file_stats(out)
    assert stats["documents"] == 2 and stats["words"] == 3
