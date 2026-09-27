import json

from mini_llm import data
from mini_llm.config import Config
from mini_llm.utils import get_logger


def _read(path):
    return [json.loads(line)["text"] for line in path.read_text(encoding="utf-8").splitlines()]


def test_download_split_skips_empty_and_respects_limit(tmp_path):
    source = ["  hola  ", "", "mundo", "extra"]

    def fake_stream(data_cfg, split, skip=0):
        yield from source[skip:]

    out = tmp_path / "train.jsonl"
    n = data.download_split(Config(), "train", 2, out, get_logger("test"), stream_fn=fake_stream)
    assert n == 2
    assert _read(out) == ["hola", "mundo"]


def test_download_split_resumes_after_network_error(tmp_path, monkeypatch):
    monkeypatch.setattr(data.time, "sleep", lambda s: None)
    source = ["a", "", "b", "c", "d", "e"]
    calls = {"n": 0}

    def flaky_stream(data_cfg, split, skip=0):
        calls["n"] += 1
        for i, text in enumerate(source[skip:], start=skip):
            if calls["n"] == 1 and i == 3:
                raise ConnectionError("red caída")
            yield text

    out = tmp_path / "train.jsonl"
    data.download_split(Config(), "train", 4, out, get_logger("test"), stream_fn=flaky_stream)
    assert _read(out) == ["a", "b", "c", "d"]  # sin duplicados ni pérdidas
    assert calls["n"] == 2


def test_file_stats(tmp_path):
    out = tmp_path / "val.jsonl"
    out.write_text('{"text": "uno dos"}\n{"text": "tres"}\n', encoding="utf-8")
    stats = data.file_stats(out)
    assert stats["documents"] == 2 and stats["words"] == 3


def test_source_split_mapping():
    cfg = Config()
    assert data.source_split(cfg.data, "train") == "train"
    assert data.source_split(cfg.data, "val") == "test"
