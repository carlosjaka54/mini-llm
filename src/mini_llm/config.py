"""Configuración tipada del proyecto, cargada desde archivos YAML.

Separar la configuración del código permite reproducir cualquier experimento
guardando solo su archivo YAML junto al checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class DataConfig:
    dataset: str = "Gabrui/multilingual_TinyStories"  # repositorio en Hugging Face
    subset: str = "spanish"                  # subconjunto (idioma); "" si no aplica
    text_field: str = "story"                # columna que contiene el texto
    train_split: str = "train"               # split de origen para entrenamiento
    val_split: str = "test"                  # split de origen para validación
    train_samples: int = 100_000             # documentos de entrenamiento a descargar
    val_samples: int = 5_000                 # documentos de validación a descargar
    vocab_size: int = 4096
    block_size: int = 128


@dataclass
class ModelConfig:
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    bias: bool = False


@dataclass
class TrainConfig:
    batch_size: int = 32
    max_iters: int = 3000
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    warmup_iters: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_iters: int = 50
    checkpoint_interval: int = 500
    num_threads: int = 0


def _build(cls: type, values: dict[str, Any] | None) -> Any:
    """Crea una sección de la config, convirtiendo tipos y rechazando claves desconocidas."""
    values = values or {}
    hints = get_type_hints(cls)
    unknown = set(values) - set(hints)
    if unknown:
        raise ValueError(f"Claves desconocidas en '{cls.__name__}': {sorted(unknown)}")
    return cls(**{k: hints[k](v) for k, v in values.items()})


@dataclass
class Config:
    experiment_name: str = "mini-llm"
    seed: int = 1337
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        d, m, t = self.data, self.model, self.train
        if d.train_samples <= 0 or d.val_samples <= 0:
            raise ValueError("train_samples y val_samples deben ser mayores que 0.")
        if m.n_embd % m.n_head != 0:
            raise ValueError(f"n_embd ({m.n_embd}) debe ser divisible por n_head ({m.n_head}).")
        if not 0.0 <= m.dropout < 1.0:
            raise ValueError("dropout debe estar en [0, 1).")
        if t.min_lr > t.learning_rate:
            raise ValueError("min_lr no puede ser mayor que learning_rate.")
        if t.warmup_iters >= t.max_iters:
            raise ValueError("warmup_iters debe ser menor que max_iters.")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError(f"device inválido: {self.device}")

    @property
    def approx_params(self) -> int:
        """Estimación: 12·d² por bloque + embeddings de tokens y de posición."""
        d = self.model.n_embd
        return 12 * self.model.n_layer * d * d + (self.data.vocab_size + self.data.block_size) * d

    @property
    def tokens_per_iter(self) -> int:
        return self.train.batch_size * self.data.block_size

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8"
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        raw = dict(raw)
        sections = {
            "data": _build(DataConfig, raw.pop("data", None)),
            "model": _build(ModelConfig, raw.pop("model", None)),
            "train": _build(TrainConfig, raw.pop("train", None)),
        }
        allowed = {"experiment_name", "seed", "device"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Claves desconocidas en la raíz de la config: {sorted(unknown)}")
        return cls(**raw, **sections)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo de configuración: {path}")
        return cls.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
