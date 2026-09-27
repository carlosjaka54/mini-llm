# mini-llm

LLM educativo entrenado **desde cero** con Python y PyTorch: un Transformer
*decoder-only* estilo GPT, con tokenizador BPE propio y sin pesos preentrenados.

## Etapas

| # | Etapa | Módulo | Estado |
|---|-------|--------|--------|
| 1 | Preparar entorno | `check_env.py` | ✅ |
| 2 | Obtener dataset | `data.py` | ✅ |
| 3 | Limpiar y tokenizar | `cleaning.py`, `tokenizer.py`, `prepare.py` | ✅ |
| 4 | Crear el modelo | `model.py`, `dataset.py`, `model_info.py` | ✅ |
| 5 | Entrenar | `train.py` | ✅ |
| 6 | Guardar checkpoints | `checkpoint.py` | ✅ |
| 7 | Evaluar | `evaluate.py` | ✅ |
| 8 | Generar texto | `generate.py` | ⏳ |

## Instalación (Windows)

```bat
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

## Verificación

```bat
mini-llm-check
mini-llm-data
mini-llm-prepare
mini-llm-tokens "Había una vez..."
mini-llm-model
mini-llm-train
mini-llm-ckpt
mini-llm-eval
pytest
```

## Estructura

```
configs/      Perfiles de hiperparámetros (cpu.yaml, gpu.yaml)
src/mini_llm/ Código fuente del paquete
tests/        Pruebas automáticas
data/         raw/ (texto original) y processed/ (tokens)
artifacts/    tokenizer/, checkpoints/, logs/
```
