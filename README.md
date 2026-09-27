# mini-llm

LLM educativo entrenado **desde cero** con Python y PyTorch: un Transformer
*decoder-only* estilo GPT, con tokenizador BPE propio y sin pesos preentrenados.

## Resultados (perfil CPU)

Modelo de 4,2 M de parámetros entrenado 3 000 iteraciones (~80 min en CPU de 12 hilos)
con 100 000 cuentos de TinyStories en español.

| Modelo | Pérdida (val) | Perplejidad |
|---|---|---|
| Uniforme (azar) | 8,318 | 4 096 |
| Unigrama | 5,810 | 333,6 |
| Bigrama | 3,294 | 27,0 |
| **mini-GPT** | **2,057** | **7,8** |

Precisión del siguiente token: 50,4 % (top-1) y 79,2 % (top-5).

> Había una vez, un niño pequeño llamado Sam. Sam amaba jugar con su pelota.
> Sam tiró la pelota. La pelota rodó y rodó. Sam la encontró.

## Interfaz web

```bat
mini-llm-web
```

Abre `http://127.0.0.1:8000` con seis secciones: resumen, generación (con comparación de
temperaturas y antes/después del entrenamiento), probabilidades del siguiente token,
visualización de la atención, tokenizador y métricas. Funciona sin conexión a internet.

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
| 8 | Generar texto | `generate.py` | ✅ |

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
mini-llm-generate
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
