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

## Etapa 9: chatbot con ajuste fino (SFT)

El modelo base solo continúa texto. Con `mini-llm-sft` se crea automáticamente un dataset de
~20 000 pares *petición → cuento* a partir de los cuentos, se añaden los tokens `<|usuario|>` y
`<|asistente|>`, y se ajusta el modelo base (la pérdida solo se calcula sobre la respuesta).
Con 20 000 ejemplos y 1 500 iteraciones (37 min), la perplejidad de la respuesta correcta
bajó de 7,5 a 5,5. Un primer intento con 1 910 ejemplos sobreajustó: la validación subía.

```bat
mini-llm-sft      # ~30-40 min en CPU; el modelo base no se modifica
mini-llm-chat     # chat en la terminal (o la pestaña Chat de mini-llm-web)
```

## Perfil tecnología

Segundo modelo (~8,4 M de parámetros, contexto de 256 tokens) entrenado con artículos de
Wikipedia en español filtrados por temas de informática, inteligencia artificial y tecnología.
Cada corpus usa sus propias carpetas, así ambos modelos conviven:

```bat
mini-llm-data     --config configs/tech.yaml
mini-llm-prepare  --config configs/tech.yaml
mini-llm-train    --config configs/tech.yaml
mini-llm-eval     --config configs/tech.yaml
mini-llm-web      --config configs/tech.yaml --port 8001
```

Texto de Wikipedia bajo licencias CC BY-SA 3.0 y GFDL (dataset `wikimedia/wikipedia`, volcado 20231101.es).
Cuentos: `Gabrui/multilingual_TinyStories` (CDLA-Sharing-1.0).

## Interfaz web (laboratorio)

Doble clic en **`iniciar.bat`** (activa el entorno solo) o, desde la terminal:

```bat
mini-llm-web
```

Abre `http://127.0.0.1:8000`; si ese puerto está ocupado, usa el siguiente libre y lo indica.
Funciona sin conexión a internet.

| Sección | Qué permite ver |
|---|---|
| Inicio | Un cuento escrito en vivo, cifras clave, etapas y arquitectura |
| Generar texto | Temperatura, top-k, top-p y penalización de repetición; comparaciones lado a lado |
| Chat | El modelo con ajuste fino frente al modelo base, y *mejor de N*: escribe varios borradores y elige el más coherente (`rerank.py`) |
| Analizar un texto | Qué palabras de cualquier frase sorprenden al modelo y qué esperaba en su lugar |
| Siguiente token | Las probabilidades del próximo token; escribir token a token |
| Atención | A qué tokens mira cada token, por capa y por cabeza |
| Embeddings | Palabras vecinas por significado y mapa 2D (PCA) |
| Tokenizador | Cómo se corta un texto en tokens BPE |
| Métricas | Curvas de aprendizaje, modelos de referencia y efecto del contexto |

Todo texto generado se muestra **resaltado según la duda del modelo**: al pasar el mouse
sobre una palabra se ve con qué probabilidad la eligió y qué otras opciones consideraba.
Tecla `P`: modo presentación (con `←` `→` para cambiar de sección).

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
| 9 | Ajuste fino (SFT) y chat | `sft.py` | ✅ |

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
