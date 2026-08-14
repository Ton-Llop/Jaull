Jaull analiza tu hardware, busca modelos, estima cuáles encajan, los recomienda, puede descargar un GGUF y ya puede ejecutarlo realmente con llama.cpp.

Aunque tengas 18 carpetas, en realidad son 4 bloques grandes:

                                             JAULL
                                             │
                              ┌──────────────┼───────────────┐
                              │              │               │
                              ▼              ▼               ▼
                         ANALIZAR       RECOMENDAR       EJECUTAR
     Primero Validate

Aquí reutilizaría al máximo lo que ya tenemos:

ExperimentRunner
      ↓
runtime_family = pytorch_transformers
      ↓
TransformersRunner
      ↓
ExecutionObservation
      ↓
PredictionComparison
      ↓
ExperimentRecord

La idea es que pulsar Validate sobre un modelo Transformers haga una ejecución controlada y guarde evidencia igual que hacemos con llama.cpp:

Runtime       Transformers / PyTorch
Backend       CPU
Artifact      safetensors
Success       yes
Duration      13.4 s
Peak RAM      2.5 GiB
Peak VRAM     unavailable

Y mantener la semántica importante:

READY = no conocemos blockers.
Validate SUCCESS = lo hemos ejecutado realmente.

Eso ya permitiría comparar predicción vs realidad para Transformers.

2. Después Benchmark

Aquí no intentaría meter Transformers dentro de llama-bench, porque llama-bench pertenece a llama.cpp.

Haría algo como:

BenchmarkRunner
├── LlamaBenchRunner
└── TransformersBenchmarkRunner

Pero compartiendo, si nuestro dominio actual lo permite:

BenchmarkRequest
BenchmarkObservation
BenchmarkRecord
BenchmarkStore

El runner de Transformers podría ejecutar un worker PyTorch dedicado.

Métricas que sí me interesan

Para Transformers mediría al menos:

Model load time
Peak RAM
Peak VRAM


Prompt processing / prefill
tokens/s


Generation
tokens/s


Time to first token
ms


Total generation latency
s

Y posiblemente warm-up/repeticiones para obtener dispersión.

Hay algo MUY evidente en tu prueba de ahora

Tus ejecuciones fueron aproximadamente:

1ª   127 s
2ª    15 s
3ª    13 s

Esto nos está enseñando precisamente por qué no podemos benchmarkear simplemente cronometrando el botón Generate.

Tenemos que separar:

COLD START
download/cache/model load/init

de:

STEADY-STATE INFERENCE
modelo ya cargado
→ prompt
→ generación

Porque si mezclamos ambos, Jaull podría concluir absurdamente que el modelo genera durante 127 segundos cuando gran parte de ese tiempo probablemente pertenece a preparación/carga inicial.

Yo incluso conservaría ambas métricas:

Startup / model load    110 s
Warm inference           14 s
Generation               X tok/s

Eso tiene bastante valor para decidir despliegues.

Y cuidado al comparar con llama.cpp

Quiero que eventualmente podamos ver:

Qwen2.5 0.5B · CPU


                     llama.cpp       Transformers
Prompt processing     XXX tok/s       YYY tok/s
Generation             XX tok/s        YY tok/s
Peak RAM               X.X GiB        X.X GiB
Startup                  X s            Y s

Pero solo cuando la metodología sea comparable.

No asumiría automáticamente:

llama-bench pp512
==
Transformers generate benchmark

porque no necesariamente están midiendo exactamente el mismo recorrido.

Primero definimos un protocolo común; luego comparamos. Hasta entonces guardamos methodology/runtime junto con la observación.                         │              │               │
                         Hardware         Discovery        Artifacts
                         HF metadata      Ranking          llama.cpp
                         Estimator        Workflow         subprocess
                         Runtime
                              │              │               │
                              └──────────────┴───────────────┘
                                             │
                                        AdvisorService
                                             │
                                        ┌──────┴──────┐
                                        ▼             ▼
                                   CLI           TUI

Ese es el dibujo que yo tendría siempre en la cabeza.

2. El bloque de análisis

Aquí está todo lo que responde:

“¿Qué máquina tengo y qué necesita este modelo?”

hardware/

Mira tu PC.

hardware/
├── cpu.py
├── memory.py
├── nvidia.py
├── storage.py
└── detector.py

Produce un:

HardwareProfile

con CPU, RAM, GPU, VRAM, etc.

huggingface/

Es la conexión con Hugging Face.

huggingface/
├── client.py
├── search_client.py
├── artifact_resolver.py
├── classifiers.py
└── ...

Hay una distinción importante:

HfSearchClient
    ↓
busca modelos

HfClient
    ↓
inspecciona un modelo concreto

HuggingFaceArtifactResolver
    ↓
encuentra el GGUF exacto que quieres descargar

Tu .env ya se carga desde ServiceContainer.default(), y HF_TOKEN acaba llegando a estos clientes.

analyzers/

Entiende qué tipo de repo has encontrado.

Transformers
GGUF
ONNX
Diffusers
Generic

No estima memoria todavía. Primero simplemente entiende:

“¿Qué coño es este repo?”

metadata/

Extrae información más profunda.

Aquí tienes una parte bastante buena de Jaull: para un GGUF puedes leer su header mediante HTTP Range sin descargar varios GB.

Ejemplo:

GGUF
 ↓
header
 ↓
architecture
context size
attention heads
KV heads
etc.

Y además intenta encontrar el modelo base.

3. El estimator

Esta es una de las piezas centrales.

estimator/

Responde:

“¿Cuánta memoria va a necesitar este modelo con esta configuración?”

Conceptualmente:

weights
   +
KV cache
   +
runtime overhead
   +
device reserve
   +
safety margin
────────────
TOTAL

Después compara eso contra:

RAM
VRAM
hardware disponible

y genera algo tipo:

comfortable
compatible
tight
offloading_required
insufficient
unknown

Aquí tienes cosas bastante curradas ya:

GGUF real vs cuantización teórica.
KV cache.
GQA.
contexto.
usuarios concurrentes.
margen de seguridad.
reserva de dispositivo.
provenance/confidence de cada estimación.
4. runtime/: aquí hay una distinción MUY importante

Esta carpeta puede confundir porque ahora contiene dos conceptos diferentes.

runtime/llama_cpp.py

NO ejecuta llama.cpp.

Hace esto:

MemoryEstimate
    +
HardwareProfile
       ↓
"te recomiendo llama.cpp
 ctx = 4096
 GPU layers = X"

Es un recomendador de configuración.

Produce:

RuntimeRecommendation
runtime/llama_cpp_runner.py

Este sí ejecuta.

Ahora mismo hace:

ModelArtifact
     ↓
validar GGUF
     ↓
construir comando
     ↓
--model ...
--ctx-size ...
--n-gpu-layers ...
--single-turn
--prompt ...
     ↓
ExecutionBackend

Y esto es precisamente lo que acabamos de validar físicamente contra tu llama-cli.

Esta distinción conviene que la tengas clarísima:

llama_cpp.py
= "cómo debería ejecutarlo"

llama_cpp_runner.py
= "ejecútalo de verdad"
5. execution/

Esto es deliberadamente genérico.

execution/
├── host.py
├── ports.py
└── errors.py

HostExecutionBackend no sabe qué es llama.cpp.

Solo sabe hacer:

subprocess.run(
    command,
    shell=False,
    capture_output=True,
    ...
)

Es decir:

LlamaCppRunner
      ↓
"ejecuta este comando"
      ↓
HostExecutionBackend
      ↓
Linux / WSL

Esto está bien separado.

Mañana podrías tener:

DockerExecutionBackend
RemoteExecutionBackend
SSHExecutionBackend

sin modificar LlamaCppRunner demasiado.

Pero no lo haría todavía.

6. artifacts/

Esta pieza convierte:

“Quiero TinyLlama Q4_K_M”

en:

“Tengo físicamente /home/ton/.../tinyllama....gguf y sé que está correcto.”

El flujo es:

resolve
  ↓
ModelArtifact
  repo
  revision
  filename
  size

download
  ↓
archivo local
  ↓
SHA-256

verify
  ↓
is_verified=True

Y los guarda en:

~/.local/share/jaull/models/

Esto ya lo has probado contra Hugging Face real.

7. Discovery + Recommendation

Aquí está el otro bloque grande.

discovery/
recommendation/
workflow/
discovery/

Responde:

“¿Qué modelos vale la pena mirar?”

No puedes inspeccionar 20.000 repos uno por uno.

Así que haces:

queries HF
    ↓
resultados
    ↓
interleave
    ↓
deduplicate
    ↓
filter
    ↓
shortlist

Actualmente el workflow permite hasta:

40 candidatos únicos
        ↓
12 inspecciones profundas
        ↓
3 recomendaciones

Esto es una optimización importante porque la inspección profunda cuesta llamadas a HF.

recommendation/

Responde:

“De los modelos que ya he analizado, ¿cuál es mejor para este usuario?”

Aquí están:

memory fit
concurrency fit
capability
task match
language
license
metadata quality
popularity
artifact realism
runtime executability

Luego aplicas gates.

Por ejemplo:

AWQ + CPU sin CUDA
        ↓
BLOCKED

aunque su score bruto fuese alto.

Esto fue precisamente para corregir el problema que tenías anteriormente con Qwen AWQ.

8. workflow/

Esta carpeta no calcula prácticamente nada importante por sí misma.

Coordina.

UserAnswers
     ↓
requirements
     ↓
buscar
     ↓
filtrar
     ↓
inspeccionar
     ↓
estimar
     ↓
enriquecer
     ↓
rankear
     ↓
RecommendationWorkflowState

Piensa en él como el director de orquesta.

WorkflowOrchestrator dice:

“ahora toca esto, luego esto, luego esto.”

pero los cálculos los hacen los servicios inferiores.

9. AdvisorService

Esta es probablemente la clase que más deberías entender de todo el repo.

advisor/service.py

Es la fachada.

La idea es que la UI no tenga que conocer:

HfClient
ArtifactService
Estimator
Workflow
Hardware
etc.

Sino:

advisor.scan_hardware()
advisor.inspect_model()
advisor.estimate_model()
advisor.recommend()

advisor.resolve_artifact()
advisor.download_artifact()
advisor.verify_artifact()
advisor.run_artifact()

Conceptualmente:

               AdvisorService
              /      |       \
             /       |        \
         análisis  recommend  execute

Para entender Jaull, empezaría por esta clase antes que por ninguna otra.