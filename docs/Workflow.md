# Arquitectura de `jaull`

Aquest document explica com està organitzat el projecte, quina responsabilitat té cada carpeta i quin recorregut segueix una recomanació des que l’usuari obre la interfície fins que es genera l’informe final. També documenta el camí explícit `run`, que queda separat del workflow guiat perquè descarrega i executa artefactes locals.

> Aquest document descriu l’estat actual del codi. No és una especificació definitiva: algunes heurístiques del recomanador encara s’han de validar amb benchmarks reals.

---

## 1. Què fa el projecte?

`jaull` és una eina de terminal que:

1. Detecta el maquinari disponible.
2. Recull les necessitats de l’usuari mitjançant un assistent guiat.
3. Cerca models públics al Hugging Face Hub.
4. Analitza els repositoris candidats.
5. Estima la memòria necessària per executar-los.
6. Selecciona una configuració tècnica probable.
7. Construeix els **plans d'execució** concrets de cada candidat (artefacte + runtime +
   backend) i els ordena amb evidència, no amb una puntuació única.
8. Mostra recomanacions explicades, amb les alternatives d'execució de cada model.
9. Opcionalment executa de veritat: `jaull run` descarrega, verifica i executa un GGUF amb
   `llama-cli`, i la TUI hi afegeix execució de repositoris Transformers, **validació**
   (experiments persistits) i **benchmarks** (`llama-bench` i un worker de Transformers).

El projecte ofereix dues formes d’ús:

- **Mode guiat:** l’usuari respon preguntes senzilles i rep recomanacions automàtiques.
- **Eines avançades:** permet executar manualment `scan`, `inspect`, `estimate`, `doctor` i, des de CLI, `run`.

---

## 2. Visió general de l’arquitectura

```text
CLI / TUI
   │
   ▼
AdvisorService                  ← façana única cap als serveis
   │
   ├── application/             ← casos d’ús (requisits, variants, recomanació)
   └── workflow/                ← orquestrador del mode guiat
   │
   ▼
domain/ + ports/                ← models immutables i contractes
   ▲
adapters/ · huggingface/ · analyzers/ · metadata/ · hardware/ · estimator/ ·
runtime/ · execution/ · artifacts/ · experiments/ · benchmarks/ · evaluation/
                                ← infraestructura, composada per bootstrap/
   ▲
reporting/ · presentation/ · diagnostics/
                                ← renderitzat sobre semàntica ja calculada
```

El mode guiat, pas a pas:

```text
WorkflowOrchestrator
   ├── Detecció de maquinari
   ├── Traducció de requisits
   ├── Descobriment de candidats al Hub
   ├── Filtratge + shortlist conscient del maquinari
   ├── Inspecció profunda del shortlist
   ├── Estimació de memòria i HardwareFit
   ├── Construcció dels plans d’execució
   ├── Avaluació i rànquing dels plans (engine v2)
   ├── Diversificació (un lloc per model lògic)
   └── Generació de l’informe
```

Els camins d’execució són més curts i explícits, i no passen pel workflow guiat:

```text
CLI run                             TUI · Execution Paths
   ├── resol artefacte GGUF            ├── prepara el pla (descarrega + verifica)
   ├── descarrega si falta             ├── executa (llama.cpp o Transformers)
   ├── verifica mida + SHA-256         ├── valida → ExperimentRecord persistit
   └── executa llama-cli               └── benchmark → BenchmarkRecord persistit
```

La idea principal és separar:

- **Presentació:** què veu l’usuari.
- **Orquestració:** en quin ordre s’executen les operacions.
- **Casos d’ús:** quina pregunta es respon, sense saber d’on venen les dades.
- **Domini:** com es representen les dades.
- **Infraestructura externa:** maquinari, Hugging Face, HTTP, sistema de fitxers i processos
  locals.

Aquesta separació permet reutilitzar els mateixos serveis des de la CLI, la TUI i els tests.

> Aquest document explica **què fa** cada carpeta. Les **regles de dependència** entre capes,
> i quines es comproven automàticament amb `tests/test_architecture_dependencies.py`, estan a
> [architecture.md](architecture.md).

---

# 3. Arrel del repositori

```text
.
├── .github/
├── docs/
├── scripts/
├── src/
├── tests/
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
└── uv.lock
```

## `.github/`

Conté automatitzacions de GitHub.

### `.github/workflows/ci.yml`

Workflow d’integració contínua. Instal·la el projecte i comprova que:

- Ruff no detecti errors.
- Mypy passi en mode estricte.
- Tots els tests passin.
- El paquet es pugui construir.
- Els recursos de la TUI s’incloguin dins del wheel.

## `docs/`

Documentació i recursos visuals.

### `docs/assets/`

Captures reals de la TUI utilitzades al README principal.

Aquest document i el glossari tècnic també es poden desar dins de `docs/`.

## `scripts/`

Scripts auxiliars de manteniment i validació.

### `scripts/check_dist.py`

Inspecciona els artefactes generats per `uv build` i comprova, entre altres coses, que el paquet inclogui fitxers no Python necessaris, com ara `styles.tcss`.

## `tests/`

Tests unitaris i d’integració.

Els tests estan separats per responsabilitat:

- maquinari;
- URLs i API de Hugging Face;
- classificació de repositoris;
- GGUF i lectura per rangs;
- estimació de memòria;
- selecció de runtime;
- descobriment i rànquing;
- workflow complet;
- CLI i TUI.

Els tests del workflow utilitzen serveis falsos i fixtures. No haurien de consultar Internet ni Hugging Face.

## `pyproject.toml`

Configuració central del projecte:

- metadades del paquet;
- dependències de producció;
- dependències de desenvolupament;
- entrada de consola `jaull`;
- configuració de Ruff;
- configuració de Mypy;
- configuració de Pytest;
- sistema de construcció amb Hatchling.

## `uv.lock`

Bloqueja les versions exactes de les dependències per obtenir instal·lacions reproduïbles.

## `LICENSE`

Llicència MIT del codi del projecte.

No s’ha de confondre amb la llicència dels models analitzats. Cada model de Hugging Face pot tenir una llicència diferent.

---

# 4. Paquet principal: `src/jaull/`

```text
src/jaull/
├── domain/            ← models immutables, sense dependències
├── ports/             ← contractes de frontera
├── adapters/          ← implementacions d’aquests contractes
├── application/       ← casos d’ús
├── bootstrap/         ← arrel de composició
├── observability/     ← telemetria
│
├── hardware/          ┐
├── huggingface/       │
├── analyzers/         │
├── metadata/          │
├── estimator/         │  infraestructura i càlcul
├── runtime/           │
├── execution/         │
├── artifacts/         │
├── execution_plans/   ┘
│
├── discovery/         ← cerca i selecció de candidats
├── recommendation/    ← avaluació i rànquing
├── workflow/          ← orquestrador del mode guiat
│
├── experiments/       ┐
├── benchmarks/        │  evidència mesurada
├── evaluation/        ┘
│
├── advisor/           ← façana única
├── reporting/         ┐
├── presentation/      │  sortida
├── diagnostics/       ┘
├── cli/               ┐
├── tui/               ┘  interfícies
│
├── exceptions.py
├── paths.py
├── __init__.py
└── __main__.py
```

L’ordre d’aquesta llista no és alfabètic: va de baix a dalt de la pila. `domain/` no importa
res de ningú; `cli/` i `tui/` només parlen amb `advisor/`.

---

## 4.1. `__main__.py` i `cli/app.py`: entrada del programa

Quan s’executa:

```bash
jaull
```

Python entra a `jaull.__main__`, que delega en l’aplicació Typer definida a:

```text
cli/app.py
```

`cli/app.py` decideix què fer:

- Si hi ha un subcomandament, executa la CLI corresponent.
- Si no hi ha subcomandament i la terminal és interactiva, obre la TUI.
- Si la sortida està redirigida, mostra l’ajuda en lloc d’intentar dibuixar una interfície de pantalla completa.

Comandes principals:

```bash
jaull scan
jaull inspect MODEL
jaull estimate MODEL
jaull doctor
jaull ui
jaull run --model MODEL --prompt TEXT
jaull experiments
```

---

# 4.2. `application/`: casos d’ús

Aquesta carpeta respon *quina pregunta es resol*, sense saber d’on venen les dades. No importa
mai infraestructura: parla amb `domain/` i amb `ports/`, i `bootstrap/` li injecta les
implementacions concretes.

## `application/requirements.py`

Tradueix les respostes del wizard en `UserRequirements` normalitzats. Abans vivia a
`workflow/requirements.py`, que ara només és un shim de compatibilitat.

## `application/model_reference.py`

Normalitza el que escriu l’usuari (`repo_id`, URL, URL amb `/blob/`) en un `repo_id` canònic.

## `application/discovery/`

Descobriment de variants d’artefacte d’un mateix model lògic, amb el seu propi pressupost
d’inspeccions (`MAX_VARIANT_DEEP_INSPECTION`).

## `application/recommendation/`

El servei de recomanació complet i **els pressupostos** (`policies.py`):

| Pressupost | Valor |
|---|---|
| Resultats per consulta | 20 |
| Candidats únics | 40 |
| Inspeccions profundes | 12 |
| Inspeccions concurrents | 4 |
| Inspeccions de variants | 6 |
| Recomanacions retornades | 5 |

`recommend()` té dos camins: **amb maquinari** ordena plans d’execució amb l’engine v2 i
després diversifica; **sense maquinari** cau al score compost antic, perquè no té plans que
avaluar.

---

# 4.3. `ports/`, `adapters/` i `bootstrap/`: injecció de dependències

## `ports/cache.py`

Protocols de frontera, només on la infraestructura és realment substituïble
(`ModelAnalysisCacheProtocol`). Un port que coneix la seva implementació no és un port.

## `adapters/cache/model_analysis_cache.py`

La implementació concreta: caché persistent de l’anàlisi cara d’un repositori, perquè repetir
una cerca no torni a inspeccionar els mateixos repos.

## `bootstrap/container.py`

L’arrel de composició de producció. És l’únic lloc autoritzat a construir adaptadors concrets
—client HTTP, analitzador de capacitat, factory del lector Range, caché persistent— i
muntar-los en un `ServiceContainer`.

`workflow/container.py` continua existint com a re-export perquè els imports històrics
resolguin. El cablejat nou va a `bootstrap/`.

---

# 4.4. `advisor/`: la façana

`advisor/service.py` és l’**únic** punt d’entrada que utilitzen la CLI i la TUI. Cap pantalla
construeix `HfClient()`, `detect_hardware` o `estimate_memory` pel seu compte.

Agrupa, entre altres:

- anàlisi i recomanació: `scan_hardware`, `diagnostics`, `inspect_model`, `estimate_model`,
  `recommend`;
- artefactes: `resolve_artifact`, `download_artifact`, `verify_artifact`, `run_artifact`;
- plans: `resolve_model_identity`, `discover_artifact_variants`,
  `execution_plans_for_recommendation`, `prepare_execution_plan`;
- runtimes: `select_runtime_backend`, `inspect_llama_cpp_runtime`, `inspect_pytorch_runtime`,
  `evaluate_execution_readiness`;
- evidència: `run_experiment`, `run_benchmark`, `run_benchmark_matrix`, els magatzems i les
  comparacions.

Dues factories: `AdvisorService.default()` per producció i `AdvisorService.build(...)` per
tests, amb cada servei injectat com a callable.

---

# 5. `domain/`: models de dades

La carpeta `domain/` defineix els objectes que circulen pel sistema. No hauria de contenir codi de Textual, Rich o Typer.

## Fitxers principals

### `domain/hardware.py`

Models relacionats amb el maquinari:

- `CpuInfo`
- `MemoryInfo`
- `StorageInfo`
- `GpuInfo`
- `HardwareProfile`

`HardwareProfile` és el resum final del maquinari que consumeixen l’estimador i el workflow.

### `domain/model.py`

Models relacionats amb repositoris i configuracions:

- `ModelFile`
- `ModelRepositoryInfo`
- `ModelConfig`
- `GgufVariant`
- `SafetensorsSummary`
- `RepositoryClassification`
- `ModelAnalysis`

`ModelAnalysis` és el resultat normalitzat d’inspeccionar un repositori.

### `domain/inference.py`

Defineix la configuració d’una estimació concreta:

- dispositiu objectiu;
- precisió dels pesos;
- context;
- batch size;
- tipus de dades del KV cache;
- marge de seguretat;
- reserva de memòria.

L’objecte principal és `InferenceConfiguration`.

### `domain/estimation.py`

Models generats per l’estimador:

- `MemoryComponent`
- `WeightEstimate`
- `KvCacheEstimate`
- `RuntimeOverheadEstimate`
- `CompatibilityAssessment`
- `MemoryEstimate`
- nivells de confiança;
- procedència de les dades;
- estat de compatibilitat.

### `domain/enrichment.py`

Models utilitzats per completar la configuració d’un GGUF:

- metadades de la capçalera GGUF;
- configuració enriquida;
- resultat global de l’enriquiment.

### `domain/runtime.py`

Models de recomanació de runtime:

- nom del runtime;
- flags;
- procedència dels flags;
- comandament orientatiu;
- warnings;
- nivell de confiança.

### `domain/artifacts.py`

Defineix `ModelArtifact`, el contracte entre “repositori + quantització” i “fitxer local executable”:

- repo_id;
- revisió;
- nom del fitxer;
- format;
- quantització;
- mida esperada;
- ruta local;
- SHA-256;
- estat de descàrrega;
- estat de verificació.

### `domain/execution.py`

Models immutables per executar processos sense acoblar el domini a `subprocess`:

- `ExecutionRequest`
- `ExecutionResult`
- `InferenceResult`

### `domain/enums.py`

Enumeracions generals, com ara:

- tipus de repositori;
- formats;
- estat dels diagnòstics.

---

### La resta de `domain/`

El domini ha crescut molt més enllà dels fitxers anteriors. Els que apareixen sovint:

| Fitxer | Conté |
|---|---|
| `candidates.py` | `ModelCandidate` i `EvaluatedCandidate` (abans a `discovery/models.py`) |
| `execution_plans.py` | `ExecutionPlan`, `ArtifactVariant`, `ModelIdentity`, `PlanCompatibilityStatus` |
| `recommendation.py` | `PlanAssessment`, `AssessmentLevel`, `HardConstraint`, `ModelRecommendation` |
| `experiments.py` | `ExperimentRecord`, immutable |
| `benchmarks.py` | `BenchmarkRecord` i els tipus de mesura |
| `comparison.py` | `PredictionComparison` predicció ↔ observació |
| `families.py`, `licenses.py`, `parameters.py`, `policies.py` | heurístiques pures i taules constants |
| `artifact_profile.py`, `requirements.py`, `enrichment.py` | perfils d’artefacte, requisits i enriquiment |

---

# 6. `hardware/`: detecció del maquinari

Aquesta carpeta consulta el sistema local.

## `hardware/detector.py`

Punt d’entrada principal:

```python
detect_hardware()
```

Coordina la resta de detectors i construeix un `HardwareProfile`.

Accepta un callback `on_step`, utilitzat per actualitzar la barra de progrés real de la TUI.

## `hardware/cpu.py`

Detecta:

- nom del processador;
- nuclis físics;
- nuclis lògics.

## `hardware/memory.py`

Detecta RAM total i RAM disponible mitjançant `psutil`.

## `hardware/storage.py`

Detecta els sistemes de fitxers útils i l’espai disponible.

Filtra mounts artificials o poc rellevants de WSL, `/run`, `/snap`, etc.

## `hardware/nvidia.py`

Utilitza NVML per detectar:

- GPU NVIDIA;
- VRAM total;
- VRAM disponible;
- driver;
- versió CUDA informada pel driver.

Si NVML o una GPU NVIDIA no estan disponibles, el projecte continua funcionant en mode CPU.

## `hardware/vulkan.py`

Detecta dispositius Vulkan a partir de `vulkaninfo --summary`. És el que permet proposar el
backend Vulkan de llama.cpp en màquines sense CUDA.

> Atenció metodològica: si `vulkaninfo` no està instal·lat, el backend no consta com a
> disponible i la selecció cau a CPU. Ara mateix «eina no instal·lada» encara es disfressa de
> «maquinari sense capacitat». Queda documentat com a limitació.

> El programa veu els recursos disponibles dins de l’entorn on s’executa. En WSL, la RAM visible pot no coincidir amb tota la RAM física de Windows.

---

# 7. `huggingface/`: accés i inspecció del Hub

Aquesta carpeta treballa amb un repositori concret de Hugging Face.

## `huggingface/url_parser.py`

Normalitza entrades com:

```text
Qwen/Qwen2.5-1.5B-Instruct
https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json
```

Totes es converteixen en un `repo_id` canònic.

## `huggingface/client.py`

Adaptador prim sobre `huggingface_hub`.

Responsabilitats:

- obtenir `model_info`;
- llistar fitxers;
- obtenir fitxers petits;
- obtenir metadades Safetensors;
- convertir excepcions externes en excepcions pròpies del projecte.

La interfície `HfClientProtocol` facilita l’ús de clients falsos als tests.

## `huggingface/classifiers.py`

Classifica el repositori a partir dels fitxers:

- Transformers;
- GGUF;
- Diffusers;
- ONNX;
- adapter;
- desconegut.

També agrupa les variants GGUF i detecta quantitzacions com `Q4_K_M` o `Q5_K_M`.

## `huggingface/repository.py`

Orquestra la inspecció profunda d’un repositori:

1. normalitza la referència;
2. consulta Hugging Face;
3. construeix la informació del repositori;
4. classifica els fitxers;
5. selecciona l’analitzador;
6. retorna un `ModelAnalysis`.

Funció principal:

```python
inspect_model()
```

---

## `huggingface/artifact_resolver.py`

Tria quin fitxer concret d’un repositori representa una quantització executable, i el
converteix en un `ModelArtifact` amb mida i checksum esperats.

---

# 8. `analyzers/`: interpretació segons el tipus de repositori

Els analitzadors converteixen la informació bruta d’un repositori en una configuració normalitzada.

## `analyzers/base.py`

Defineix el contracte comú dels analitzadors.

## `analyzers/registry.py`

Selecciona l’analitzador segons el tipus de repositori.

## `analyzers/transformers.py`

Interpreta `config.json` i extreu camps com:

- arquitectura;
- `model_type`;
- nombre de capes;
- `hidden_size`;
- caps d’atenció;
- caps KV;
- context màxim;
- `torch_dtype`;
- sliding window;
- indicadors MoE, MLA, multimodal o `auto_map`.

## `analyzers/gguf.py`

Analitza repositoris amb un o diversos fitxers `.gguf` i les seves variants de quantització.

## `analyzers/diffusers.py`

Analitza repositoris Diffusers, normalment utilitzats per models de generació d’imatges.

Actualment el workflow guiat està centrat en models de text.

## `analyzers/onnx.py`

Detecta i descriu artefactes ONNX.

## `analyzers/generic.py`

Fallback per repositoris que no encaixen en un analitzador específic.

---

# 9. `metadata/`: enriquiment de repositoris GGUF

Un repositori GGUF sovint conté el fitxer quantitzat, però no conté tota la configuració original del model. Aquesta carpeta intenta recuperar-la sense descarregar els pesos complets.

## `metadata/base_model_resolver.py`

Intenta identificar el model base utilitzant evidència estructurada:

1. `base_model` de la model card;
2. metadades internes del GGUF;
3. URLs o referències declarades;
4. pistes de nom només com a ajuda, no com a prova definitiva.

Retorna també la procedència i la confiança de la resolució.

## `metadata/range_reader.py`

Fa peticions HTTP `Range` per llegir només els primers bytes d’un fitxer GGUF.

Característiques importants:

- lectura en streaming;
- límit màxim de bytes;
- creixement progressiu del rang;
- detecció de servidors que ignoren `Range`;
- evita descarregar accidentalment tot el model.

## `metadata/gguf_reader.py`

Parser mínim de la taula de metadades de les versions GGUF compatibles.

No carrega els tensors ni els pesos.

## `metadata/config_merger.py`

Combina:

```text
metadades del fitxer GGUF
+
config.json del model base
```

La configuració de l’artefacte GGUF té prioritat quan hi ha conflictes, perquè representa el fitxer concret que s’executarà.

## `metadata/service.py`

Orquestra tot el procés d’enriquiment:

1. llegeix la capçalera GGUF;
2. resol el model base;
3. recupera el seu `config.json`;
4. fusiona la configuració;
5. retorna warnings i procedència.

## `metadata/policies.py`

Constants centralitzades, com:

- rang inicial;
- factor de creixement;
- límit màxim de lectura;
- timeouts.

---

# 10. `estimator/`: estimació de memòria

Aquesta carpeta respon:

> Quanta memòria necessitaria aquesta configuració concreta?

La fórmula conceptual és:

```text
memòria total =
    pesos
  + KV cache
  + overhead del runtime
  + reserva del dispositiu
  + marge de seguretat
```

## `estimator/service.py`

Servei principal:

```python
estimate_memory()
```

Coordina:

1. estimació dels pesos;
2. enriquiment GGUF;
3. selecció de la configuració efectiva;
4. estimació del KV cache;
5. overhead;
6. reserva;
7. marge;
8. compatibilitat amb RAM/VRAM;
9. recomanació de runtime.

## `estimator/weights.py`

Calcula la memòria dels pesos.

Prioritats:

### GGUF

Utilitza el mida real del fitxer de la variant seleccionada.

### Safetensors

Utilitza el recompte de paràmetres i els dtypes disponibles.

### Fallback

Utilitza la mida dels fitxers o una estimació teòrica segons bytes per paràmetre.

## `estimator/gguf_selection.py`

Selecciona una variant GGUF concreta.

Pot utilitzar una quantització demanada explícitament o aplicar una política per defecte.

## `estimator/kv_cache.py`

Estima la memòria del KV cache amb una fórmula equivalent a:

```text
2 × capes × caps_KV × dimensió_cap × context × batch × bytes_element
```

El factor `2` representa Key i Value.

Redueix la confiança o retorna desconegut per arquitectures no suportades, com alguns casos de MLA, MoE o multimodals.

## `estimator/overhead.py`

Heurística per representar:

- buffers;
- allocator;
- kernels;
- estructures del runtime;
- memòria temporal.

No és una mesura real.

## `estimator/compatibility.py`

Compara la memòria estimada amb els recursos disponibles.

Estats actuals:

- `comfortable`;
- `compatible`;
- `tight`;
- `offloading_required`;
- `insufficient`;
- `unknown`.

En mode automàtic prova, conceptualment:

1. GPU completa;
2. RAM + VRAM amb offloading;
3. CPU;
4. insuficient.

## `estimator/policies.py`

Conté els números i llindars centralitzats:

- bytes per dtype;
- marges;
- reserves;
- límits de compatibilitat;
- valors per defecte.

Quan es canviï una heurística de memòria, aquest és un dels primers fitxers que s’ha de revisar.

---

## `estimator/hardware_fit.py`

**Anàlisi d’encaix conscient de la col·locació.** No pregunta «hi cap?», sinó «on hi cap?»:

| Mode | Significat |
|---|---|
| `GPU_RESIDENT` | Tot el model viu a la VRAM |
| `GPU_OFFLOAD` | Part a VRAM, part a RAM del sistema |
| `CPU_RAM` | Només RAM, sense GPU |
| `TOO_LARGE` | No hi cap enlloc |

El resultat és un `HardwareFitResult`, que és el que fa que RAM i VRAM no es tractin mai com
una sola bossa de memòria. `analyze_components()` és la funció central.

## `estimator/artifact_analysis.py`

Interpreta quin artefacte real hi ha darrere d’una configuració proposada, per distingir una
quantització que existeix d’una que només és teòrica.

## `estimator/configuration.py`

Selecciona la configuració efectiva (quantització o precisió) segons la prioritat de
l’usuari, provant candidats en ordre fins a trobar-ne un amb compatibilitat acceptable.

---

# 11. `runtime/`: motors d’execució, capacitats i runners

Aquesta carpeta ja no només *proposa* un runtime: també el **troba**, comprova què sap fer i
l’executa. És la carpeta més gran del projecte després de la TUI.

## Proposta

### `runtime/service.py`

Dispatcher principal: GGUF → `llama.cpp`; Transformers → Hugging Face Transformers;
Transformers compatible i GPU adequada → vLLM com a possible alternativa; format no suportat
→ runtime desconegut.

### `runtime/llama_cpp.py` i `runtime/transformers.py`

Construeixen la recomanació orientativa de cada runtime: fitxer, context, capes GPU
aproximades, comandament de mostra i warnings.

### `runtime/vllm.py`

Comprova una llista conservadora d’arquitectures i condicions abans de proposar vLLM.

### `runtime/policies.py`

Constants del selector de runtime.

## Descobriment i capacitats

### `runtime/locator.py`

Descobriment i resolució dels binaris locals: on és `llama-cli`, quina versió, què hi ha al
`PATH`. Si no en troba cap, produeix també la guia d’instal·lació que es mostra a l’usuari.

### `runtime/backend_selection.py`

Selecció **pura** del backend de còmput a partir del maquinari ja detectat (CUDA, HIP, Vulkan,
CPU), amb el motiu de la tria.

### `runtime/llama_cpp_capability.py` i `runtime/pytorch_capability.py`

Inspeccionen les capacitats *observables* d’un binari o d’un entorn Python que ja existeixen:
no assumeixen res del que hauria de saber fer, ho comproven.

### `runtime/executability.py`

Valida una recomanació de runtime contra el maquinari detectat.

> Distinció important: **`executability`** és tècnica (aquest runtime pot carregar aquest
> artefacte?) i sí que entra al rànquing; **`runtime_readiness`** és operativa (aquesta màquina
> podria llançar-ho ara?) i **mai** entra al rànquing. Veure
> [recommendation.md](recommendation.md).

## Execució

### `runtime/llama_cpp_runner.py`

Executa un `ModelArtifact` GGUF ja descarregat i verificat amb `llama-cli`. Valida format
GGUF, `local_path` present, artefacte descarregat, verificat, fitxer existent i prompt no
buit, i després construeix un comandament `llama-cli --single-turn` amb `--ctx-size`,
`--n-gpu-layers`, `--model` i `--prompt`.

### `runtime/transformers_runner.py` i `runtime/transformers_worker.py`

Executen un repositori Transformers en un **worker Python aïllat**, en un procés a part. Així
un `torch` que peta o que es menja tota la memòria no s’emporta la TUI.

## Mesura

### `runtime/llama_bench_capability.py`, `llama_bench_runner.py`, `llama_bench_parser.py`

Comproven si `llama-bench` existeix, l’executen i converteixen la seva taula de resultats en
dades. El parser és pur i es pot provar sense binari.

### `runtime/transformers_benchmark_runner.py` i `transformers_benchmark_worker.py`

L’equivalent per Transformers, també en un worker aïllat.

> Els dos `*_worker.py` són punts d’entrada de subprocés: ningú no els importa, s’executen amb
> `python -m`. Per això la cobertura els compta al 0 % tot i estar provats a través del runner.

---

# 11.1. `artifacts/`: resolució, descàrrega i verificació

Aquesta carpeta converteix una recomanació abstracta en un fitxer local verificat.

## `artifacts/service.py`

Compon:

1. resolver d’artefactes;
2. storage local;
3. downloader injectable.

Flux:

```text
resolve(repo_id, quantization, revision)
   ↓
download(artifact)
   ↓
verify(artifact, full=False)
```

La descàrrega desa un sidecar `.sha256`. La verificació ràpida comprova existència, mida i sidecar; `full=True` recalcula el hash del fitxer.

## `artifacts/storage.py`

Defineix el layout local dels artefactes i els sidecars de SHA-256.

## `artifacts/ports.py`

Protocol del resolver. Permet provar el servei sense Hugging Face real.

## `artifacts/errors.py`

Errors específics:

- artefacte no trobat;
- format no suportat;
- error de descàrrega;
- error de verificació.

Limitació actual: només es resolen GGUF single-file. Repositoris Transformers i GGUF multipart es rebutgen explícitament.

---

# 11.2. `execution/`: execució de processos locals

Aquesta carpeta és deliberadament petita. No sap què és Hugging Face ni què és un model.

## `execution/host.py`

Backend host que executa un `ExecutionRequest` amb timeout i retorna `ExecutionResult`.
La durada es mesura amb `time.perf_counter()` des de l'arrencada correcta del
procés fins que acaba o és matat per timeout. Durant aquesta finestra es
mostreja cada 50 ms el RSS del procés principal (`peak_ram_bytes`) i, si NVML
pot atribuir memòria al PID, la memòria NVIDIA del procés (`peak_vram_bytes`).
En CPU-only o si NVML falla, VRAM queda com `None`; la inferència no falla per
absència de mètrica. Com que és sampling, pics molt curts poden quedar
infraestimats.

## `execution/ports.py`

Protocol injectable per substituir l’execució real als tests.

## `execution/errors.py`

Errors d’execució:

- executable absent;
- timeout;
- procés amb exit code no zero.

---

# 11.3. `execution_plans/`: identitat lògica i plans

Un mateix model lògic pot tenir moltes maneres d’executar-se. Aquesta carpeta les construeix,
amb funcions pures sobre objectes de domini.

## `execution_plans/service.py`

- `resolve_model_identity()` — quin model *lògic* hi ha darrere d’un repositori (l’original
  Transformers i les seves conversions GGUF són el mateix model).
- `build_execution_plan()` — un `ExecutionPlan` = artefacte + runtime + backend +
  compatibilitat + predicció de memòria (+ readiness quan es coneix).

`_compatibility()` respon **només** a artefacte ↔ runtime. Que un binari estigui instal·lat o
no és una altra pregunta, i es guarda a part.

## `execution_plans/quantization.py`

Semàntica de quantització compartida, inclosos AWQ i GPTQ.

---

# 11.4. `evaluation/`, `experiments/` i `benchmarks/`: evidència

Aquí és on el projecte deixa d’estimar i comença a mesurar.

## `experiments/`

- `runner.py` — orquestra **un** experiment controlat d’inferència. Exigeix readiness `READY`:
  aquí sí que cal el binari.
- `storage.py` — magatzem JSON de `ExperimentRecord`, immutables.
- `reevaluation.py` — reavaluació *offline* de registres antics, sense tornar a executar res.

## `benchmarks/`

- `matrix.py` — executa matrius petites de benchmarks sense posar bucles a la interfície.
- `storage.py` — magatzem JSON de `BenchmarkRecord`.

## `evaluation/`

Funcions pures, sense E/S:

- `comparison.py` — compara una predicció de memòria amb una execució observada:

  ```text
  error_bytes   = mesurat − predit
  error_percent = (mesurat − predit) / predit × 100
  ```

  Un error positiu vol dir que Jaull ha **infraestimat**. La comparació de RAM només es
  calcula quan l’execució és CPU-only o sense offload; amb offload, Jaull encara no separa
  host i dispositiu i la comparació queda marcada `methodologically_unavailable`. La de VRAM
  ho està sempre en aquesta fase.
- `benchmark_comparison.py` — compara benchmarks com a **plans d’execució complets**, no com a
  runtimes aïllats.
- `hardware_fingerprint.py` — identitat estable de la màquina, per no comparar mesures fetes
  en ordinadors diferents.

---

# 12. `discovery/`: cerca i selecció preliminar de models

Aquesta carpeta troba candidats abans de fer les operacions cares. Els models de dades
(`ModelCandidate`, `EvaluatedCandidate`) viuen ara a `domain/candidates.py`.

## `discovery/query_builder.py`

Tradueix els requisits en diverses consultes al Hub: `instruct`, `chat`, `coder`,
`multilingual`, formats preferits, idiomes, tendència. Fer diverses consultes evita dependre
d’una sola cadena de cerca.

## `discovery/search_client.py`

Crida `HfApi.list_models()` i normalitza els resultats en `ModelCandidate`. Encara no
inspecciona cap repositori a fons.

## `discovery/candidate_filter.py`

Fa dues feines ben diferents, i convé no confondre-les.

**Filtrar** — rebutja només el que és genuïnament inservible: repositoris privats, gated,
pipeline incorrecte, multimodals fora d’abast, adapters sense model base, llicències
incompatibles quan l’ús comercial és obligatori. Les metadades primes **no** són mai un
rebuig: es converteixen en una penalització registrada i menys confiança.

**Fer el shortlist** — decidir quins candidats es mereixen una de les 12 inspeccions
profundes. Des de l’agost del 2026 aquesta tria és **conscient del maquinari**:
`coarse_placement_hint()` dedueix, només amb metadades, si un candidat cauria a VRAM, a
offload, a RAM o enlloc, i `_FIT_BONUS` pondera la cua amb aquesta pista:

| Mode | Bonus |
|---|---|
| `GPU_RESIDENT` | +4.0 |
| `GPU_OFFLOAD` | +3.0 |
| `CPU_RAM` | +1.5 |
| `TOO_LARGE` | −9.0 |

L’objectiu és que les tres col·locacions viables continuïn representades, en comptes de gastar
els 12 llocs en una sola classe de mida.

`parameter_count_hint()` dedueix una mida aproximada del nom (`…-7B-…`). Aquesta dada
**només** ordena la cua d’inspecció: no es persisteix, no s’informa i no entra al rànquing.
Després de la inspecció, `MemoryEstimate` i `HardwareFitResult` són la font de veritat.

## `discovery/enrichment.py`

Converteix un candidat preliminar en un `EvaluatedCandidate`:

1. inspecciona el repositori;
2. selecciona una configuració;
3. estima la memòria;
4. calcula els subscores;
5. captura errors individuals sense aturar tot el workflow.

## `discovery/grouping.py` i `discovery/series.py`

`grouping.py` agrupa repositoris que representen la mateixa família quan hi ha evidència de
`base_model` (l’original Transformers i la conversió GGUF declarada són el mateix model). No
hauria d’agrupar per semblança textual del nom.

`series.py` agrupa per **sèrie**: mateixa família, mides de paràmetres diferents (0.5B · 1.5B ·
3B · 7B).

---

# 13. `workflow/`: orquestració del mode guiat

Aquesta carpeta **ja no és el centre del producte**. Continua sent l’índex del mode guiat —
decideix l’ordre— però els casos d’ús que abans tenia (normalització de requisits, polítiques
de recomanació, pressupostos, telemetria, cablejat) han marxat a `application/`,
`observability/` i `bootstrap/`. Els camins antics sobreviuen com a shims.

## `workflow/orchestrator.py`

Fitxer central del mode guiat.

### `scan_hardware()`

Executa la detecció del maquinari i informa del progrés real.

### `run_workflow()`

Recorregut complet:

```text
UserAnswers
   ↓
build_requirements
   ↓
build_queries
   ↓
search
   ↓
_interleave
   ↓
filter + shortlist (conscient del maquinari)
   ↓
_evaluate  (inspecció + estimació + HardwareFit)
   ↓
recommend  (plans → engine v2 → diversificació)
   ↓
RecommendationWorkflowState
```

Gestiona progrés, cancel·lació, errors globals de Hugging Face, errors d’un sol candidat,
caché i el cas de zero resultats.

### `_interleave()`

Barreja els resultats de les diferents consultes en round-robin, perquè les primeres consultes
no omplin tot el límit i deixin fora les cerques de GGUF o d’idiomes.

### `_evaluate()`

Inspecciona el shortlist amb inspeccions concurrents i fa servir cachés per no repetir
anàlisis ni estimacions dins de la mateixa execució.

## `workflow/state.py`

`RecommendationWorkflowState`: hardware, respostes, requisits, candidats, candidats avaluats,
recomanacions, progrés, warnings i errors.

## `workflow/models.py`

Models de les preguntes i dels passos del workflow.

## `workflow/cache.py` i `workflow/model_analysis_cache.py`

`cache.py` és la caché en memòria d’una sola execució. `model_analysis_cache.py` és un shim
cap a la caché **persistent** de `adapters/`, que és la que fa que repetir una cerca no torni
a inspeccionar els mateixos repositoris.

## `workflow/progress.py`

Models i callbacks de progrés independents de Textual.

## `workflow/telemetry.py`

Shim cap a `observability/telemetry.py`.

## Shims de compatibilitat

Aquests fitxers ja no contenen lògica; només re-exporten:

| Fitxer | Apunta a |
|---|---|
| `workflow/requirements.py` | `application/requirements.py` |
| `workflow/policies.py` | `application/recommendation/policies.py` |
| `workflow/ranking.py` | `application/recommendation/service.py` |
| `workflow/container.py` | `bootstrap/container.py` |
| `workflow/model_analysis_cache.py` | `adapters/cache/model_analysis_cache.py` |
| `workflow/telemetry.py` | `observability/telemetry.py` |

El codi nou hauria d’importar de la destinació, no del shim.

---

# 14. `recommendation/`: avaluació, rànquing i explicacions

Aquesta carpeta decideix quins candidats apareixen al resultat final. El canvi conceptual més
important del projecte viu aquí: **ja no hi ha una puntuació única que ho decideixi tot**.

## `recommendation/engine_v2.py`

El motor actual. Avalua **plans d’execució**, no repositoris: cada candidat que sobreviu a la
inspecció s’expandeix en les maneres concretes en què es podria executar (una per variant
d’artefacte i runtime), i són aquests plans els que s’ordenen.

`assess_plan()` omple un `PlanAssessment` amb eixos separats:

| Eix | Què respon |
|---|---|
| `suitability` | Encaixa amb la tasca declarada? |
| `capability` | Senyal de família + nombre de paràmetres |
| `feasibility` | Hi cap, en aquest maquinari? |
| `executability` | El pla és tècnicament coherent (artefacte ↔ runtime)? |
| `execution_fitness` | Els dos anteriors combinats |
| `performance_evidence` | Hi ha benchmark mesurat d’aquest pla en aquesta màquina? |
| `confidence` | Confiança de l’estimació de base |
| `runtime_readiness` | **Només operatiu.** Es podria llançar ara? Mai es rankeja |

`_ranking_key()` construeix una tupla lexicogràfica per prioritat: la prioritat decideix *quin
eix es consulta primer*, no quant pesa. Totes les tuples acaben igual —evidència, confiança,
`repo_id`, quantització, runtime— de manera que els empats es trenquen de forma determinista.

Quatre `HardConstraint` eliminen un pla del tot: artefacte incompatible amb el runtime,
memòria insuficient, llicència incompatible i idioma incompatible. **Un binari que falta no hi
és**, i això és deliberat: veure [recommendation.md](recommendation.md).

## `recommendation/diversity.py`

Col·lapsa els plans ordenats abans que es converteixin en recomanacions. El millor pla d’una
`ModelIdentity` és el primari; els altres plans del mateix model queden com a alternatives i
no gasten cap dels cinc llocs. Si el següent candidat té la mateixa signatura d’avaluació que
el líder, prefereix un que canviï de família, de mida o de perfil d’execució.

## `recommendation/capability.py`

Senyal de capacitat derivat de les metadades inspeccionades: família i nombre de paràmetres.

## `recommendation/actionability.py`

Respon «això arrencaria de veritat?»: distingeix un artefacte confirmat d’un de probable i
d’un de purament teòric. Un pla especulatiu es continua rankejant, però no pot encapçalar la
targeta com a `BEST MATCH`.

## `recommendation/tier.py`

Tria el titular de la targeta (`BEST MATCH`, `RECOMMENDED`, `CLOSEST OPTION`,
`BEST-EFFORT SUGGESTION`) a partir de compatibilitat, confiança, penalització dura i
actionability. Existeix perquè una targeta que sempre crida *BEST MATCH* és deshonesta.

## `recommendation/requirements_gate.py`

Comprovacions de requisits durs que penalitzen la puntuació: ús comercial, idiomes,
concurrència.

## `recommendation/scoring.py` i `recommendation/policies.py`

El score compost de vuit components, amb els pesos i els modificadors per prioritat. Continua
existint, però **ja no ordena el mode guiat**: només s’utilitza al camí sense maquinari, i
s’exporta a l’informe com a `score_breakdown` perquè l’esquema del report és un contracte
byte a byte. Ja no es mostra a la TUI.

`policies.py` també conté la taula conservadora de llicències, les paraules clau per tasca i
els factors de confiança.

## `recommendation/ranker.py`

Combina els subscores del camí sense maquinari, aplica confiança i ordena de manera
determinista. També decideix les etiquetes de les alternatives.

## `recommendation/explanations.py`

Genera els textos amb regles, no amb un LLM: coincidència de tasca, idioma declarat, marge de
memòria, llicència, warning de confiança, offloading, concurrència no modelada.

## `recommendation/report.py` i `recommendation/models.py`

`report.py` és un shim cap a `reporting/recommendation.py`. `models.py` conté els models
finals que veu l’usuari: `ScoreBreakdown`, `ModelRecommendation`, `SeriesSibling`.

---

# 15. `presentation/`: sortida de la CLI

Aquesta carpeta converteix models de domini en sortides Rich o JSON.

## `presentation/hardware_report.py`

Renderitza el resultat de `scan`.

## `presentation/model_report.py`

Renderitza el resultat de `inspect`.

## `presentation/estimation_report.py`

Renderitza l’estimació de memòria. La representació JSON estable ja no viu aquí: es
re-exporta de `reporting/estimation.py`, que n’és l’únic productor.

## `presentation/execution_report.py`

Renderitza el resultat d’una execució real: sortida, durada, observació del procés.

## `presentation/comparison_report.py`

Renderitza les comparacions predicció ↔ observació i benchmark ↔ benchmark.

## `presentation/plan_labels.py`

Etiquetes en llenguatge humà per a models, artefactes i plans d’execució. Aquí viu
`runtime_block_reason()`, el motiu únic que mostren Run, Validate i Benchmark quan el runtime
no està llest, perquè cap de les tres pantalles s’inventi la seva pròpia frase.

## `presentation/console.py`

Crea i configura la consola Rich i formata bytes.

---

# 15.1. `reporting/`: serialització

`reporting/estimation.py` és **l’únic** productor de la representació JSON d’un
`MemoryEstimate`. `reporting/recommendation.py` és l’únic que construeix l’informe complet
d’una execució guiada, en JSON i en Markdown.

Els dos contractes són **byte a byte**: `tests/test_reporting_regression.py` compara la
sortida contra `tests/snapshots/report.json` i `report.md`. Qualsevol canvi que trenqui la
igualtat exacta obliga a pujar `REPORT_SCHEMA_VERSION` explícitament.

---

# 15.2. `diagnostics/`: comprovacions d’entorn

`diagnostics/service.py` recull l’estat de Python, xarxa, API de Hugging Face, NVML, runtimes
i caché. És el que hi ha darrere de `jaull doctor` i de la pantalla Doctor.

Diagnostica l’entorn; no decideix quin model és millor.

---

# 15.3. `observability/`: telemetria

`observability/telemetry.py` són comptadors i temps lleugers per a les etapes llargues
(`filter`, `deep_inspection`, …). No és un sistema de mètriques ni surt de la màquina.

---

# 16. `tui/`: interfície interactiva

La TUI utilitza Textual. És la carpeta més gran del projecte.

## `tui/app.py`

Aplicació principal (`JaullApp`):

- registra pantalles;
- manté el `HardwareProfile` de l’execució;
- manté l’últim `RecommendationWorkflowState`;
- injecta l’`AdvisorService`;
- controla la navegació.

## `tui/styles.tcss`

Estils visuals. S’ha d’incloure dins del wheel perquè la TUI instal·lada funcioni fora del
repositori.

## `tui/evidence.py`, `tui/artifact_preparation.py`, `tui/palette.py`

Lògica de suport compartida entre pantalles: resum de l’evidència d’un pla, preparació
d’artefactes amb progrés, i la paleta de colors.

## `tui/screens/`

### Flux guiat

| Pantalla | Què fa |
|---|---|
| `welcome.py` | Mode guiat, eines avançades, sortir |
| `hardware_analysis.py` | Detecció en un worker, amb progrés real |
| `requirements_wizard.py` | Recull les respostes de l’usuari |
| `model_discovery.py` | Executa el workflow llarg en segon pla; permet cancel·lar |
| `recommendation_results.py` | Millor recomanació, alternatives, compatibilitat, memòria, raons, warnings, exportació |

### Del resultat a l’evidència

| Pantalla | Què fa |
|---|---|
| `execution_paths.py` | Els plans d’execució d’una recomanació: variants, backend, estat **Ready** i comparació entre plans |
| `recommendation_execution.py` | Execució real d’un sol torn (llama.cpp o Transformers) |
| `recommendation_validation.py` | Validació: executa i desa un `ExperimentRecord` |
| `recommendation_benchmark.py` | Benchmark: `llama-bench` o worker de Transformers, amb registre persistit |

Quan el runtime no està llest, aquestes tres últimes **no s’amaguen**: mostren el motiu que
retorna `presentation/plan_labels.runtime_block_reason()` en comptes d’invocar un binari que
no hi és. Descarregar sí que es permet: baixar un artefacte no necessita runtime.

### Eines avançades

`advanced_tools.py` és el menú; `scan.py`, `inspect.py`, `estimate.py` i `doctor.py` són les
versions visuals de les comandes.

## `tui/widgets/`

Components reutilitzables: banner, logo, targetes resum, barra de memòria, badge
d’avaluació, passos de progrés, panell de warnings, detalls tècnics, capçalera del workflow i
el comandament CLI equivalent.

Hi ha també dos widgets decoratius dibuixats amb blocs de quadrant (U+2596–U+259F):
`ocean.py` a la pantalla inicial i `patrol.py` mentre dura la cerca, amb el motor compartit a
`subpixel.py` i l’art generat a `shark_art.py`, `swim_art.py` i `fin_art.py`. No informen de
res: existeixen perquè una pantalla quieta durant minuts sembla un programa penjat.

Aquests widgets no haurien de contenir mai l’algoritme de recomanació.

---

# 17. `cli/`: comandes no interactives

## `cli/scan.py`

Detecta i mostra el maquinari.

## `cli/inspect.py`

Analitza un repositori concret de Hugging Face.

## `cli/estimate.py`

Construeix una `InferenceConfiguration`, executa `estimate_memory()` i mostra Rich o JSON.

## `cli/doctor.py`

Comprova:

- versió de Python;
- Internet;
- API de Hugging Face;
- NVML;
- GPU NVIDIA;
- caché;
- dependències.

`doctor` diagnostica l’entorn, però no decideix quin model és millor.

## `cli/experiments.py`

Llista i mostra els `ExperimentRecord` que ja s’han desat, sense executar res.

## `cli/run.py`

Executa un artefacte GGUF concret amb `llama-cli`.

Flux:

```text
referència Hugging Face
   ↓
normalize_repo_id()
   ↓
AdvisorService.resolve_artifact()
   ↓
AdvisorService.download_artifact() si falta
   ↓
AdvisorService.verify_artifact()
   ↓
LlamaCppRunner(HostExecutionBackend).run()
   ↓
stdout del model
```

Codis de sortida principals:

- `0`: execució correcta;
- `2`: referència de model invàlida;
- `3`: quantització no disponible;
- `4`: error de resolució, descàrrega o verificació d’artefacte;
- `5`: error d’execució o de `llama-cli`.

`run` és l’únic subcomandament que descarrega pesos i executa inferència. La resta de camins (`inspect`, `estimate`, TUI i workflow guiat) continuen sent metadata-only.

---

# 18. `exceptions.py`

Excepcions pròpies del projecte:

- referència invàlida;
- model no trobat;
- accés denegat;
- Hugging Face no disponible;
- error de configuració;
- error de detecció de maquinari;
- quantització inexistent;
- capçalera GGUF invàlida;
- resolució del model base.

L’objectiu és que la resta del projecte no depengui directament dels tipus d’error de `huggingface_hub`, `httpx` o NVML.

---

# 19. Recorregut complet d’una recomanació

Exemple: l’usuari demana chat general, prioritat equilibrada, anglès i 2–5 usuaris.

```text
 1. JaullApp
    └── obre WelcomeScreen

 2. HardwareAnalysisScreen
    └── advisor.scan_hardware()
        └── hardware.detect_hardware()  →  HardwareProfile

 3. RequirementsWizardScreen
    └── UserAnswers

 4. application.requirements.build_requirements()
    └── UserRequirements

 5. discovery.query_builder.build_queries()
    └── llista de SearchQuery

 6. discovery.search_client.HfSearchClient.search()
    └── ModelCandidate barats

 7. orchestrator._interleave()
    └── barreja les consultes en round-robin

 8. candidate_filter.filter_candidates()
    └── elimina el que és inservible; les metadades primes només penalitzen

 9. candidate_filter.shortlist(hardware=…)
    └── 12 llocs, ponderats per coarse_placement_hint

10. discovery.enrichment.evaluate_candidate()   × 12, concurrent
    ├── inspect_model()
    ├── select_configuration()
    ├── estimate_memory()  →  MemoryEstimate + HardwareFitResult
    └── score_candidate()

11. execution_plans.resolve_model_identity() + build_execution_plan()
    └── un pla per variant d’artefacte × runtime

12. recommendation.engine_v2.rank_execution_plans()
    ├── assess_plan()   → PlanAssessment (eixos separats)
    └── _ranking_key()  → ordre lexicogràfic segons la prioritat

13. recommendation.diversity.diversify_ranked_plans()
    └── un lloc per model lògic; la resta, alternatives

14. recommendation.tier + recommendation.explanations
    └── titular de la targeta, raons i warnings

15. RecommendationResultsScreen
    └── mostra i exporta

16. ExecutionPathsScreen  (opcional, ja fora del workflow)
    └── prepara → executa → valida → benchmark
```

---

# 20. On s’ha de tocar segons el canvi que es vulgui fer?

| Objectiu | Fitxers principals |
|---|---|
| Afegir una pregunta al wizard | `workflow/models.py`, `application/requirements.py`, `tui/screens/requirements_wizard.py` |
| Canviar les cerques de Hugging Face | `discovery/query_builder.py`, `discovery/search_client.py` |
| Canviar quins candidats arriben a inspecció | `discovery/candidate_filter.py`, `application/recommendation/policies.py` |
| Afegir un nou tipus de repositori | `huggingface/classifiers.py`, `analyzers/registry.py`, nou analitzador |
| Millorar la lectura GGUF | `metadata/range_reader.py`, `metadata/gguf_reader.py` |
| Canviar fórmules de memòria | `estimator/*`, especialment `estimator/policies.py` |
| Canviar l’encaix al maquinari (VRAM/offload/RAM) | `estimator/hardware_fit.py` |
| Canviar la selecció de quantització | `estimator/configuration.py`, `recommendation/policies.py` |
| **Canviar l’ordre de les recomanacions** | `recommendation/engine_v2.py` (`_ranking_key`, `assess_plan`) |
| Canviar quantes vegades surt el mateix model | `recommendation/diversity.py` |
| Canviar el titular de la targeta | `recommendation/tier.py`, `recommendation/actionability.py` |
| Canviar textos explicatius | `recommendation/explanations.py` |
| Canviar el runtime proposat | `runtime/service.py`, `runtime/policies.py` |
| Canviar com es troben els binaris | `runtime/locator.py`, `runtime/backend_selection.py` |
| Canviar execució amb `llama-cli` | `cli/run.py`, `runtime/llama_cpp_runner.py`, `execution/*` |
| Canviar execució de Transformers | `runtime/transformers_runner.py`, `runtime/transformers_worker.py` |
| Canviar benchmarks | `benchmarks/matrix.py`, `runtime/llama_bench_*` |
| Canviar resolució/descàrrega/verificació d’artefactes | `artifacts/*`, `huggingface/artifact_resolver.py`, `domain/artifacts.py` |
| Canviar el workflow complet | `workflow/orchestrator.py` |
| Canviar el cablejat de serveis | `bootstrap/container.py` |
| Canviar la interfície | `tui/screens/`, `tui/widgets/`, `tui/styles.tcss` |
| Canviar el format exportat | `reporting/recommendation.py` (i pujar `REPORT_SCHEMA_VERSION`) |

---

# 21. Ordre recomanat per estudiar el codi

No és necessari llegir els fitxers en ordre alfabètic.

## Primera volta: entendre el flux

1. `cli/app.py`
2. `tui/app.py`
3. `advisor/service.py` — la façana per on passa tot
4. `workflow/state.py`
5. `workflow/orchestrator.py`
6. `bootstrap/container.py`

## Segona volta: entendre com es troben models

7. `application/requirements.py`
8. `discovery/query_builder.py`
9. `discovery/search_client.py`
10. `discovery/candidate_filter.py` — filtre **i** shortlist, no ho confonguis
11. `discovery/enrichment.py`

## Tercera volta: entendre la memòria

12. `huggingface/repository.py`
13. `analyzers/transformers.py`
14. `analyzers/gguf.py`
15. `metadata/service.py`
16. `estimator/service.py`
17. `estimator/weights.py`
18. `estimator/kv_cache.py`
19. `estimator/hardware_fit.py`
20. `estimator/compatibility.py`

## Quarta volta: entendre per què guanya un model

21. `domain/execution_plans.py`
22. `execution_plans/service.py`
23. `domain/recommendation.py` — `PlanAssessment` i els seus eixos
24. `recommendation/engine_v2.py` — `assess_plan` i `_ranking_key`
25. `recommendation/diversity.py`
26. `recommendation/tier.py`
27. `recommendation/explanations.py`
28. `application/recommendation/service.py` — com es lliga tot

## Cinquena volta: comprovar el comportament

29. `tests/test_workflow_orchestrator.py`
30. `tests/test_recommendation_engine_v2.py`
31. `tests/test_recommendation_runtime_agnostic.py`
32. `tests/test_recommendation_diversity.py`
33. `tests/test_discovery_search.py`
34. `tests/test_hardware_fit_scenarios.py`
35. `tests/test_architecture_dependencies.py` — les fronteres que no es poden trencar

## Sisena volta: entendre execució i evidència

36. `domain/artifacts.py`
37. `artifacts/service.py`
38. `runtime/locator.py`
39. `runtime/llama_cpp_runner.py`
40. `execution/host.py`
41. `experiments/runner.py`
42. `evaluation/comparison.py`
43. `cli/run.py`

Els tests són una forma molt útil d’entendre quines regles es consideren contracte del
sistema.

---

# 22. Limitacions actuals que cal recordar

1. **El rànquing no és una mesura directa de qualitat del model.** Mesura adequació segons
   les regles actuals, i només té evidència mesurada quan s’ha executat un benchmark.
2. **La concurrència és aproximada.** No hi ha benchmark multiusuari ni model complet de
   throughput.
3. **Int8 i int4 de Transformers poden ser teòrics.** Que la memòria estimada càpiga no
   garanteix un artefacte o runtime vàlid; per això existeix `actionability`.
4. **La popularitat no equival a qualitat.** Només és una senyal secundària.
5. **Les llicències personalitzades queden com a desconegudes.** El projecte no ofereix
   assessorament legal.
6. **La mida del model no demostra automàticament més qualitat.**
7. **L’offloading és aproximat.** Encara no es calcula capa per capa, i amb offload Jaull no
   separa host i dispositiu: la comparació de RAM queda `methodologically_unavailable`.
8. **La VRAM no es pot comparar encara.** El model d’estimació no reté la VRAM atribuïda al
   PID executat.
9. **No hi ha model explícit de workload ni de SLO.** No es demanen objectius de tokens/s,
   TTFT ni latència.
10. **El workflow està limitat a text-generation.** Imatge, àudio i RAG complet queden fora.
11. **Les dades de les model cards poden ser incompletes o incorrectes.** El sistema redueix
    la confiança, però no pot corregir-ho tot.
12. **`jaull run` és explícit i limitat.** Només GGUF single-file amb `llama-cli`; els
    Transformers només s’executen des de la TUI, amb worker aïllat.
13. **Una eina que falta encara es pot confondre amb maquinari sense capacitat.** Si
    `vulkaninfo` o NVML no hi són, el backend no consta com a disponible. Això sí que està
    resolt per als *runtimes* (veure `runtime_readiness`), però no encara per a la detecció
    de maquinari.

---

# 23. Idea mental per recordar el projecte

```text
Hardware
   +
Necessitats humanes
   ↓
Requisits tècnics
   ↓
Cerca barata de candidats
   ↓
Shortlist conscient del maquinari
   ↓
Inspecció profunda (12 llocs)
   ↓
Memòria + HardwareFit
   ↓
Plans d’execució concrets
   ↓
Avaluació per eixos + rànquing explicable
   ↓
Diversificació → 5 recomanacions
   ↓
Executar · Validar · Mesurar
```

La carpeta més important per entendre **l’ordre** és `workflow/`.

La carpeta més important per entendre **els càlculs de memòria** és `estimator/`.

La carpeta més important per entendre **per què guanya un model** és `recommendation/`, i
concretament `engine_v2.py`.

La carpeta més important per entendre **si això arrencaria de veritat** és `runtime/`.
