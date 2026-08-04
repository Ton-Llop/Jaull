# Arquitectura de `jaull`

Aquest document explica com està organitzat el projecte, quina responsabilitat té cada carpeta i quin recorregut segueix una recomanació des que l’usuari obre la interfície fins que es genera l’informe final.

> Aquest README descriu l’estat actual del codi. No és una especificació definitiva: algunes heurístiques del recomanador encara s’han de validar amb benchmarks reals.

---

## 1. Què fa el projecte?

`jaull` és una eina de terminal que:

1. Detecta el maquinari disponible.
2. Recull les necessitats de l’usuari mitjançant un assistent guiat.
3. Cerca models públics al Hugging Face Hub.
4. Analitza els repositoris candidats.
5. Estima la memòria necessària per executar-los.
6. Selecciona una configuració tècnica probable.
7. Puntua els candidats i mostra recomanacions explicades.

El projecte ofereix dues formes d’ús:

- **Mode guiat:** l’usuari respon preguntes senzilles i rep recomanacions automàtiques.
- **Eines avançades:** permet executar manualment `scan`, `inspect`, `estimate` i `doctor`.

---

## 2. Visió general de l’arquitectura

```text
CLI / TUI
   │
   ▼
WorkflowOrchestrator
   │
   ├── Detecció de maquinari
   ├── Traducció de requisits
   ├── Descobriment de candidats
   ├── Inspecció de repositoris
   ├── Estimació de memòria
   ├── Selecció de configuració
   ├── Puntuació i rànquing
   └── Generació de l’informe
```

La idea principal és separar:

- **Presentació:** què veu l’usuari.
- **Orquestració:** en quin ordre s’executen les operacions.
- **Lògica de negoci:** com es calcula, filtra o puntua.
- **Domini:** com es representen les dades.
- **Infraestructura externa:** maquinari, Hugging Face i HTTP.

Aquesta separació permet reutilitzar els mateixos serveis des de la CLI, la TUI i els tests.

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
├── analyzers/
├── cli/
├── discovery/
├── domain/
├── estimator/
├── hardware/
├── huggingface/
├── metadata/
├── presentation/
├── recommendation/
├── runtime/
├── tui/
├── workflow/
├── exceptions.py
├── __init__.py
└── __main__.py
```

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
```

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

### `domain/enums.py`

Enumeracions generals, com ara:

- tipus de repositori;
- formats;
- estat dels diagnòstics.

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

# 11. `runtime/`: recomanació del motor d’execució

Aquesta carpeta intenta convertir una estimació en una ruta d’execució.

## `runtime/service.py`

Dispatcher principal:

- GGUF → `llama.cpp`;
- Transformers → Hugging Face Transformers;
- Transformers compatible i GPU adequada → vLLM com a possible alternativa;
- format no suportat → runtime desconegut.

## `runtime/llama_cpp.py`

Genera una recomanació orientativa per a `llama.cpp`, incloent:

- fitxer GGUF;
- context;
- nombre aproximat de capes GPU;
- comandament de mostra;
- warnings.

## `runtime/transformers.py`

Genera una configuració orientativa per carregar el model amb la biblioteca Transformers.

## `runtime/vllm.py`

Comprova una llista conservadora d’arquitectures i condicions abans de proposar vLLM.

## `runtime/policies.py`

Constants del selector de runtime.

> Actualment la recomanació de runtime és orientativa. Que un model càpiga en memòria no garanteix que tots els backends o quantitzacions siguin executables al maquinari detectat.

---

# 12. `discovery/`: cerca i selecció preliminar de models

Aquesta carpeta troba candidats abans de fer les operacions cares.

## `discovery/models.py`

Models principals:

- `SearchQuery`
- `ModelCandidate`
- `EvaluatedCandidate`

`ModelCandidate` conté metadades barates de cerca.

`EvaluatedCandidate` ja inclou anàlisi, estimació, configuració i subscores.

## `discovery/query_builder.py`

Tradueix els requisits en diverses consultes al Hub.

Exemples de conceptes utilitzats:

- `instruct`;
- `chat`;
- `coder`;
- `multilingual`;
- formats preferits;
- idiomes;
- tendència o popularitat.

Fer diverses consultes evita dependre d’una sola cadena de cerca.

## `discovery/search_client.py`

Crida `HfApi.list_models()` i normalitza els resultats en `ModelCandidate`.

No inspecciona encara profundament cada repositori.

## `discovery/candidate_filter.py`

Executa el filtratge barat:

- deduplicació;
- repositoris privats;
- gated;
- pipeline incorrecte;
- multimodals fora d’abast;
- adapters incomplets;
- llicències incompatibles quan és obligatori l’ús comercial;
- penalitzacions per metadades absents.

També crea el shortlist que passarà a inspecció profunda.

La funció `parameter_count_hint()` intenta deduir una mida aproximada a partir del nom, com `7B`. Aquesta dada només serveix per ordenar la cua d’inspecció i no s’hauria de presentar com una mesura fiable.

## `discovery/enrichment.py`

Converteix un candidat preliminar en un `EvaluatedCandidate`:

1. inspecciona el repositori;
2. selecciona una configuració;
3. estima la memòria;
4. calcula els subscores;
5. captura errors individuals sense aturar tot el workflow.

## `discovery/grouping.py`

Agrupa repositoris que representen la mateixa família quan existeix evidència de `base_model`.

Exemple:

```text
model original Transformers
+
conversió GGUF declarada
=
mateixa família
```

No hauria d’agrupar únicament per semblança textual del nom.

---

# 13. `workflow/`: orquestració del mode guiat

Aquesta carpeta és l’índex del producte. Decideix l’ordre del procés, però delega els càlculs.

## `workflow/orchestrator.py`

Fitxer central del mode guiat.

Funcions principals:

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
filter + shortlist
   ↓
evaluate candidates
   ↓
recommend
   ↓
RecommendationWorkflowState
```

Gestiona:

- progrés;
- cancel·lació;
- errors globals de Hugging Face;
- errors d’un candidat;
- caché en memòria;
- casos sense resultats.

### `_interleave()`

Barreja els resultats de les diferents consultes en round-robin.

Això evita que les primeres consultes omplin tot el límit i deixin fora les cerques de GGUF o d’idiomes.

### `_evaluate()`

Inspecciona el shortlist i utilitza cachés per no repetir anàlisis o estimacions dins de la mateixa execució.

## `workflow/models.py`

Models de les preguntes i requisits:

- cas d’ús;
- prioritat;
- idioma;
- concurrència;
- escala documental;
- ús comercial;
- passos i estats del workflow.

## `workflow/requirements.py`

Tradueix respostes humanes a requisits tècnics.

Exemple:

```text
“2–5 usuaris”
→ concurrent_users = 3
→ concurrency_range = "2-5 users"
```

També decideix:

- context orientatiu;
- `pipeline_tag`;
- formats preferits;
- política de llicència.

## `workflow/state.py`

Defineix `RecommendationWorkflowState`, l’objecte que acompanya tota l’execució:

- hardware;
- respostes;
- requisits;
- candidats;
- candidats avaluats;
- recomanacions;
- progrés;
- warnings;
- errors.

## `workflow/container.py`

Contenidor senzill de dependències.

Agrupa serveis com:

- detector de maquinari;
- client de cerca;
- client de repositori;
- inspector;
- estimador;
- factory del lector HTTP Range.

Permet substituir dependències reals per fakes als tests.

## `workflow/cache.py`

Caché en memòria limitada a una execució guiada.

No és persistència ni base de dades.

## `workflow/progress.py`

Models i callbacks de progrés independents de Textual.

## `workflow/policies.py`

Límits centralitzats del workflow, com:

- resultats per consulta;
- candidats únics;
- candidats d’inspecció profunda;
- recomanacions finals.

---

# 14. `recommendation/`: selecció, puntuació i rànquing

Aquesta carpeta decideix quins candidats apareixen al resultat final.

## `recommendation/configuration.py`

Selecciona automàticament una configuració per candidat.

### GGUF

Prova quantitzacions en un ordre que depèn de la prioritat:

- qualitat;
- equilibri;
- velocitat;
- memòria.

Selecciona una variant que existeixi realment al repositori i que tingui una compatibilitat acceptable.

### Transformers

Prova precisions com:

- float16;
- int8;
- int4.

Les opcions int8 i int4 poden ser teòriques si el repositori no publica un artefacte directament executable amb aquella quantització. En aquests casos es redueix la confiança.

## `recommendation/scoring.py`

Calcula sis subscores:

1. ajust al maquinari;
2. ajust a la tasca;
3. ajust als idiomes;
4. llicència;
5. qualitat de metadades;
6. popularitat.

Limitació important: actualment no mesura directament la qualitat real de les respostes del model.

## `recommendation/policies.py`

Conté:

- pesos del score;
- modificadors segons prioritat;
- taula conservadora de llicències;
- paraules clau per tasca;
- factors de confiança;
- heurístiques de concurrència.

Aquest és un dels fitxers principals quan es modifica l’algoritme de recomanació.

## `recommendation/ranker.py`

Combina els subscores, aplica confiança i ordena de manera determinista.

També decideix:

- quins estats poden ser recomanació principal;
- desempats;
- màxim de resultats;
- etiquetes de les alternatives.

Limitació actual: una etiqueta del tipus “Higher quality but tighter fit” no hauria de deduir-se només perquè un artefacte consumeix més memòria.

## `recommendation/service.py`

Compon:

1. scoring;
2. agrupació per família;
3. rànquing;
4. explicacions;
5. selecció de la recomanació principal i alternatives.

## `recommendation/explanations.py`

Genera textos explicatius amb regles, no amb un LLM.

Exemples:

- coincideix amb la tasca;
- idioma declarat;
- cap a la memòria;
- llicència;
- warning de confiança;
- offloading;
- concurrència no modelada.

## `recommendation/report.py`

Exporta el resultat a:

- JSON;
- Markdown.

Inclou hardware, requisits, candidats, scores, raons, warnings i recomanacions.

## `recommendation/models.py`

Models finals que veu l’usuari:

- `ScoreBreakdown`
- `ModelRecommendation`

---

# 15. `presentation/`: sortida de la CLI

Aquesta carpeta converteix models de domini en sortides Rich o JSON.

## `presentation/hardware_report.py`

Renderitza el resultat de `scan`.

## `presentation/model_report.py`

Renderitza el resultat de `inspect`.

## `presentation/estimation_report.py`

Renderitza l’estimació de memòria i produeix la representació JSON estable.

## `presentation/console.py`

Crea i configura la consola Rich i formata bytes.

---

# 16. `tui/`: interfície interactiva

La TUI utilitza Textual.

## `tui/app.py`

Aplicació principal:

- registra pantalles;
- manté el `HardwareProfile` de l’execució;
- manté l’últim `RecommendationWorkflowState`;
- injecta el `ServiceContainer`;
- controla la navegació;
- conserva les eines avançades.

## `tui/styles.tcss`

Estils visuals de la interfície.

S’ha d’incloure dins del wheel perquè la TUI instal·lada funcioni fora del repositori.

## `tui/screens/`

### `welcome.py`

Pantalla inicial:

- mode guiat;
- eines avançades;
- sortir.

### `hardware_analysis.py`

Executa la detecció de maquinari en un worker i mostra progrés.

### `requirements_wizard.py`

Recull les respostes de l’usuari.

### `model_discovery.py`

Executa el workflow llarg en segon pla, actualitza el progrés i permet cancel·lar.

### `recommendation_results.py`

Mostra:

- millor recomanació;
- alternatives;
- score;
- compatibilitat;
- memòria;
- raons;
- warnings;
- exportació.

### `advanced_tools.py`

Menú de les utilitats manuals.

### `scan.py`, `inspect.py`, `estimate.py`, `doctor.py`

Versions visuals de les comandes avançades.

## `tui/widgets/`

Components reutilitzables:

- banner;
- logo;
- targetes resum;
- barra de memòria;
- badge de compatibilitat;
- barra de score;
- passos de progrés;
- targeta de recomanació;
- panell de warnings;
- comandament CLI equivalent.

Aquests widgets no haurien de contenir l’algoritme de recomanació.

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
1. LocalAiCheckApp
   └── obre WelcomeScreen

2. HardwareAnalysisScreen
   └── workflow.scan_hardware()
       └── hardware.detect_hardware()
           └── HardwareProfile

3. RequirementsWizardScreen
   └── UserAnswers

4. workflow.requirements.build_requirements()
   └── UserRequirements

5. discovery.query_builder.build_queries()
   └── llista de SearchQuery

6. discovery.search_client.HfSearchClient.search()
   └── ModelCandidate barats

7. orchestrator._interleave()
   └── barreja consultes

8. candidate_filter.deduplicate()
   └── elimina repo_id repetits

9. candidate_filter.filter_candidates()
   └── elimina incompatibles evidents

10. candidate_filter.shortlist()
    └── selecciona candidats per inspecció profunda

11. discovery.enrichment.evaluate_candidate()
    ├── inspect_model()
    ├── select_configuration()
    ├── estimate_memory()
    └── score_candidate()

12. discovery.grouping.collapse_families()
    └── agrupa original i conversions quan hi ha evidència

13. recommendation.ranker
    └── score final + ordre determinista

14. recommendation.explanations
    └── raons i warnings

15. RecommendationResultsScreen
    └── mostra i exporta el resultat
```

---

# 20. On s’ha de tocar segons el canvi que es vulgui fer?

| Objectiu | Fitxers principals |
|---|---|
| Afegir una pregunta al wizard | `workflow/models.py`, `workflow/requirements.py`, `tui/screens/requirements_wizard.py` |
| Canviar les cerques de Hugging Face | `discovery/query_builder.py`, `discovery/search_client.py` |
| Canviar quins candidats arriben a inspecció | `discovery/candidate_filter.py`, `workflow/policies.py` |
| Afegir un nou tipus de repositori | `huggingface/classifiers.py`, `analyzers/registry.py`, nou analitzador |
| Millorar la lectura GGUF | `metadata/range_reader.py`, `metadata/gguf_reader.py` |
| Canviar fórmules de memòria | `estimator/*`, especialment `estimator/policies.py` |
| Canviar la selecció de quantització | `recommendation/configuration.py`, `recommendation/policies.py` |
| Canviar el score | `recommendation/scoring.py`, `recommendation/policies.py`, `recommendation/ranker.py` |
| Canviar textos explicatius | `recommendation/explanations.py` |
| Canviar el runtime proposat | `runtime/*` |
| Canviar el workflow complet | `workflow/orchestrator.py` |
| Canviar la interfície | `tui/screens/`, `tui/widgets/`, `tui/styles.tcss` |
| Canviar el format exportat | `recommendation/report.py`, `presentation/estimation_report.py` |

---

# 21. Ordre recomanat per estudiar el codi

No és necessari llegir els fitxers en ordre alfabètic.

## Primera volta: entendre el flux

1. `cli/app.py`
2. `tui/app.py`
3. `workflow/models.py`
4. `workflow/state.py`
5. `workflow/orchestrator.py`
6. `workflow/container.py`

## Segona volta: entendre com es troben models

7. `workflow/requirements.py`
8. `discovery/query_builder.py`
9. `discovery/search_client.py`
10. `discovery/candidate_filter.py`
11. `discovery/enrichment.py`

## Tercera volta: entendre la memòria

12. `huggingface/repository.py`
13. `analyzers/transformers.py`
14. `analyzers/gguf.py`
15. `metadata/service.py`
16. `estimator/service.py`
17. `estimator/weights.py`
18. `estimator/kv_cache.py`
19. `estimator/compatibility.py`

## Quarta volta: entendre per què guanya un model

20. `recommendation/configuration.py`
21. `recommendation/scoring.py`
22. `recommendation/policies.py`
23. `recommendation/ranker.py`
24. `recommendation/service.py`
25. `recommendation/explanations.py`

## Cinquena volta: comprovar el comportament

26. `tests/test_workflow_orchestrator.py`
27. `tests/test_recommendation_configuration.py`
28. `tests/test_recommendation_ranking.py`
29. `tests/test_discovery_search.py`
30. `tests/test_memory_estimator.py`

Els tests són una forma molt útil d’entendre quines regles es consideren contracte del sistema.

---

# 22. Limitacions actuals que cal recordar

1. **El score no és una mesura directa de qualitat del model.** Mesura adequació segons les regles actuals.
2. **La concurrència és aproximada.** No hi ha benchmark multiusuari ni model complet de throughput.
3. **Int8 i int4 de Transformers poden ser teòrics.** Que la memòria estimada càpiga no garanteix un artefacte o runtime vàlid.
4. **La popularitat no equival a qualitat.** Només és una senyal secundària.
5. **Les llicències personalitzades queden com a desconegudes.** El projecte no ofereix assessorament legal.
6. **La mida del model no demostra automàticament més qualitat.** Calen benchmarks o metadades de qualitat fiables.
7. **L’offloading és aproximat.** Encara no es calcula capa per capa.
8. **No es mesuren tokens/s ni TTFT.** La compatibilitat actual és sobretot de memòria.
9. **El workflow està limitat a text-generation.** Imatge, àudio i RAG complet encara queden fora.
10. **Les dades de les model cards poden estar incompletes o ser incorrectes.** El sistema redueix la confiança, però no pot corregir totes les mancances.

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
Inspecció profunda d’un shortlist
   ↓
Configuració + memòria + runtime
   ↓
Score explicable
   ↓
Recomanació i warnings
```

La carpeta més important per entendre **l’ordre** és `workflow/`.

La carpeta més important per entendre **els càlculs de memòria** és `estimator/`.

La carpeta més important per entendre **per què guanya un model** és `recommendation/`.
