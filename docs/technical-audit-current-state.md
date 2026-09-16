# Auditoria tecnica del estado actual de Jaull

Auditoria original: 2026-09-03, branch `execution-plan`, commit
`0910967e0768548b5fd84c83427a1754daf5816c`.

**Revision: 2026-09-16, branch `master`, commit `5bf954b`.**

El cuerpo del documento (secciones A-H) se conserva tal como se escribio el 03/09: es un
registro fechado, no un documento vivo. La seccion R de mas abajo dice que ha cambiado desde
entonces. Donde un hallazgo esta cerrado, lleva una marca **CERRADO (16/09)** en su sitio con
la evidencia. Lo que no lleva marca sigue abierto.

Esta auditoria describe el estado real del repositorio en el commit indicado. No presupone
que cambios existentes en otras branches formen parte de este estado.

---

## R. Revision del 2026-09-16

### R.1 Gates

| Gate | 03/09 | 16/09 |
|---|---|---|
| ruff | clean | clean |
| mypy | clean | clean, 244 ficheros |
| pytest | 1381 | **1613 passed** |
| cobertura | 84 % | **85 %** (17 271 stmts, 2 603 sin cubrir) |
| allowlist arquitectura | vacia | vacia |
| TODO/FIXME/HACK en `src/` | — | **0** |

### R.2 Hallazgos cerrados

| Id | Hallazgo | Evidencia de cierre |
|---|---|---|
| D.1 | Memoria incompleta podia producir `comfortable` | `estimator/compatibility.py:211` corta antes: si falta weights, KV u overhead devuelve `assess(None, …)` con motivo `"Cannot confirm compatibility: missing …"`. |
| D.2 | Matching de evidencia local demasiado amplio | `recommendation/local_evidence.py:22-48` compara backend seleccionado, backend observado, `inference_configuration` completa y los flags de runtime incluido `--n-gpu-layers`. |
| E.1 | Fallback por bytes devolvia `TOO_LARGE` | La sonda del propio documento (`weights=1000, VRAM=800, RAM=300`, 10 blocks) devuelve hoy `gpu_offload 8/10`. La busqueda usa una cota cerrada documentada como que nunca descarta una colocacion viable. |
| E.2 | `WeightEstimate` sin descomposicion estructural | `transformer_block_decomposition` es ya la autoridad y llega a `assess_components_with_fit`. |
| E.3 | Capacity y ocupacion mezcladas | Separadas y nombradas: `planning_accelerator_memory_bytes` (capacidad, usada por `discovery/candidate_filter.py:713` y `workflow/orchestrator.py:406`) frente a `available_accelerator_memory_bytes` (disponibilidad observada, usada por `estimator/compatibility.py:174` y `hardware_fit.py:728`). El no-determinismo entre dos runs sigue existiendo, pero ahora es deliberado. |
| E.6 | Hardware Fit de aceleradores no NVIDIA | Cerrado por el fix de Vulkan (`ea26ffc`). Las dos vistas se fusionan en `domain/hardware.py`. Verificado: una RX 7800 XT de 16 GB sin entrada NVML da `gpu_resident 32/32`. La memoria sale de DRM sysfs, no de Vulkan; ver R.4. |
| — | La comparacion prediccion/observacion no producia ningun numero | El observation contract (`3a6dc26`) anade una segunda fuente de medicion. `runtime/llama_cpp_memory_report.py` parsea los buffers de llama.cpp y `evaluation/comparison.py:274` los usa cuando NVML no atribuye. |

### R.3 Hallazgos que siguen abiertos

- **E.4 `AdvisorService` sigue creciendo:** 1305 -> **1483** lineas. `engine_v2.py` 1084,
  `orchestrator.py` 699. La direccion es la contraria a la recomendada.
- **E.5 Replayability depende del caller:** `prediction_input` sigue opcional en
  `advisor/service.py:493`, `evaluation/experiments.py:40` y `domain/experiments.py:149,221,239`.
- **Workers sin cobertura:** `transformers_benchmark_worker.py` 158/158 y
  `transformers_worker.py` 76/76 sin cubrir.
- **F: dos taxonomias en `workflow/`** y la capa `adapters/ports` parcial. Sin cambios.

### R.4 Hallazgos nuevos

1. **El camino HIP/ROCm esta en master sin validar en hardware.** El mensaje del commit
   `a53bc11` lo dice: *"pending hardware validation"*. Deteccion, seleccion de backend y ruta
   de lanzamiento existen y tienen tests unitarios; ninguna medicion en una GPU AMD los
   respalda. Documentado ahora en `limitations.md`. Lo mismo vale para la ejecucion Vulkan,
   ejercitada solo a traves de un bug de deteccion reportado por un usuario, no de una
   ejecucion medida.
2. **La memoria de aceleradores no-NVIDIA es mas estrecha de lo que sugiere el cierre de
   E.6.** No viene de Vulkan sino de DRM sysfs
   (`/sys/class/drm/card*/device/mem_info_vram_total`), emparejado por PCI vendor/device ID.
   En la practica: **AMD discreta sobre Linux con driver estilo amdgpu**. No funciona en
   Windows, ni en Intel o Apple, ni con dos tarjetas identicas — el resumen de Vulkan no
   lleva PCI bus ID y una coincidencia ambigua se descarta en vez de adivinarse. Ademas, sin
   `mem_info_vram_used` la tarjeta entra en la shortlist pero no en Hardware Fit, que exige
   disponibilidad observada.
3. **El tier de recomendacion solo podia valer `best_effort`.** `tier.py:choose_tier` regla 2
   devolvia `BEST_EFFORT` para confianza LOW/UNKNOWN, y `scoring.py:319 confidence_of` lee la
   confianza del `MemoryEstimate`, cuyo eslabon debil es el `runtime_overhead` siempre
   `ASSUMED`. De cuatro tiers solo uno era alcanzable: sobre un perfil RTX 4060 los cinco
   resultados salian `best_effort`.

   **CERRADO (16/09).** `LOW` ya no cortocircuita a `BEST_EFFORT`; limita a `RECOMMENDED`,
   que es la lectura correcta — significa que la estimacion lleva dentro una suposicion de
   politica documentada, no que la colocacion sea desconocida. `UNKNOWN` sigue siendo limite
   duro. En el mismo perfil RTX 4060, los `best_effort` que quedan son todos por la regla 1
   (requisito duro incumplido: licencia no confirmada como comercial en dos casos, idioma no
   confirmado en el tercero), que es lo que debe pasar.

   **Matiz que hay que mantener en la memoria del TFG: `BEST_MATCH` sigue sin ser alcanzable
   en produccion.** `estimator/overhead.py:39` emite `EstimationConfidence.LOW`
   incondicionalmente — es la unica que ese componente puede producir — y
   `_combined_confidence` se queda con la mas debil de las cuatro, asi que la confianza global
   de un `MemoryEstimate` nunca pasa de `LOW`, y la regla 7 pide `HIGH`. El balance real es
   **1 tier alcanzable -> 3** (`best_effort` / `recommended` / `closest_option`), no 4. El
   cuarto queda abierto hasta que se calibre el `runtime_overhead`, que es trabajo aparcado
   por falta de una segunda maquina.

   De paso se cerro un agujero preexistente que este cambio dejaba mas cerca de ser
   alcanzable: la regla 4 atrapaba `OFFLOADING_REQUIRED` y `UNKNOWN` pero no `INSUFFICIENT`,
   asi que un modelo que no cabe caia hasta la regla 5 o la 7 (`insufficient` + `high` ->
   `best_match`). No era alcanzable por la ruta de produccion — forzado con un perfil de
   1 GiB VRAM / 2 GiB RAM, ningun `insufficient` llega a mostrarse — pero `choose_tier` es
   publica y su propio test lo alcanzaba. `INSUFFICIENT` esta ahora en la regla 4.
4. **Tres modelos sub-1.5B ocupan el Top 5 por delante de un 4B.** Cadena verificada:
   `capability` colapsa 0.5B y 4B en el mismo bucket `adequate` (la separacion en la curva de
   tamano, 0.158, es menor que el ancho de bucket, 0.17); al empatar decide
   `execution_fitness`, que es funcion pura del estado de memoria, y el mas pequeno gana. Por
   debajo, el 55 % de `capability_score` no mide capacidad: a Qwen3-4B le falta la linea
   `language:` en el model card y eso le cuesta mas (-0.123) de lo que le dan 3.4B de
   parametros extra (+0.106). El fix de `2f85c45` es real pero ortogonal: baja los modelos con
   estado `UNKNOWN`, no los pequenos con estado `comfortable`.
5. **Politica de "confirmed memory" duplicada** en `engine_v2._prioritize_confirmed_memory_fit`
   y `ranker.select_ranked:172`. Solo la primera corre con hardware; divergiran.
6. **Rama muerta en `estimator/service.py:315-319`:** las dos ramas de `_sum_components`
   devuelven `sum(known)`. El aviso que describe el comentario lo da hoy D.1 en
   `compatibility.py`. Cosmetico.

### R.5 Documentacion corregida en esta revision

Afirmaciones que el codigo habia dejado falsas, ya corregidas:

- `limitations.md`: la VRAM ya no es "solo NVIDIA"; anadida la ruta real (DRM sysfs) con sus
  limites, la segunda fuente de observacion, el desglose por componentes y el aviso de que
  HIP/ROCm no esta validado.
- `evidence.md`, `architecture.md`, `Workflow.md`, `Glossari.md`,
  `experimental-protocol-biosfer.md`, `README.md`: decian que la VRAM estaba *siempre*
  `methodologically_unavailable` y que con offload faltaba una separacion host/dispositivo.
  Lo primero dejo de ser cierto con `3a6dc26`; lo segundo nunca lo fue — el obstaculo es el
  `mmap`, que hace que el RSS siga el artefacto y no la colocacion.
- `estimation.md`: la VRAM ya no viene solo de NVML.
- `evaluation/comparison.py`: el docstring de `compare_prediction` contradecia al de
  `_predicted_ram` veinte lineas mas abajo.

## A. Estado general

| Area | Valoracion | Motivo |
|---|---|---|
| Arquitectura | buena | Fronteras verificadas por AST, allowlist vacia y composicion centralizada; quedan `AdvisorService` y `workflow/` sobredimensionados. |
| Correctitud | aceptable | El camino habitual funciona, pero hay un falso positivo con memoria incompleta y evidencia local insuficientemente identificada. |
| Robustez | aceptable | Buenos fallbacks y contratos, aunque uno de los fallbacks de offload tiene una busqueda incorrecta. |
| Tests | muy buena | 1381 tests, 84% de cobertura global y cobertura alta en estimacion y ranking. Faltan algunos casos semanticos importantes. |
| Mantenibilidad | aceptable | Buen dominio y typing, pero archivos de 1100-1300 lineas, legacy y documentacion desfasada. |
| Preparacion para continuar | buena | La shortlist hardware-aware ya existe; antes de usarla experimentalmente deben cerrarse los P0. |

Dato importante: la descomposicion de pesos esta en el commit `a880093` de `hfa-layer`,
pero no es ancestro del commit auditado. En esta branch todavia no forma parte de
`WeightEstimate`.

## B. Arquitectura actual

El flujo real es:

```text
Hardware scan + respuestas
        |
        v
UserRequirements
        |
        v
queries Hugging Face -> deduplicacion -> maximo 40 repos
        |
        v
filtro + Hardware-aware coarse shortlist -> maximo 12
        |
        v
ModelAnalysis cache / inspect_model / config y artifacts
        |
        v
seleccion de quant/dtype -> MemoryEstimator
        |
        v
weights + KV + overhead + reserve + margin
        |
        v
HardwareFitResult + CompatibilityAssessment
        |
        v
ExecutionPlans -> PlanAssessment -> ranking v2 -> diversity -> Top 5
        |
        v
resolucion/descarga fisica del artifact
        |
        v
execution planner -> llama.cpp / Transformers
        |
        v
observation -> comparison -> experiment/benchmark persistence
```

El pipeline esta coordinado en
[`workflow/orchestrator.py`](../src/jaull/workflow/orchestrator.py#L104). La metadata del
artifact interviene antes del ranking; la descarga fisica ocurre despues, al preparar la
ejecucion.

La arquitectura es un modular monolith con `domain`, una capa `application` todavia parcial,
composition root en [`bootstrap/container.py`](../src/jaull/bootstrap/container.py#L26),
infraestructura agrupada por feature y presentacion CLI/TUI.

## C. Lo que esta bien disenado

- `HardwareFitResult` conserva placement, topologia, split GPU/RAM, diagnosticos y
  compatibilidad legacy en un modelo congelado:
  [`domain/estimation.py`](../src/jaull/domain/estimation.py#L167).
- HFA no suma RAM y VRAM, distingue memoria unificada y permanece runtime-agnostic. Existe
  un guard especifico que impide conectarlo accidentalmente con `--n-gpu-layers`:
  [`test_architecture_dependencies.py`](../tests/test_architecture_dependencies.py#L125).
- Los pesos GGUF usan tamano real del artifact; KV usa arquitectura, contexto, batch y
  concurrencia: [`estimator/kv_cache.py`](../src/jaull/estimator/kv_cache.py#L85).
- Overhead, reserve, margenes y ladders estan centralizados y documentados en
  [`estimator/policies.py`](../src/jaull/estimator/policies.py#L43).
- La shortlist coarse ya es barata, determinista, consciente de CPU/GPU/RAM y no persiste
  sus estimaciones como verdad final:
  [`discovery/candidate_filter.py`](../src/jaull/discovery/candidate_filter.py#L277).
- Balanced mantiene el gate de runnability antes de capability sin convertirlo en score:
  [`recommendation/engine_v2.py`](../src/jaull/recommendation/engine_v2.py#L1068).
- Diversity solo intercambia planes con la misma firma semantica:
  [`recommendation/diversity.py`](../src/jaull/recommendation/diversity.py#L85).
- El planner de ejecucion es la autoridad comun de CLI/TUI y todavia no consume HFA:
  [`application/execution/planner.py`](../src/jaull/application/execution/planner.py#L28).
- La re-evaluation usa hardware y metadata congelados, sin detectar el host ni consultar la
  red: [`experiments/reevaluation.py`](../src/jaull/experiments/reevaluation.py#L115).
- Stores de experimentos y benchmarks son atomicos, y artifacts protege traversal y
  checksum.

## D. Problemas criticos

### 1. Memoria incompleta puede producir `comfortable`

> **CERRADO (16/09).** `estimator/compatibility.py:211` devuelve `assess(None, ...)` con
> motivo explicito cuando falta weights, KV u overhead. Ver R.2.

**Problema:** un total parcial puede clasificarse como si fuera completo.

**Donde:** [`_sum_components()`](../src/jaull/estimator/service.py#L301), fallback en
[`compatibility.py`](../src/jaull/estimator/compatibility.py#L215), y aceptacion de ladder en
[`configuration.py`](../src/jaull/estimator/configuration.py#L107).

**Por que ocurre:** si falta KV u overhead, se suma lo conocido y se usa el assessment
legacy. El servicio baja `confidence` a `UNKNOWN`, pero conserva `status=comfortable`; v2
traduce ese status a feasibility `STRONG` sin comprobar la confianza:
[`engine_v2.py`](../src/jaull/recommendation/engine_v2.py#L670).

**Impacto:** puede detener prematuramente una ladder, recomendar una precision demasiado
grande o presentar como ejecutable una configuracion cuyo KV es desconocido. Un probe de
auditoria produjo `comfortable / high / fit=None` con KV ausente; el servicio completo rebaja
la confianza, no el status.

**Severidad:** CRITICA.

**Solucion recomendada:** representar el total como lower bound y no emitir
`comfortable/compatible/tight` cuando falta un componente requerido. La ladder no debe tratar
ese resultado como fit confirmado.

### 2. Matching de evidencia local demasiado amplio

> **CERRADO (16/09).** `recommendation/local_evidence.py` usa ya una clave canonica
> completa, flags de runtime incluidos. Ver R.2.

**Problema:** un benchmark o experimento diferente puede influir en el ranking actual.

**Donde:** [`_matching_experiment()`](../src/jaull/recommendation/engine_v2.py#L855),
[`_matching_benchmark()`](../src/jaull/recommendation/engine_v2.py#L869) y
[`_artifact_matches()`](../src/jaull/recommendation/engine_v2.py#L887).

**Por que ocurre:** solo se comparan artifact basico, runtime, maquina y, a veces, backend.
Se ignoran revision/hash, contexto, token sizes, `n_gpu_layers`, runtime version, metodologia
y workload. Despues la evidencia recibe confianza `HIGH`, y la memoria observada se reduce a
`max(RAM, VRAM)`: [`engine_v2.py`](../src/jaull/recommendation/engine_v2.py#L905).

**Impacto:** resultados no comparables pueden alterar performance evidence, confidence y
orden. Es especialmente peligroso para la calibracion del TFG.

**Severidad:** CRITICA para instalaciones con historico experimental; no afecta a una
instalacion sin records.

**Solucion recomendada:** crear una unica clave canonica de comparabilidad reutilizando los
ejes que ya aplica
[`evaluation/benchmark_comparison.py`](../src/jaull/evaluation/benchmark_comparison.py#L248),
y conservar RAM/VRAM como medidas separadas.

## E. Problemas importantes pero no criticos

### 1. Busqueda incorrecta en el fallback por bytes

> **CERRADO (16/09).** La sonda de este documento devuelve hoy `gpu_offload 8/10`. Ver R.2.

**Problema:** puede devolver `TOO_LARGE` aunque exista un split valido.

**Donde:** [`estimator/hardware_fit.py`](../src/jaull/estimator/hardware_fit.py#L317).

**Por que ocurre:** `_build_offload_result()` devuelve `None` tanto por exceso de GPU como de
RAM, pero el binary search siempre reduce GPU allocation.

**Impacto:** el probe `weights=1000, VRAM=800, RAM=300` devuelve `TOO_LARGE`, aunque
`GPU=700/RAM=300` cabe. Hoy el estimator productivo conoce blocks siempre que conoce KV, por
lo que afecta principalmente a la API directa y scripts.

**Severidad:** ALTA.

**Solucion recomendada:** resolver la interseccion de los limites minimo por RAM y maximo por
VRAM, con un regression test RAM-constrained.

### 2. `WeightEstimate` aun no posee la descomposicion estructural

> **CERRADO (16/09).** `transformer_block_decomposition` es ya la autoridad. Ver R.2.

**Problema:** HFA sigue calculando `ceil(total_weights / blocks)`.

**Donde:** [`domain/estimation.py`](../src/jaull/domain/estimation.py#L120) y
[`estimator/hardware_fit.py`](../src/jaull/estimator/hardware_fit.py#L232).

**Por que ocurre:** el commit de descomposicion existe en otra punta de branch y no esta
integrado aqui.

**Impacto:** embeddings/head inflan el coste marginal por bloque y pueden hacer el placement
conservador.

**Severidad:** ALTA.

**Solucion recomendada:** integrar la descomposicion como autoridad de `WeightEstimate`,
inicialmente diagnostica y sin asumir placement de non-block weights.

### 3. Capacity y ocupacion actual siguen mezcladas

> **CERRADO (16/09).** Separadas en `planning_` frente a `available_`. Ver R.2.

**Problema:** la shortlist usa capacidad fisica, pero HFA usa RAM/VRAM libre instantanea.

**Donde:** [`candidate_filter.py`](../src/jaull/discovery/candidate_filter.py#L303) y
[`hardware_fit.py`](../src/jaull/estimator/hardware_fit.py#L91).

**Impacto:** dos ejecuciones en la misma maquina pueden producir distinta compatibilidad y
Top 5 por procesos ajenos.

**Severidad:** MEDIA-ALTA.

**Solucion recomendada:** modelar explicitamente `planning capacity` frente a
`run-now availability`; no cambiar formulas hasta decidir que contrato consume cada caso de
uso.

### 4. `AdvisorService` sigue siendo facade y composition root parcial

> **ABIERTO (16/09).** Ha crecido de 1305 a 1483 lineas.

**Problema:** tiene 1305 lineas, muchas dependencias y construye servicios concretos mediante
memoizacion mutable sobre una dataclass frozen.

**Donde:** [`advisor/service.py`](../src/jaull/advisor/service.py#L110) y
[`advisor/service.py`](../src/jaull/advisor/service.py#L901).

**Impacto:** alto coste de cambio y tests con fakes grandes, aunque el grafo no este roto.

**Severidad:** MEDIA.

**Solucion recomendada:** reducirlo gradualmente a facade, desplazando construccion hacia
`bootstrap`, sin crear servicios ceremoniales.

### 5. Replayability depende del caller

> **ABIERTO (16/09).** `prediction_input` sigue opcional.

**Problema:** `ExperimentPredictionInput` es opcional y el snapshot productivo se adjunta
desde la validacion TUI:
[`recommendation_validation.py`](../src/jaull/tui/screens/recommendation_validation.py#L256).

**Impacto:** callers programaticos pueden persistir records modernos no reproducibles.

**Severidad:** MEDIA.

**Solucion recomendada:** centralizar la captura en el caso de uso que crea experimentos
cuando los inputs esten disponibles.

### 6. Hardware Fit de aceleradores no NVIDIA

> **CERRADO (16/09), con matices.** La memoria llega via DRM sysfs, no via Vulkan: en la
> practica AMD discreta sobre Linux. Ver R.2 y R.4.2.

**Problema:** Vulkan detecta AMD/Intel, pero no obtiene memoria; HFA consume `gpus`, poblado
por NVML.

**Donde:** [`hardware/detector.py`](../src/jaull/hardware/detector.py#L45) y
[`estimator/hardware_fit.py`](../src/jaull/estimator/hardware_fit.py#L689).

**Impacto:** esas maquinas caen a CPU para el memory fit aunque el backend exista.

**Severidad:** MEDIA; esta documentado y no bloquea la campana NVIDIA prevista.

**Solucion recomendada:** mantenerlo como limitacion hasta disponer de una fuente fiable de
memoria por vendor.

## F. Problemas menores / deuda tecnica

- `workflow/` conserva orchestration real y varios shims muy usados por tests; no es codigo
  muerto, pero prolonga dos taxonomias.
- `ScoreBreakdown` y el ranker legacy siguen activos para `hardware=None` y reporting
  compatible. Estan aislados del ranking v2 normal.
- `engine_v2.py` y `AdvisorService` son grandes, pero partirlos ahora sin cambiar
  responsabilidades aportaria poco.
- La capa `adapters/ports` es parcial: formaliza cache, mientras runtimes, Hugging Face,
  hardware y stores siguen organizados por feature.
- La documentacion contiene afirmaciones ya falsas sobre RAM/VRAM y
  `PredictionComparison`: [`estimation.md`](estimation.md#L79) y
  [`experimental-protocol-biosfer.md`](experimental-protocol-biosfer.md#L262).
- No se encontro un modulo productivo demostrablemente muerto que sea seguro eliminar sin
  revisar compatibilidad publica.

## G. Hardware Fit Analyzer

1. **Formulas:** componentes y conservacion por pool son coherentes. GGUF real, KV
   arquitectonico y heuristicas estan diferenciados. Los dos defectos son el fallback por
   bytes y el coste uniforme basado en todos los pesos.
2. **Arquitectura:** correcta en lo esencial. Es runtime-agnostic, consume estimaciones y no
   decide ranking.
3. **Autoridad duplicada:** si. Actualmente HFA inventa la distribucion por blocks porque
   `WeightEstimate` no incluye la descomposicion.
4. **Offloading:** razonable como capacity planning: pools separados, block-aware, KV
   proporcional, unified memory explicita y diagnostics exactos. No representa la asignacion
   interna de llama.cpp.
5. **Limitaciones:** blocks uniformes, non-block no separado, overhead/margin heuristicos,
   ocupacion instantanea, una sola GPU efectiva, NVIDIA para VRAM y fallback por bytes
   defectuoso.
6. **No tocar todavia:** mapping a `n_gpu_layers`, `+1`, tensor parsing, constantes calibradas
   con Qwen, cambios de reserve/overhead/margin, multi-GPU o ranking.

## H. Recommendation Engine

El ranking v2 es ordinal y explicito:
[`engine_v2.py`](../src/jaull/recommendation/engine_v2.py#L958). Hard constraints cubren
artifact/runtime, memoria insuficiente, licencia comercial e idioma declarado; runtime
instalado no altera la recomendacion.

No existe un sesgo universal hacia modelos pequenos: capability recompensa escala y Balanced
permite que capability decida dentro de la misma clase de runnability. Si existe un riesgo
condicionado: un HFA conservador y el gate de Balanced pueden hacer que un modelo pequeno
residente quede por encima de otro mayor con offload. Esa es una policy defendible, pero debe
validarse contra workloads reales, no asumirse como calidad.

El problema mas serio del engine no es su tuple de ranking, sino la evidencia local
insuficientemente comparable. Tampoco predice rendimiento: popularity/activity solo aporta
un prior pequeno dentro de capability, no TPS o latency.

## I. Tests y CI

- `1381 passed` en `183.51s` con coverage.
- Coverage global: `84%`.
- HFA `91%`, estimator `92%`, weights `95%`, KV `97%`, engine v2 `89%`, diversity
  `97%`, shortlist `96%`, orchestrator `91%` y execution planner `98%`.
- Ruff: OK. Mypy strict: OK. Architecture tests: 4 passed. Compileall: OK.
  `git diff --check`: OK.
- Ultimo CI de `master`: [verde en Linux 3.12/3.13 y Windows
  3.12](https://github.com/Ton-Llop/Jaull/actions/runs/33631688663). La branch
  `execution-plan` aun no tiene run propio.
- Bien cubierto: cuatro fit modes, limites, KV/context/concurrency, unified memory,
  diagnostics, shortlist 6/8/24 GiB, CPU-only, determinismo, diversity, ranking, caches,
  persistencia y lifecycle TUI.
- Falta cubrir: fallback byte con RAM restrictiva, status cuando falta KV, matching por
  contexto/revision/runtime/metodologia, tied/untied decomposition y ejecucion real de
  workers Transformers.
- `transformers_worker.py` y `transformers_benchmark_worker.py` aparecen con 0%: se prueban
  comandos y parsers, no el worker aislado real.
- Los tests screenshot/TUI son los mas caros, pero protegen packaging, navegacion, lifecycle
  y `DuplicateIds`; no deberian eliminarse indiscriminadamente.
- Codecov se sube, pero no existe threshold local de cobertura. El 84% no evita los fallos
  semanticos encontrados.

## J. Preparacion para Hardware-aware shortlist

**PREPARADO**, con una precision importante: la v1 **ya esta implementada e integrada**.
Incluye offload en 8 GiB, amplia escala en 24 GiB, funciona CPU-only, limita imposibles,
diversifica familias y mantiene `MAX_DEEP_INSPECTION`:
[`test_discovery_search.py`](../tests/test_discovery_search.py#L444) y
[`test_workflow_orchestrator.py`](../tests/test_workflow_orchestrator.py#L231).

Antes de considerarla fiable para los experimentos faltan exclusivamente:

- corregir los dos fallos de soundness del estimator;
- endurecer el matching de evidencia;
- integrar correctamente la descomposicion de pesos;
- congelar un trace experimental de `40 inputs -> coarse hints -> 12 elegidos`;
- medir recall con candidatos reales, especialmente nombres sin parameter hint y MoE.

## K. Prioridades

| Prioridad | Tarea | Motivo | Bloquea siguiente milestone |
|---|---|---|---|
| P0 | Tratar componentes de memoria desconocidos como fit no confirmado | Evita falsos positivos y ladders incorrectas | Si |
| P0 | Definir matching canonico de evidencia local | Evita que datos no comparables cambien ranking/calibracion | Si para experimentos |
| P1 | Corregir fallback de offload por bytes | Puede producir falsos `TOO_LARGE` | Si para garantizar el contrato HFA |
| P1 | Integrar decomposition en `WeightEstimate` | Elimina coste por block conceptualmente inflado | Si para calibracion HFA |
| P1 | Separar planning capacity de run-now availability | Hace reproducibles recomendaciones por hardware | No para la v1 |
| P1 | Revisar reserve/margin en rutas CPU explicitas | El reserve de GPU puede contaminar el total CPU | No |
| P2 | Capturar snapshots de prediction input centralmente | Mejora replayability | No |
| P2 | Reducir Advisor/workflow/legacy y actualizar docs | Mantenibilidad y rigor documental | No |

## L. Que haria ahora

### Siguiente milestone

`Estimator fit soundness`.

### Objetivo

Eliminar falsos positivos con componentes desconocidos y falsos negativos del fallback por
bytes.

### Por que ahora

Son errores de correctitud pequenos y aislables. Ajustar shortlist, calibrar HFA o alquilar
GPUs antes de corregirlos produciria evidencia contaminada.

### Que entra

- contrato explicito para totales incompletos/lower bounds;
- impedir que la ladder acepte un fit no confirmado;
- corregir el intervalo de viabilidad RAM/VRAM del fallback por bytes;
- regression tests exactos y property tests de fronteras;
- demostrar que los escenarios completos block-aware conservan sus resultados.

### Que NO entra

- weight decomposition;
- cambios de overhead, margin o reserve;
- mapping a llama.cpp;
- evidence matching;
- ranking o shortlist;
- calibracion.

### Criterio de terminado

Los dos probes fallan antes y pasan despues, todos los escenarios HFA completos mantienen sus
resultados, y suite, ruff, mypy y arquitectura permanecen verdes.
