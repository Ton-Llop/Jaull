# Auditoria tecnica del estado actual de Jaull

Auditoria original: 2026-09-03, branch `execution-plan`, commit
`0910967e0768548b5fd84c83427a1754daf5816c`.

**Revisiones: R (2026-09-16, `5bf954b`), S (2026-09-17, `059dac3`), T (2026-09-20) y U (2026-09-24, working tree sobre `1c341c4`).**

El cuerpo original (secciones A-L) se conserva tal como se escribio el 03/09: es un
registro fechado, no un documento vivo. Sus cifras, prioridades y tareas pendientes
no describen por si solas el estado actual. Las revisiones R, S, T y U anteriores
en este archivo documentan los cambios posteriores; para saber si un hallazgo sigue
abierto hay que leer la revision mas reciente, aunque la seccion original no lleve
una marca **CERRADO**.

La auditoria original describe el commit `0910967`; cada revision tiene su propia
fecha y base. U se preparo sobre cambios del working tree posteriores a `1c341c4`:
sus medidas R9 y tests no forman parte de ese commit base.

---

## R. Revision del 2026-09-16

### R.1 Gates

| Gate | 03/09 | 16/09 |
|---|---|---|
| ruff | clean | clean |
| mypy | clean | clean, 244 ficheros |
| pytest | 1381 | **1614 passed** |
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
4. **Tres modelos sub-1.5B ocupaban el Top 5 por delante de un 4B.** Cadena verificada:
   `capability` colapsaba 0.5B y 4B en el mismo bucket `adequate` (la separacion en la curva
   de tamano, 0.158, era menor que el ancho de bucket, 0.17); al empatar decidia
   `execution_fitness`, que es funcion pura del estado de memoria, y ganaba el mas pequeno.
   Por debajo, el 55 % de `capability_score` no media capacidad: a Qwen3-4B le falta la linea
   `language:` en el model card y eso le costaba mas (-0.123) de lo que le daban 3.4B de
   parametros extra (+0.106). El fix de `2f85c45` es real pero ortogonal: baja los modelos con
   estado `UNKNOWN`, no los pequenos con estado `comfortable`.

   **CERRADO (16/09).** `capability_score` era `0.45*size + 0.30*metadata + 0.15*artifact +
   0.10*activity`, y `metadata`, `activity` y la licencia ya tenian peso propio en
   `policies.BASE_WEIGHTS` (`metadata_quality` 0.07, `popularity` 0.03, `license` 0.08). Eran
   evidencia contada dos veces, y en la ruta con hardware contaba mas que el propio numero de
   parametros: `engine_v2` ordena por el *bucket*, asi que una vez dos candidatos caen del
   mismo lado de la curva de tamano, metadata y descargas eran los unicos terminos que podian
   moverlo. Los dos terminos duplicados se han quitado y los dos que quedan se han
   renormalizado a `0.75*size + 0.25*artifact`; los umbrales de `_capability_level` no se han
   tocado. `_artifact_signal` se queda porque no esta cubierto por ningun otro eje y medido
   sobre candidatos reales es casi constante (0.143-0.150), asi que no distorsiona.

   Resultado en el replay del RTX 4060: Qwen3-4B pasa de #5 a **#2**, por detras solo del
   Llama-3.1-8B, y los tres sub-1.5B caen a `capability=weak` en las posiciones #3-#5. Las
   otras prioridades siguen comportandose como deben: con `memory` el orden se invierte y el
   0.5B vuelve a ser #1.

   Efecto lateral esperado: los niveles absolutos bajan un escalon (un 1.5B pasa de `adequate`
   a `weak`, un 4B de `strong` a `adequate`) porque el suelo constante de ~0.55 que aportaban
   metadata y actividad ha desaparecido. La escala nueva es aproximadamente <2.5B `weak`,
   2.5-7B `adequate`, >=7B `strong`. El snapshot del report se movio en cuatro numeros y en
   nada mas: `capability` 0.743 -> 0.700 y `total` 0.619 -> 0.614 para un unico candidato.
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

---

## S. Revision del 2026-09-17

Dos bloques que la revision R no recoge, porque salieron *despues* de escribirla, de
ejecutar Jaull en la RTX 4060 y leer el report que produce.

### S.1 Gates

| Gate | 16/09 | 17/09 |
|---|---|---|
| ruff | clean | clean |
| mypy | clean, 244 ficheros | clean, **245** ficheros |
| pytest | 1614 | **1674 passed** |
| cobertura | 85 % (17 271 stmts) | 85 % (17 392 stmts, 2 561 sin cubrir) |

Commits: `bb1a7a9`, `92c541c`, `89a30c7`, `4a054d6`, `7c68cb0`, `78ecf2e`, `059dac3`.

### S.2 La configuracion estimada no era la que se ejecutaba

**El hallazgo mas grave de los dos**, porque no era un problema de presentacion: rompia la
cadena de evidencia sobre la que se apoya el TFG.

`runtime/transformers.py` convertia `WeightPrecision.INT4` **y** `INT8` en
`torch_dtype="torch.int8"`. Eso no cuantiza nada: `from_pretrained(torch_dtype=torch.int8)`
pide un dtype de tensor que el cargador no puede usar para pesos. Y no era solo el snippet
que se mostraba — `transformers_worker.py` y `transformers_benchmark_worker.py` pasaban ese
mismo dtype al cargador sin ninguna `quantization_config`, asi que **Run y Benchmark
ejecutaban pesos sin cuantizar y los reportaban como la configuracion predita**. Un
`ExperimentRecord` a int4 habria emparejado una prediccion de 0,5 bytes/parametro con una
ejecucion que no lo era, y la capa de comparacion habria calculado un porcentaje de error
entre dos cosas distintas.

Comprobado que **ningun bundle de `validation/` menciona int4 ni int8**: todo lo medido hasta
hoy es llama.cpp GGUF, asi que ninguna medicion existente esta contaminada. El riesgo era
prospectivo — el `#1` del report de la 4060 es exactamente esa configuracion.

**CERRADO (17/09).** `runtime/transformers_quantization.py` concentra la traduccion
precision <-> mecanismo. Ahora la precision viaja en un solo flag: `torch_dtype` para los
floats, `quantization` para los cuantizados, que el worker convierte en
`BitsAndBytesConfig`. Sin `bitsandbytes` **falla con un motivo explicito** en vez de degradar
en silencio. No se anade la dependencia: es CUDA-centrica, incomoda en Windows, y un plan
Transformers cuantizado ya se reporta como artefacto teorico.

Verificado en esta maquina (RTX 2060, sin `bitsandbytes` instalado), contra los workers
reales y sin descargar ningun modelo:

```
$ python -m jaull.runtime.transformers_worker --quantization 4bit ...
{"error": "This configuration needs the 'bitsandbytes' package, which Jaull does not
 install. ... Jaull will not run an unquantized model in place of a quantized estimate:
 the measurement would not describe the configuration that was predicted.",
 "success": false}                                                            exit=1
```

Identico por el worker de benchmark. Control negativo: con `--torch-dtype torch.float16` el
guard no interviene y el fallo llega mas tarde, al cargar. Falta la verificacion de la
cadena entera (`jaull run` desde la TUI sobre la recomendacion `#1`), que necesita el modelo
descargado.

Dos defectos colaterales del mismo cambio, encontrados en revision cruzada:

- El helper importaba `WeightPrecision`, que arrastra Pydantic. Los workers corren en el
  Python de PyTorch del usuario, que no es el venv de Jaull: el worker moria con
  `ModuleNotFoundError: pydantic` **antes** de poder emitir su error estructurado. El modulo
  ya no importa nada del dominio; indexa por el valor de la precision.
- Los workers solo pasaban `device_map` cuando era `"auto"`; con `"cuda"` cargaban y despues
  hacian `.to("cuda")`. Sobre un modelo bnb de 4 bits eso no es estilo: **bitsandbytes
  rechaza `.to()`**. Un plan cuantizado se coloca ahora al cargar, como dice el snippet.

Y `recommendation/local_evidence.py` comparaba solo `torch_dtype`, asi que un record int4 y
uno int8 respondian ambos `None` y se consideraban la misma configuracion — la misma familia
que D.2.

### S.3 El report no describia la ejecucion que habia hecho

Cuatro contradicciones, las cuatro visibles en un unico report de la 4060:

1. **`evaluated_candidates` publicaba los defaults del modelo.** Todos los scores a `0.0`,
   `requirement_penalty` a `1.0` y `unmet_requirements` vacio, mientras la recomendacion
   construida desde *el mismo candidato* reportaba `0.51` y dos requisitos incumplidos. El
   orquestador guardaba la lista previa al enriquecimiento, que ocurre dentro de `recommend`.
2. **El score contradecia la posicion en pantalla.** Rank 1 con 44/100, rank 4 con 65/100. El
   orden lo da la tupla lexicografica de `engine_v2`; el numero impreso al lado venia del
   compuesto ponderado, que con hardware no ordena nada.
3. **La confianza baja se atribuia siempre a metadata ausente.** Como `overhead.py` emite
   `LOW` incondicionalmente, ese aviso salia en el 100 % de las recomendaciones, con ficha
   completa o sin ella.
4. **"No precision fits" se decia con peldanos en `offloading_required`**, y `unknown` se
   presentaba como si se hubiera demostrado que no cabe.

**CERRADO (17/09).**

- El orquestador publica el resultado de `enrich_candidate_features` — *la misma funcion que
  llama `recommend`*, sobre las mismas entradas. Verificado digito a digito: `capability`
  `0.7003645535942264` en los dos bloques. Los candidatos fallidos nunca pasan por el
  enriquecimiento, asi que salen `null`: "nunca se evaluo" no es "saco cero en todo".
- `ranking.criteria` publica los ejes reales, y `score_role` distingue `diagnostic` de
  `ordering` — sin hardware el compuesto **si** ordena, y etiquetarlo diagnostico habria sido
  la misma mentira en direccion contraria.
- `_confidence_warning` nombra el componente que topa la confianza.
  `_exhausted_ladder_message` distingue los tres desenlaces, y describe **la configuracion
  devuelta**, no el conjunto: `best_effort` guarda el primer peldano, asi que afirmar
  viabilidad de offload para algo marcado INSUFFICIENT era otra contradiccion. La afirmacion
  global "ninguna cabe" solo se emite si toda la escalera salio INSUFFICIENT.

**Schema 3**: el cambio a `null` no es compatible hacia atras para un lector estricto. Nada
dentro de Jaull lee el payload de vuelta (`reporting/writer.py` solo lo escribe), asi que la
ruptura queda confinada a lo que consuma el fichero exportado.

Tres iteraciones de revision cruzada corrigieron, sobre el propio texto publicado: que
`ranking_criteria` omitia la particion de viabilidad que corre *despues* del sort (en
`quality` la explicacion contradecia la lista), que `viability` publicaba el estado crudo y
hacia que `offloading_required` pareciera ir por encima de `comfortable` cuando el eje solo
compara dos estados, y que `hard_constraints: none` chocaba con la seccion "Unmet
requirements" de la misma entrada, que es otra puerta.

### S.4 Lo que sigue abierto

Ademas de R.3:

- **Dos definiciones de "requisito duro".** `_hard_constraints` (que ordena) solo rechaza
  licencias `COMMERCIAL_RESTRICTED`; el compuesto penaliza tambien las `UNKNOWN`. Y
  `requirements_gate.py:118` marca el idioma como `required=False` con penalizacion 0,15, que
  aun asi arrastra `hard_penalty` por debajo de 1,0 y fuerza `BEST_EFFORT` por la regla 1 de
  `choose_tier`. Un check blando produciendo una degradacion dura. **Es decision de producto,
  no un bug**, y por eso sigue sin tocarse.
- `quantization_quality: n/a` aparece en toda entrada de Transformers, donde el eje no
  significa nada. Ruido, no falsedad.

### S.5 El hueco real de la campana experimental

**Ninguna medicion registrada pasa por el observation contract.**

```
b001-r4-full-offload    runtime_allocation=None   peak_vram=None
b001-r4-launch-policy   runtime_allocation=None   peak_vram=None

ultima medicion fisica : 14/09
ultimo commit          : 16/09
```

Los bundles son anteriores al contrato. Todo el trabajo del 15 al 17 mejora lo que Jaull
*dice*; la campana experimental, que es donde vive la tesis, no se ha movido.

La consecuencia concreta: la frase que persigue el TFG — *"Jaull lo ha predicho, lo ha
ejecutado, lo ha medido y sabe en que se ha equivocado, termino a termino"* — sigue sin
respaldo experimental, **aunque el codigo que la sostiene ya este escrito y probado**. Una
re-ejecucion de B001 con el codigo actual seria el primer `ExperimentRecord` con
`runtime_allocation`, y por tanto el primer error por componente: pesos DIRECT, KV DIRECT,
overhead PROXY.

Es lo unico de esta lista que no puede hacerse sin la GPU, y lo que mas valor tiene.

**PARCIALMENTE CERRADO (17/09).** Ejecutado: ver
[B001-R7](../docs/qwen2.5-tests/b001-r7-observation-contract.md). La re-ejecucion encontro
primero que **el contrato no llegaba a dispararse nunca**: `llama_cpp_runner.py` pasaba
`--verbose` solo si un flag de runtime lo pedia, y nada lo pone. Sin el, este build no
escribe *ninguna* linea a stderr, asi que el parser no tenia lineas de buffer y
`_observed_backend` tampoco encontraba backend. Todos los tests del contrato le daban al
parser un log de agosto ya grabado; ninguno comprobaba que una ejecucion viva produjera uno.

Corregido (se pide el log siempre) y re-ejecutado. **Primera medicion de dispositivo de la
historia del proyecto en esta maquina:**

| Buffer | Medido |
|---|---:|
| `CUDA0 model` | 3741.47 MiB |
| `CUDA0 KV` | 200.00 MiB |
| `CUDA0 compute` | 183.44 MiB |
| **total dispositivo** | **4124.91 MiB** |

`source = runtime_reported_allocation`, `driver_confirmed = false` — NVML no atribuye nada
bajo WDDM, asi que el reporte del propio runtime es la unica observacion disponible, que es
para lo que se construyo.

**Lo que sigue abierto es la otra mitad.** La comparacion sigue
`methodologically_unavailable`: la prediccion esta dimensionada en 20 bloques de transformer
y la ejecucion uso 26 unidades de `--n-gpu-layers`. Son colocaciones distintas, asi que la
diferencia entre 3372.7 MiB predichos y 3741.47 MiB medidos es sobre todo colocacion, no
error del modelo de memoria. **Ningun numero de esta ejecucion debe usarse para calibrar.**

El estado paso de *"no hay medicion"* a *"hay medicion y no hay prediccion comparable"*. Lo
que bloquea el resto es el mapeo bloque <-> unidad de lanzamiento, no el lado de la medida.

---

## T. Estado posterior: consolidacion 2026-09-20

La seleccion de configuracion ya no conserva automaticamente la primera opcion
insuficiente cuando otra precision o cuantizacion de la misma escalera tiene un
placement `OFFLOADING_REQUIRED` conocido. El fallback prioriza, en este orden,
offload conocido, resultado `UNKNOWN` e insuficiente. La busqueda de fits
residentes no cambia.

La readiness de experimentos Transformers incluye ahora el requisito real de
bitsandbytes para planes INT4/INT8. Las variantes alternativas no heredan una
readiness que puede corresponder a otros flags; se vuelven a evaluar al preparar
el plan concreto.

Cuando se exige uso comercial, una licencia confirmada se ordena antes que una
licencia desconocida. La desconocida sigue visible como alternativa y no se
trata como una incompatibilidad legal confirmada. El score compuesto conserva
su significado diagnostico y no se modifica para forzar el orden.

La campaña [B001-R8](qwen2.5-tests/b001-r8-2060-context-matrix.md) ejecuto el
artefacto local Q4_K_M en la RTX 2060 con contextos 512, 2048 y 4096, mas una
repeticion a 4096. Las cuatro ejecuciones arrancaron. La comparacion numerica sigue metodologicamente
indisponible porque HFA expresa bloques de transformer y llama.cpp expresa
unidades `--n-gpu-layers`; los resultados no se han usado para calibrar
overhead, reserve, margen ni politica de lanzamiento.

---

## U. Revision del 2026-09-24

### U.1 Gates

| Gate | 17/09 | 24/09 |
|---|---|---|
| ruff | clean | clean |
| mypy | clean, 245 | clean, 245 |
| pytest | 1674 | **1705 passed** |
| cobertura | 85 % | **86 %** (17 544 stmts, 2 416 sin cubrir) |

### U.2 Lo que ha empeorado

Los ficheros que E.4 senalo por tamano han seguido creciendo, en la direccion
contraria a la recomendada:

| Fichero | 03/09 | 16/09 | 24/09 |
|---|---:|---:|---:|
| `advisor/service.py` | 1305 | 1483 | **1570** |
| `recommendation/engine_v2.py` | — | 1084 | **1356** |
| `tui/screens/recommendation_results.py` | — | 1262 | **1290** |

No bloquea nada y no es lo que se evalua en el TFG, pero conviene no seguir
anadiendo ahi sin motivo.

Sin cambios: **E.5** (`prediction_input` opcional en cinco sitios), la rama
muerta de `_sum_components`, la politica de memoria confirmada duplicada entre
`engine_v2` y `ranker`, y `BEST_MATCH` inalcanzable porque `overhead.py:39`
emite `LOW` incondicionalmente.

Tambien sigue abierto el **check blando de idioma que degrada duro**:
`requirements_gate.py:118` marca `language:` como `required=False` con
penalizacion 0,15, que aun asi baja `hard_penalty` de 1,0 y fuerza
`BEST_EFFORT` por la regla 1 de `choose_tier`. El eje `requirement_confirmation`
resolvio la mitad de licencias de este problema; esta mitad no.

### U.3 B001-R8 media una cosa que no contaba

La matriz de contextos registro que todos los niveles de offload arrancaron,
pero no registro a que velocidad. Leida asi, invitaba a una conclusion falsa:
HFA predice 15–20 bloques mientras llama.cpp arranca con 29/29, asi que el hueco
parecia una perdida de rendimiento de 3–4x.

**No lo es**, porque el numero de bloques de HFA no es lo que Jaull lanza. La
politica de lanzamiento emite su propio valor en sus propias unidades, y ese
valor no se habia medido nunca.

Medido en [B001-R9](qwen2.5-tests/b001-r9-launch-policy-throughput.md), mismo
artefacto, maquina, build e invocacion que R8, con el nivel tomado de la ruta de
produccion (`estimate_model` -> `runtime_recommendation`), no elegido a mano:

| Contexto | HFA | Politica | t/s politica | t/s full | Ratio |
|---:|---:|---:|---:|---:|---:|
| 512 | 20/28 | `ngl 26` | 27,1 | 39,9 | **1,47x** |
| 2048 | 20/28 | `ngl 26` | 27,9 | 39,4 | **1,41x** |
| 4096 | 19/28 | `ngl 25` | 24,8 | 52,0 | **2,10x** |

En estas ejecuciones, el cociente entre full offload y la politica fue
**1,4x–2,1x**, no el 3–4x que se inferia al confundir bloques HFA con unidades
de lanzamiento. Solo hay una ejecucion por celda, asi que no es una estimacion
estable del coste de la politica. A offload completo quedaron 196–366 MiB libres
de 6144; esa cifra ya descuenta la ocupacion basal del momento. Una mayor carga
del escritorio podria agotar ese margen, pero comparar directamente los MiB
libres con la ocupacion basal de R8 (930–1359 MiB) no demuestra que vaya a ocurrir.

**Lo que no queda establecido** es si 512 MiB de reserve mas 256 de headroom es
la cantidad correcta. Nunca se ha calibrado; es una suposicion documentada. R9
tampoco la calibra, y a proposito: tres medidas de una sola repeticion, en una
maquina, un modelo y un build no son base para mover una constante de seguridad.
El propio dato lo recuerda — 52,0 t/s a contexto 4096 frente a 39,4 y 39,9 a
contextos menores es ruido, no tendencia.

R8 queda enlazado a R9 para que no se lea suelto.

### U.4 Cobertura del benchmark worker

`transformers_benchmark_worker.py` estaba a **0 %** (164/164 sin cubrir) y
mientras tanto habia recibido la rama de cuantizacion: `build_quantization_config`,
la decision `place_at_load` y `_load_device`. El test del runner solo comprobaba
que `--quantization` llegaba a la linea de comandos; lo que recibia el *cargador*
estaba sin verificar, y esa es justo la mitad donde un plan cuantizado se
convierte en silencio en una medicion sin cuantizar.

`tests/test_transformers_benchmark_worker.py` lo cubre con torch y transformers
falseados (el worker corre en el entorno PyTorch del usuario, asi que el test no
puede necesitar uno): colocacion al cargar frente a `.to()` posterior, `auto`
intacto, el alias HIP->CUDA, la negativa sin bitsandbytes **antes** de cargar
nada, el payload estructurado del fallo, y la cadena `methodology` sobre la que
`local_evidence` empareja records.

Resultado: **0 % -> 92 %**, y el total del repositorio 85 % -> 86 %.

Queda a 61 % `transformers_worker.py`, que es el camino de `run`.

### U.5 El observation contract sigue con cero records

B001-R7 arreglo el runner para que pida el log siempre, y la fontaneria esta
probada. Pero **ningun fichero de `validation/` lleva todavia un
`runtime_allocation` dentro de un `ExperimentRecord`**: los bundles son
anteriores al contrato, y R8/R9 no pasan por `ExperimentRequest` — generan
`report.json` y `prediction.json` propios.

Lo que R8 hace bien y conviene no perder: sus `vram_error_pct_*` salen `None`
con una nota en el propio record explicando por que no se calculan. El dato dice
que no sabe en lugar de inventar un porcentaje.

### U.6 Lo que haria ahora

1. Una campana que pase por `ExperimentRequest` para que exista al menos un
   record con `runtime_allocation`. Es lo unico que convierte la fontaneria en
   evidencia.
2. El mapeo bloque <-> unidad de lanzamiento, que es lo que bloquea el error por
   componente. B001-R6 ya lo deriva para `689e227db` + `qwen2` denso dentro de la
   politica; la capa de comparacion no lo consume.
3. La decision de producto pendiente sobre el idioma blando que degrada duro.

Lo que **no** haria todavia: calibrar reserve y headroom. R9 mide lo que cuestan
pero no da base para moverlos.

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
