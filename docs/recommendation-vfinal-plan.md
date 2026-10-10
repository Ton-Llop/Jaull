# Recommendation vFinal: plan de ejecución

Propuesta para revisión de Ton. Base inspeccionada: `master`, `7aac979`,
2026-10-07. Este documento no implementa ni activa ninguna política.
Ton autoriza ejecutar este plan directamente en `master`, con cambios sin stagear.

Objetivo: que Search produzca hasta cinco recomendaciones justificadas según
requisitos, viabilidad y evidencia. No demostrar que son los mejores modelos
existentes. Cerrar esta fase y pasar a validación del TFG y al caso Biosfer.

## 1. Alcance y decisiones

- Search consulta metadata y evidencia existente. Nunca descarga pesos, arranca
  Docker ni mide modelos. La evaluación posterior requiere una acción explícita.
- Se conservan HFA, fórmulas de memoria, colocación, restricciones de licencia,
  idioma, tarea y elegibilidad. Runtime instalado es preparación operativa,
  no calidad del modelo ni requisito para recomendarlo.
- Se cambia la política de ordenación de Quality, Fastest y Balanced, primero
  en shadow mode. La prioridad Memory y el camino legacy sin hardware se conservan.
- No hay score global, bonus por familia/generación, ni conversión de tok/s a calidad.
- Sin evidencia aplicable se mantiene un fallback determinista, identificado
  como heurístico. Ausencia no equivale a cero ni a peor calidad.
- El prior de escala existente puede seguir siendo fallback; no se presenta
  como medición. No se inventa un predictor de velocidad en esta fase.
- Registros y búsquedas anteriores son inmutables. Recompare produce un nuevo
  resultado, vinculado al anterior, con versión de política y evidencia utilizada.
- No se necesita SQLite nuevo, catálogo automático, servicio remoto ni scheduler.

Esta propuesta cambia el borrador de `politica-calidad-velocidad-equilibrio.md`:
Balanced pasa a Pareto dentro de grupos comparables; un mínimo de velocidad no
se convierte aquí en una nueva puerta de elegibilidad. Las predicciones existentes
pueden explicarse como tales, pero no se mezclan numéricamente con mediciones.
Actualizar ese borrador en el paso 0; no mantener dos decisiones contradictorias.

## 2. Qué existe realmente

| Área | Punto de extensión comprobado |
|---|---|
| Discovery | `discovery/query_builder.py`, `search_client.py`, `candidate_filter.py`, `workflow/orchestrator.py` |
| Pool | El orchestrator filtra antes de `MAX_UNIQUE_CANDIDATES` y limita después la inspección profunda |
| Ranking | `recommendation/engine_v2.py`: `PlanRankingContext`, `assess_plan`, `_ranking_key`, `ranking_criteria` |
| Top 5 | `application/recommendation/service.py` aplica `diversify_ranked_plans` después de ordenar planes |
| Evidencia publicada | `domain/capability_evidence.py`, `recommendation/capability_catalog.py`; referencias de base no son evidencia del GGUF |
| Calidad local | `evaluation/quality_records.py`, `quality_storage.py`, `quality_comparison.py` |
| Velocidad | `recommendation/local_evidence.py`, `evaluation/benchmark_comparison.py`, `benchmarks/` |
| Evaluación opt-in | `advisor/quality.py`, `advisor/quality_candidates.py`, `runtime/quality_eval_runner.py`, CLI/TUI existentes |
| Serialización | `reporting/recommendation.py`, schema actual 3; snapshots de JSON y Markdown |

El contexto de ranking ya recibe los registros de calidad local como referencias
diagnósticas. `QualityAssessment` selecciona un perfil objetivo por tarea/idioma
y distingue ausencia de referencias. Ninguna suite generativa está habilitada
todavía: el lector sigue validando únicamente scoring de continuaciones.
`PlanAssessment.performance_evidence` expresa respaldo, no rapidez. No reutilizar
ese concepto como nivel de capacidad. `CapabilitySignal` conserva el prior y
sus referencias; no construir un segundo motor de capacidades.

El productor tiene únicamente `smoke` y `hellaswag100`. El lector admite `full`
pero exige el split completo; valida el protocolo de loglikelihood y las métricas
`acc`/`acc_norm`. El comparador también está especializado en esas métricas.
Por tanto, añadir tareas generativas afecta productor, contrato, comparación y UI.

`run_candidates` ya ejecuta de uno a tres candidatos secuencialmente y persiste
cada éxito. No volver a construir esa cola ni otro botón de evaluación.

## 3. Política mínima, antes de escribir el ranker

### Calidad y evidencia

`QualityAssessment` será una proyección de las evidencias existentes por plan y
tarea: métricas con unidad/dirección, IDs de evidencia, alcance del sujeto,
protocolo/cohorte, cobertura, aplicabilidad, limitaciones y motivo del fallback.
Conservar referencias a los registros completos; evitar copiar todo el historial
en cada candidato o inventar otra base de datos.

No asignar `STRONG`/`HIGH` por tener una evaluación completa. Completitud,
confianza y resultado son dimensiones distintas. Inicialmente mostrar el valor
del benchmark, su alcance y el estado `measured`, `published` o `estimated`.
Un ejemplo de redacción: «HumanEval: N tareas evaluadas; pass@1 X; protocolo Y».
N/N muestras procesadas no significa N/N problemas resueltos.

Una evidencia solo puede ordenar si cumple todas estas condiciones:

1. Identifica al sujeto exacto al que se aplica. Para una medición local, SHA256
   del artefacto y configuración de evaluación conocida.
2. Es pertinente para la tarea y el idioma cubierto. HellaSwag no ordena código
   y una evaluación en inglés no demuestra calidad en español o catalán.
3. Está completada y cumple el contrato de reutilización. Smokes, fallos,
   parciales y `limited` permanecen diagnósticos.
4. Tiene protocolo y métrica comparables con los otros miembros del grupo.
5. Sus limitaciones no contradicen la afirmación que hará la recomendación.

La aplicabilidad distingue `controlled_artifact` de `current_configuration`.
Quality puede ordenar los artefactos exactos bajo su protocolo común de evaluación,
con ese alcance visible, aunque el contexto propuesto para el usuario sea distinto.
Para afirmar calidad medida de la configuración actual se comprueban además sus
condiciones. Balanced v1 requiere correspondencia entre las condiciones de calidad
y las del benchmark de velocidad. No inferir portabilidad entre GPUs o colocaciones.

Las fuentes coexisten; local no borra publicado. El catálogo actual, con revisión
y precisión desconocidas, sigue siendo referencia. Una futura publicación solo
podrá ordenar si pasa atribución y comparabilidad y describe la representación
que se compara. Declarar un modelo base no transfiere su resultado a un GGUF.
No rellenar campos desconocidos para conseguir cobertura.

### Orden determinista con datos incompletos

Para v1 usar una política conservadora que pueda probarse sin comparadores
no transitivos:

1. Obtener el orden base actual, preservando sus grupos de viabilidad,
   restricciones confirmadas y adecuación a la tarea. Evidencia nueva no permite
   cruzar esas restricciones. Capturar este orden antes de la diversidad final.
2. Elegir una suite principal por tarea, declarada antes de ver resultados.
   Elegir un protocolo versionado por grupo; no buscar el que dé el mejor score.
   Para calidad, seleccionar la campaña/ejecución antes de ver scores; v1 usa una
   ejecución por artefacto con seed fijada. Repeticiones requieren un perfil que
   declare seeds/cantidad, completitud y agregación de la misma métrica, conservando
   resultados y dispersión. Variación entre ejecuciones no es conflicto automático.
   Para velocidad, mantener la selección por última ejecución válida del protocolo
   fijado, con ID como desempate. Fuentes incompatibles conservan sus bloqueos.
3. Formar grupos disjuntos de planes con evidencia aplicable y comparable.
   Reutilizar las puertas existentes cuando correspondan al perfil. Las suites
   generativas necesitan un contrato propio verificado; permiten tokenizers
   diferentes y registran su identidad. Raw exige texto común; chat exige mensajes
   comunes y una regla auditada para aplicar cada template oficial, con su hash
   y texto final. La identidad de artefacto
   debe coincidir con su propio plan, no ser igual entre modelos diferentes.
4. Reordenar solo las posiciones que ya ocupan los miembros de cada grupo.
   Los planes sin datos comparables mantienen su posición de fallback en esta
   etapa. Un registro aislado no promueve a un modelo por estar medido.
5. Aplicar la diversidad y deduplicación existentes. Guardar por separado los
   cambios de orden y las exclusiones por diversidad; no atribuir todo al benchmark.

Es una elección de producto deliberada: con cobertura parcial no podemos afirmar
un ganador global. Un modelo desconocido puede quedar entre dos evaluados por su
posición base. La interfaz debe explicar esa limitación. El caso sin evidencia
debe conservar exactamente el orden actual; no introducir el bonus «measured».

- **Quality:** ordenar dentro de cada grupo por la métrica primaria predefinida
  de la tarea. Empates conservan el orden base. Resto de métricas visibles;
  no promediarlas ni presentar el resultado como inteligencia general.
- **Fastest:** comparar `tg128` de ejecuciones aplicables a la máquina y al plan;
  mostrar `pp512` por separado. No usar el máximo entre longitudes de generación:
  `_measured_generation_tps` hoy toma un máximo y debe revisarse para esta política.
  No estimar TTFT a partir de pp512 ni atribuir al benchmark un `--ctx-size` que
  llama-bench no ejecutó. Sin medición comparable, fallback explícito.
- **Balanced:** dentro de grupos con calidad y velocidad comparables, calcular
  frentes Pareto. A domina B si no es peor en ninguna de las dos métricas y es
  estrictamente mejor en alguna. Ordenar frentes; dentro del frente, orden base.
  No añadir epsilon, tolerancias ni pesos intuitivos. Con un eje desconocido,
  no inferir dominancia. Se mantiene el fallback y se explica.

Dentro de cada frente Pareto, el desempate sigue siendo el orden base. La
explicación debe atribuirlo al baseline: Pareto no decide por sí solo el trade-off
entre un modelo más rápido y otro de mayor calidad que están en el mismo frente.

Pareto describe métricas del protocolo fijado, no una diferencia estadísticamente
demostrada de capacidad general. Una diferencia de 25 frente a 23 tok/s no
demuestra por sí sola una ventaja robusta: mostrar repeticiones/dispersión.
En el ejemplo B = calidad media/70 tok/s y C = baja/75 tok/s, B no domina C:
C sigue siendo más rápido. Eliminarlo por «apenas gana velocidad» exigiría otra
preferencia o tolerancia explícita. Las etiquetas HIGH/MEDIUM tampoco permiten
tratar dos resultados de calidad diferentes como si fueran exactamente iguales.

La compatibilidad de velocidad debe comprobar hardware, metodología, build,
cargas pp/tg y flags efectivos. Dos planes pueden tener distinto offload y
compararse como planes completos si cada observación corresponde a su plan.
No confundir esto con la puerta más estricta de colocación de calidad existente.
Un resultado histórico sin datos necesarios se muestra, pero no se fuerza al ranking.

## 4. Suites: límite cerrado

No añadir IFEval, HumanEval, MBPP y GSM8K a la vez.

| Uso | Producción de evidencia en esta fase | Límite de la afirmación |
|---|---|---|
| Infraestructura | HellaSwag smoke y 100 existentes | Diagnóstico; nunca ranking |
| General/instrucciones | IFEval, split completo, suite versionada | Seguimiento de instrucciones bajo ese protocolo |
| Código | HumanEval, una generación por problema, pass@1, suite versionada | Resolución de esos problemas de código |
| Otros/idiomas no cubiertos | Evidencia publicada atribuible y fallback | Sin afirmar medición específica |

No sumar IFEval y HellaSwag ni llamarlo evaluación general completa. Reasoning,
writing, translation, summarization y documentos/Biosfer no añaden nuevas suites
en este cierre. El soporte de búsqueda para esas tareas sigue existiendo.

**Esto es un bloque de ingeniería propio.** La configuración de
[IFEval en el pin actual](https://github.com/EleutherAI/lm-evaluation-harness/blob/ad8737ae7fad24cf64e50fc7fc31397bff586b9e/lm_eval/tasks/ifeval/ifeval.yaml)
usa generación y varias métricas; fijar `prompt_level_strict_acc` como primaria.
El piloto actual solo valida scoring de continuaciones. Auditar chat templates,
tokenización, EOS/stop, contexto, truncamiento y presupuesto realmente enviados
al servidor. Flags aceptados por la CLI no bastan. Mantener response cache apagada.

[HumanEval en upstream](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/humaneval/humaneval.yaml)
está marcado como ejecución de código no confiable. Esta referencia es informativa,
no un pin: verificar soporte y fijar commit/config/dataset antes de implementar.
Generar con llama-server y ejecutar los tests de código en un proceso/contenedor
separado, sin red, sin Docker socket, sin credenciales ni montajes del proyecto,
sin privilegios y con límites de tiempo/memoria/procesos. El contenedor evaluador
que habla con el servidor no debe ejecutar libremente el código generado.
Si no puede verificarse ese aislamiento, bloquear la evaluación de código;
no activar flags de código inseguro como único control.

Usar prompt/formato fijado por suite; conservar template y revisión cuando se
aplique uno. Protocolos de prompt incompatibles permanecen NOT COMPARABLE.
Cambiar de raw a chat o de estrategia de extracción crea otra versión de suite.
Registrar respuestas truncadas y errores, sin excluirlos silenciosamente del denominador.

Smokes nuevos prueban el transporte y la extracción, no capacidad. Solo el split
completo será reutilizable en v1. Un subset «representativo» requiere otra decisión
metodológica; no lo autoriza aumentar `--limit`. No relabelar registros anteriores.

## 5. Pasos ejecutables, uno por turno

### Paso 0. Congelar contrato y capturar baseline

Leer este plan, AGENTS.md y el borrador de políticas. Actualizar el borrador para
reflejar las decisiones de arriba y su estado propuesto. Guardar fixtures pequeños
reproducibles con inputs, planes, evidencia, orden antes/después de diversidad y
versión. Distinguir metadata capturada de fixtures sintéticos. No guardar pesos.

Gate: mismos inputs y evidencia producen los mismos IDs y orden. Identificar las
expectativas existentes que cambiarán intencionadamente; no borrar tests para
hacer pasar la política nueva. No modificar todavía el ranking activo.

### Paso 1. Auditar y explicar el candidate pool

Seguir consultas, orden/limit de HF, interleaving, deduplicación, filtros,
presupuesto único, shortlist de hardware, inspección, planes y diversidad final.
Registrar por etapa cantidades y motivos de descarte. Reutilizar estado/reporte
existente; añadir solo los datos que falten para justificar candidatos cercanos.

Hacer como máximo una pequeña consulta metadata-only por escenario representativo
si hay red; conservar fecha, parámetros y resultados revisables. No declarar recall
en vivo a partir de fixtures. No crear cuota por marca ni subir límites a ciegas.

Gate: identificar dónde desaparece cada candidato de ejemplo. Solo corregir un
defecto genérico reproducido y añadir su regresión. La auditoría puede concluir
sin cambio de discovery. Congelar el pool antes de comparar rankings shadow.

#### Auditoría realizada: 2026-10-07

Baseline: `master`, `7aac979`. Dos capturas públicas de metadata de HF,
sin tokens, pesos, inspección profunda ni ejecución. Inputs: chat/código,
Balanced, inglés, un usuario, uso comercial. Hardware **sintético**:
2060 (6 GiB VRAM/8 GiB RAM), 4060 (8/32 GiB), CPU (32 GiB RAM).
Los nombres identifican escenarios; no son validaciones físicas de esas máquinas.

Camino comprobado en código:

1. Seis consultas por petición, `pipeline_tag=text-generation`, `gated=False`,
   `cardData=True`, límite 20 por consulta. Chat: `instruct`, `chat`,
   `multilingual instruct`, `instruct` filtrado por `gguf`, `instruct` por
   `safetensors`, `instruct` con `trending_score`. Código: `coder instruct`,
   `code generation`, `programming assistant`, `coder instruct` por `gguf`,
   `coder instruct` por `safetensors`, `coder instruct` con `trending_score`.
   Todas salvo la última ordenan por `downloads`. Buscar por nombre no demuestra
   pertinencia semántica; popularidad y trending tampoco demuestran calidad.
2. Interleaving round-robin, deduplicación y unión de etiquetas de origen.
   Filtros de metadata eliminan privados/gated, incompatibilidades conocidas
   de modalidad/artefacto/licencia; metadata escasa no implica descarte automático.
3. Los primeros 40 elegibles pasan al presupuesto. Solo entonces se escogen
   12 para inspección por heurísticas gruesas de hardware, escala y diversidad.
   Por tanto, el presupuesto único puede perder un candidato antes de comparar
   su viabilidad con hardware. El hint grueso no equivale al resultado del HFA.
4. La inspección/config/estimación puede fallar y los planes pueden ser rechazados.
   Existe un presupuesto adicional de 6 inspecciones de variantes. El engine
   ordena planes; la aplicación agrupa modelos lógicos/alternativas y aplica
   diversidad para producir hasta 5 recomendaciones. Estos pasos se trazaron
   en código y fixtures offline, **no se ejecutaron sobre las capturas reales**.

Capturas iniciadas a `18:04:58.150760 UTC` (chat) y `18:05:42.780771 UTC`
(código). Se reutilizó cada respuesta sin red para 4060 y CPU.

| Petición | Avistamientos | Repos únicos | Filtrados | Fuera del presupuesto 40 | Fuera de shortlist | Shortlist |
|---|---:|---:|---:|---:|---:|---:|
| Chat, tres perfiles | 120 | 91 | 0 | 51 | 28 | 12 |
| Código, tres perfiles | 100 | 67 | 0 | 27 | 28 | 12 |

No falló ninguna consulta. `programming assistant` devolvió cero repositorios;
las otras once consultas devolvieron 20. Cero filtrados solo describe esta muestra:
no confirma que todas las licencias, tareas o configuraciones sean adecuadas.
Los totales iguales entre perfiles no significan que se eligieran los mismos repos.

Ejemplos verificables por etapa (no son recomendaciones de calidad):

| Repositorio exacto y petición | 2060 | 4060 | CPU |
|---|---|---|---|
| `Qwen/Qwen3-4B-Instruct-2507`, chat | Shortlist | Shortlist | Shortlist |
| `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF`, chat/código | Fuera de shortlist | Shortlist | Shortlist |
| `Qwen/Qwen2.5-32B-Instruct`, chat | Fuera de shortlist | Shortlist | Shortlist |
| `Qwen/Qwen2.5-Coder-32B-Instruct`, código | Fuera de shortlist | Fuera de shortlist | Shortlist |

`Qwen/Qwen3-4B-Instruct-2507` no apareció en las consultas de código. No se
deduce de ello que sea mejor programando que los candidatos presentes, ni que
el presupuesto deba aumentar. La muestra no mide recall exhaustivo ni demuestra
una pérdida injustificada concreta. Se mantienen consultas, presupuestos y ranking.

Fallo genérico reproducido: HF utiliza `httpx` y un `ConnectError` de DNS no
entraba en la traducción de errores del cliente. Ahora `httpx.TransportError`
se convierte en el error recuperable existente, como `OSError`: el workflow
puede tolerar una consulta fallida y continuar con las demás. No se ocultan
errores arbitrarios de programación ni se modifican los resultados exitosos.

Herramienta: `python -m scripts.audit_recommendation_pool --answers answers.json
--hardware hardware.json --hardware-kind synthetic --output capture.json`.
Repetir con `--replay capture.json` usa solo metadata guardada; exige las consultas
actuales y un fichero de salida nuevo. Guarda queries, orden, metadata, fechas,
counts y etapa de descarte. No invoca el inspector ni el ranker. Las capturas
completas y los inputs de esta sesión permanecen locales en
`.codex-night/pool-live-{chat,code}-20261007.json` y `pool-audit-*.json`;
no son fixtures distribuidos. Reconsultar HF en otra fecha puede dar otros resultados.

Regresiones offline: seis escenarios con duplicados, rechazo de privacidad,
descarte por ambos presupuestos y top 5. Se contrasta el orden capturado con
`run_workflow`, no solo con una segunda implementación de la heurística.
Los resultados simulados no miden calidad, rendimiento o ajuste físico.

El reporte normal conserva los candidatos ya presupuestados, no todos los
descartes anteriores; esta herramienta cubre ese hueco de auditoría sin ampliar
el esquema de producto.

#### Cola hasta el top 5: 2026-10-07

`python -m scripts.audit_recommendation_tail --capture <pool-live>.json ...`
reproduce las mismas capturas como resultados de búsqueda (pool congelado) y
ejecuta todo lo posterior con `run_workflow` real: metadata de HF, config,
cabeceras GGUF por rango. Sin pesos, runtime ni evaluación. Dos comprobaciones
impiden que la reconstrucción invente una explicación: los 40 presupuestados
deben coincidir con los de la auditoría del pool, y el top 5 reconstruido con el
del workflow. Pasaron en los 6 escenarios. La inspección usa metadata del día
de la traza, no de la captura.

Destino de los 12 inspeccionados (antes de los arreglos):

| Escenario | Top 5 | Mismo modelo lógico | Por debajo del corte | Planes rechazados | Inspección fallida |
|---|---:|---:|---:|---:|---:|
| Chat 2060 | 5 | 1 | 6 | 0 | 0 |
| Chat 4060 | 5 | 1 | 5 | 1 | 0 |
| Chat CPU | 5 | 1 | 5 | 1 | 0 |
| Código 2060 | 5 | 3 | 2 | 0 | 2 |
| Código 4060 | 5 | 2 | 1 | 2 | 2 |
| Código CPU | 5 | 1 | 2 | 2 | 2 |

Justificado: «mismo modelo lógico» son repacks GGUF que quedan como alternativas
del original; «por debajo del corte» es orden, y la diversidad no desplazó ni
promovió ningún modelo en ningún escenario.

Dos defectos genéricos reproducidos y corregidos:

1. **Motivo falso para el MoE.** `Qwen3-Coder-30B-A3B` (unsloth GGUF y huihui)
   caía con `artifact_runtime_incompatible`. El estimador no modela el KV de
   MoE, no se elige runtime, y `unknown` no está en `compatible_runtimes`. El
   GGUF sí es compatible con llama.cpp. Ahora el código es `runtime_undetermined`
   y el mensaje dice por qué. **El rechazo se mantiene**: ajuste desconocido no
   es ajuste confirmado, y en el 2060 físico (6 GiB + 7,7 GiB) un 30B no cabe.
   Top 5 idéntico tras el cambio. El arreglo de fondo, que el estimador demuestre
   «no cabe» por los pesos aunque no modele el KV, toca fórmulas congeladas.
2. **Adaptadores gastando inspección.** El filtro solo descartaba adaptadores sin
   base declarada; la inspección descarta todos. Los de `yusifnuri` ocupaban 2 de
   12 plazas en cada escenario de código. Ahora la relación de HF
   `base_model:adapter:` los descarta antes; `lora`/`peft` sueltos no, porque un
   fine-tune fusionado puede llevarlos.

Efecto del segundo arreglo: chat idéntico en los tres perfiles. En código entran
2 candidatos más por escenario, y `secmlr/…critic_qwen_code_14b`, un fine-tune
de investigación para SWE-bench, entra en los tres top 5 por el prior de escala.
Sale `ajaythumu/code_generation_model` (CPU), `Qwen2.5-Coder-3B-Instruct` (4060)
y `deepseek-coder-6.7b-instruct` (2060). El presupuesto ya no se pierde; qué hace
el ranking con él es el problema del prior de escala, no de discovery.

Conclusión del paso 1: no queda ninguna pérdida injustificada demostrada en
estas muestras. Lo que llega al top 5 de código con prioridad Balanced incluye
modelos genéricos de baja calidad (`kaiest/Python_Code_Generation_GPT2`) y
fine-tunes de investigación, porque la capacidad es hoy un prior de escala.
Eso es el paso 2.

### Paso 2. Proyectar evidencia para las políticas

Primera pieza implementada (2026-10-07): `PlanAssessment.quality` separa el
prior de metadata, de confianza baja, de referencias históricas a registros
locales. Se emparejan por SHA256 exacto; cada referencia incluye el digest del
registro completo, muestras, métricas, clasificación y bloqueos. Dos ejecuciones
con resultados distintos permanecen visibles, sin escoger la más favorable.
El registro completo sigue siendo la fuente de protocolo, runtime y hardware:
la proyección es un índice diagnóstico, no un informe autónomo ni un quality score.
Los registros antiguos sin `quality` se siguen leyendo con un default desconocido.

Las referencias publicadas conservan su atribución en `external_evaluations`
y el filtro estricto existente; no se duplica el catálogo en este contrato.
Advisor recarga el catálogo y el store en cada nueva petición de recomendación.
Actualizar el fichero versionado o guardar un registro afecta a la siguiente
búsqueda, no modifica retrospectivamente los resultados anteriores. No hay
servicio de actualización automático ni evaluaciones durante Search.

Segunda pieza (2026-10-08): `PlanAssessment.speed` es el gemelo de `quality`
para Fastest. Por plan, referencia los benchmarks guardados del artefacto exacto
por SHA256, con tg128 y pp512 por separado (nunca el máximo entre longitudes) y
los bloqueos de `fastest_benchmark_blockers` en cada referencia. Orden: más
reciente primero, ID como desempate. Solo existen `absent` y `reference_only`:
los ajustes que reporta llama-bench son peticiones, no valores aplicados, así
que no hay cohortes ni selección hasta poder auditar el protocolo efectivo.
Los registros de calidad se validan y hashean una vez por ranking
(`PlanRankingContext.quality_index`), no una vez por plan. Gate verificado con
un test: cambiar `speed` o `quality` no altera `_ranking_key` en ninguna prioridad.

Todavía no se declara ninguna referencia local aplicable al ranking: falta
seleccionar un protocolo pertinente a tarea/idioma y verificar las condiciones
efectivas del plan. Ni siquiera `full` basta por sí solo. El rendimiento mantiene
su contrato actual, independiente. La siguiente pieza de este paso es definir
la aplicabilidad/comparabilidad requerida por las políticas shadow, sin convertir
referencias históricas en evidencia de la configuración actual.

Extender los contratos existentes con `QualityAssessment` y la información mínima
de rendimiento aplicable. Añadir registros locales a `PlanRankingContext` mediante
Advisor; nada de lecturas de disco o red dentro del ranker. Mantener lectura de
registros antiguos y exportación con atribución completa.

Tests: SHA/revisión/variante equivocados, linaje heurístico, tarea/idioma distintos,
protocolo incompatible, nulls, fallos y limited no ordenan; evidencia corrupta
produce diagnóstico y fallback. Distinguir referencia de modelo de medición del
artefacto. Tests de fixtures nuevos de tareas generativas quedan para el paso 4.

Gate: añadir evidencia diagnóstica no cambia el ranking activo. No modificar el
contrato v1 para aceptar arbitrariamente métricas o campos desconocidos.

### Paso 3. Implementar las tres políticas en shadow

Construir los planes una sola vez y evaluar ambos órdenes sobre ese mismo pool.
Aplicar el algoritmo del apartado 3; reutilizar diversidad y restricciones.
Exportar orden actual/propuesto, grupos, IDs de evidencia, razones de movimiento,
fallback, candidatos no seleccionados y versión de política. La explicación debe
venir de la decisión real, no de una segunda fórmula ni del ScoreBreakdown legacy.

Tests con datos sintéticos etiquetados: calidad pertinente vence al prior dentro
del grupo; HellaSwag no vence en código; benchmark lento medido no recibe bonus
por existir; carga tg distinta no se mezcla; ausencia/incompatibilidad conserva
fallback; Pareto ordena dominados y conserva alternativas con trade-offs.
Añadir invariancia al orden de entrada, desempates, grupos disjuntos y concordancia
entre reporte y top 5 posterior a diversidad. Memory/legacy siguen iguales.

Gate: el resultado visible normal sigue siendo el actual; el shadow no ejecuta
discovery otra vez, ni Docker, ni descargas. La evidencia sintética nunca entra
en un catálogo/store de producción.

Implementado (2026-10-08): `recommendation/shadow.py`, política `shadow-v1`.
`recommend` expone el mismo `ranked_plans` a un observador opcional; el
orquestador construye `state.shadow` desde ese pool, sin rankear dos veces. Un
fallo del shadow se registra y deja `shadow=None`; el resultado no cambia. El
informe exportado no lee `state.shadow`, así que JSON y snapshots no cambian.

Decisión de diseño: los grupos se forman dentro de tramos **contiguos** del orden
base con el mismo estrato (colocación confirmada, rechazo, confirmación comercial, suitability,
licencia, idioma). Agrupar solo por estrato dejaría que dos miembros separados
por un plan de otro estrato se intercambiaran por encima de él.
La partición de colocación confirmada coincide con la del orden activo. Balanced
usa pares de cohortes como clave, no concatenación ambigua de sus nombres. Cada
movimiento cita solo los ejes usados por la política; al iniciar una nueva
búsqueda se descarta el shadow anterior, incluso si la nueva se cancela.

`PlanEvidence` es la frontera: hoy `production_evidence` no devuelve ninguna
medida aplicable, solo motivos, y el shadow coincide con el orden activo.
Verificado en una traza real (código, 2060 sintético): 12 planes en fallback,
cero grupos. La auditoría de cola llama a `run_workflow` directamente, sin los
stores locales que carga el Advisor; sus motivos no ven registros locales.

### Paso 4. Ampliar evaluación, una suite por turno

Primero IFEval; HumanEval en un turno separado después. Inspeccionar fuentes del
harness fijado y sus APIs, no asumir soporte por el nombre de la tarea.
Modificar únicamente los puntos requeridos del productor, contratos, comparación,
storage, perfiles y UI. Versionar el esquema para nuevas formas de resultados;
seguir leyendo v1 con sus validaciones estrictas y sin migraciones destructivas.
Un registro por artefacto y suite basta; no crear un framework de plugins.

Tests HTTP verifican settings efectivos, chat/raw, stops, límites y caché apagada.
Tests de resultados verifican denominadores, extracción y métricas desde muestras,
completitud, identidad, hit exacto/miss por protocolo o artefacto y lectura antigua.
HumanEval añade prueba de aislamiento y timeout del código generado.

Gate real por suite: smoke de un artefacto local verificado; después dos artefactos
pequeños, secuenciales, si existen y caben. Guardar comandos, resultados, muestras
y errores. Las corridas completas necesitan un presupuesto explícito de tiempo
y disco aprobado por Ton; no iniciar campañas de horas automáticamente.
Si falta un requisito, terminar código/tests seguros y señalar el gate bloqueado.

#### Auditoría de IFEval en el pin: 2026-10-08

Fuentes leídas en el commit `ad8737ae…` (manifiesto con SHA256 de cada fichero
en `.codex-night/ifeval-audit-20261008/`): tarea `ifeval` v4.0, sus checkers,
y los backends `gguf`, `local-chat-completions` y `TemplateAPI`. Nada ejecutado.

Lo que fija la tarea: dataset `google/IFEval` **sin revisión** (`train`, 541
prompts); cero shots; `until: []`, `temperature 0`, `do_sample false`,
`max_gen_toks 1280`. Métricas: `prompt_level_strict_acc` (primaria),
`inst_level_strict_acc`, y las dos `loose`. Una respuesta vacía falla todo.
Revisión actual del dataset: `966cd89545d6b6acfd7638bc708b98261ca58e84`,
un fichero `ifeval_input_data.jsonl`.

Hallazgos que obligan a decidir algo:

1. **El backend del piloto no aplica plantilla de chat.** `gguf` manda el prompt
   en crudo a `/v1/completions` y devuelve `text.strip()`. Un modelo instruct
   recibe una cadena sin marcas de turno, y sin stops el límite real es EOS o
   1280 tokens.
2. **`local-chat-completions` sí sirve, y la plantilla la aplica el artefacto.**
   Con `tokenizer_backend=None` envía `messages` sin renderizar: llama-server
   usa la plantilla jinja incrustada en el GGUF. `/apply-template` permite
   registrar el texto final y su hash por artefacto.
3. **Fallos convertidos en mala calidad.** Si `parse_generations` no puede leer
   la respuesta, guarda `""` y se puntúa como instrucciones no seguidas. El
   productor debe rechazar la ejecución ante cualquier fallo de parseo.
4. **Razonamiento puntuado como respuesta.** Sin `think_end_token`, un `<think>`
   cuenta como texto. El build fijado tiene `--reasoning on|off|auto` y
   `--reasoning-format`; el modo debe ser parte del protocolo y verificarse en
   cada respuesta (`reasoning_content` vacío si se fija `off`).
5. **Descarga de red al importar.** `instructions_util` llama a
   `nltk.download("punkt_tab")` si falta. Exige `nltk>=3.9.1`. `punkt_tab` debe
   ir dentro de la imagen con hashes, sin descarga durante la evaluación. El
   contenedor usa red bridge para llegar al servidor del host; esto no equivale
   a aislamiento de red ni bloquea por sí solo el acceso externo.
6. **`langdetect` sin semilla.** Tres checkers usan `langdetect.detect` y nadie
   fija `DetectorFactory.seed`: la misma respuesta puede puntuar distinto entre
   ejecuciones. Fijar la semilla es una modificación, así que la suite es propia
   (`jaull-ifeval-*-v1`) y no comparable con cifras publicadas.
7. **El truncado no consta.** El harness no registra `finish_reason`. Las
   respuestas cortadas a 1280 se cuentan desde el HTTP capturado y siguen en el
   denominador.

Decidido por Ton (2026-10-08): **chat con la plantilla del GGUF** (hallazgo 2) y
**razonamiento off verificado**: `--reasoning off`, y cualquier respuesta con
`reasoning_content` o etiquetas de pensamiento invalida la ejecución entera. Los
modelos con razonamiento quedan etiquetados como medidos en modo no-thinking.

Contrato: chat vía
`local-chat-completions` y plantilla del GGUF; `num_concurrent 1`, `--parallel 1`,
`--no-cache-prompt`, sin `--cache_requests`; payload verificado en HTTP
(`stop` vacío, `temperature 0`, `max_tokens 1280`, `seed`, ninguna clave extra);
`DetectorFactory.seed = 0`; contexto que quepa prompt renderizado + 1280 o falla
cerrado; solo el split completo es reutilizable, un smoke de claves fijas es
`plumbing`. Presupuesto **estimado, no medido**: con ~50 tok/s, 541 respuestas
de ~300 tokens son ~1 h por modelo; si ninguna emite EOS, ~4 h.

Productor implementado (2026-10-08): `pilot/quality_eval/ifeval.py` y
`suite_ifeval.yaml`; perfiles `ifeval-smoke` (3 prompts, `plumbing`) e `ifeval`
(541, `full`); imagen con `langdetect`/`immutabledict` fijados por hash y
`punkt_tab` por commit y SHA256. Al construir, la imagen verifica los seis
digests auditados contra los módulos instalados y la suite real; ambos pasaron.
Las imágenes sin la etiqueta `io.jaull.quality.suites` se rechazan para IFEval.
En esa primera versión, un bundle de IFEval no se importaba como registro:
faltaba la forma de resultado de generación, implementada más abajo.

Revisión del productor (Codex, 2026-10-08): el preflight exige una plantilla
`tokenizer.chat_template` textual incrustada en el GGUF. Rechaza metadata
ausente, vacía o el alias `chatml`, porque llama.cpp puede sustituirla por su
plantilla interna; una cadena no vacía en `/props` no demuestra su procedencia.
Las plantillas nombradas sin plantilla textual por defecto quedan fuera de
este piloto. El hash de `/props` identifica la plantilla efectiva del runtime,
que puede aplicar sus propios ajustes, no garantiza bytes sin modificar del
GGUF. El validador HTTP también rechaza contadores de tokens negativos o
un prompt de cero tokens. HellaSwag no necesita plantilla y no cambia.

El límite de contexto se verifica actualmente con el uso reportado **después**
de cada respuesta; no es un preflight de tokenización de todos los prompts.
La ejecución falla cerrada y no genera un bundle válido si ese límite se viola.

Smoke real: Qwen2.5-1.5B-Instruct Q5_K_M, RTX 2060, imagen
`sha256:683154e2…`, bundle `.codex-night/ifeval-smoke-20261008-qwen-01`. Las tres
peticiones con las claves auditadas, `cached_tokens 0`, sin razonamiento. Dos
hechos que el contrato no preveía: la plantilla del GGUF **inserta su propio
prompt de sistema** («You are Qwen…») aunque no se envíe ninguno, y una de las
tres respuestas llegó a 1280 tokens (`length`), contada y dentro del
denominador. 1/3 estricto: plumbing, no calidad. Generación a ~155 tok/s.
Presupuesto recalibrado, con una media de solo tres muestras: ~617 tokens por
respuesta; ~35-40 min el 1,5B y ~1,5-2 h un 7B Q4 en la 2060.

Registro de generación (2026-10-08): `schema_version 2`, separado de la v1 de
loglikelihood; un registro nunca cambia de esquema y la comparación entre ambos
es `NOT_COMPARABLE`. La identidad fija plantilla `gguf`, razonamiento `off`,
`until: []`, temperatura 0, 1280 tokens y semilla. Las cuatro métricas se
recalculan desde las muestras; la de prompt debe ser `all()` de sus
instrucciones. `outcome` guarda respuestas y truncados. El productor exige que
cada respuesta puntuada sea el texto exacto de un intercambio HTTP auditado.

Revisión del lector v2 (Codex, 2026-10-08): los argumentos deben conservar la
serialización real de `JsonChatStr`, un único mensaje de usuario y la configuración
de generación fijada. El lector verifica el hash del prompt y del target, y
recalcula `doc_hash` con la serialización del harness; además exige que el prompt
del documento coincida con el mensaje enviado. Argumentos ausentes, documentos
alterados o instrucciones cambiadas no pueden reutilizar la identidad anterior
ni producir una comparación falsamente válida. La lectura v1 no cambia.

La plantilla servida no es la del GGUF byte a byte: llama-server le quita el
espacio final (2509 → 2508 caracteres en Qwen2.5). El host guarda el SHA256 de
la plantilla incrustada y el de su `strip()`; el importador exige que la servida
sea una de las dos y rechaza cualquier otra normalización. Comprobado en dos
bundles reales.

Comparación: la plantilla queda ligada al artefacto, como su tokenizer, así que
no impide comparar dos modelos; su origen `gguf` sí debe coincidir. Las métricas
de prompt se emparejan por prompt y llevan bootstrap cuando procede; las de
instrucción no, porque las instrucciones van agrupadas dentro de cada prompt.

Smoke real reimportado: imagen `jaull-quality-eval:ifeval-chat-v1-r2`
(`sha256:fea1e5c5…`), bundle `ifeval-smoke-20261008-qwen-02`, registro v2 leído
por el contrato del producto y por el store: 1/3 prompts, 2/5 instrucciones,
1 truncado, `plumbing`, no reutilizable.

Gate de dos artefactos pequeños: split completo de Qwen2.5-1.5B Q5_K_M y
TinyLlama-1.1B Q4_K_M, los mismos del piloto de HellaSwag, en secuencia.
El primer intento completo de Qwen quedó interrumpido con 58 intercambios HTTP:
sin resultados y estados finales, no es un registro importable. No se mezclan
esas respuestas con una nueva ejecución ni se reutiliza su caché.

Evidencia aplicable en el shadow (2026-10-08): `QualityAssessment` pasa a
`controlled_artifact` solo con perfil `ifeval-instructions-en-v1` (chat, inglés),
un único registro v2 `full` y reutilizable de `jaull-ifeval-chat-v1` en modo
`chat_generation` para el SHA exacto. La métrica es `prompt_level_strict_acc`,
declarada antes de ver resultados. La cohorte es `comparison_key`, la misma
función que decide si el comparador acepta dos registros. Varias ejecuciones
completas del mismo artefacto se abstienen. Un registro de loglikelihood
renombrado a la suite no se promueve. El shadow lo usa como `Measured`; el
ranking activo no cambia.

Revisión de aplicabilidad (Codex, 2026-10-08): el nombre de la suite y un
`full` declarado por el registro no prueban el perfil auditado. Se exige su
SHA256 de suite, dataset y revisión fijados, los 541 IDs, commit y hashes de las
seis fuentes del evaluador, contexto 4096, semilla del detector 0 y binario/build
de llama-server auditados. Una desviación válida se conserva como referencia,
pero no ordena el shadow. La clave de cohorte sigue verificando el resto de las
condiciones entre los dos registros. Esto comprueba las declaraciones del
registro; la trazabilidad HTTP y la verificación de bytes corresponden al productor,
no se convierten en una firma criptográfica de confianza.

Gate de dos artefactos pequeños, cerrado (2026-10-08). TinyLlama-1.1B fue rechazado
en el preflight: su contexto máximo es 2048 y el contrato fija 4096. Bajárselo solo
a él lo haría incomparable, así que lo sustituyó Qwen2.5-0.5B-Instruct Q4_K_M
(manifiesto desde la revisión y el sidecar en caché; bytes reverificados por el
runner). Una primera ejecución de Qwen2.5-1.5B quedó interrumpida (58/541) y su
bundle no se puede importar.

| Split completo, 541 prompts | Qwen2.5-1.5B Q5_K_M | Qwen2.5-0.5B Q4_K_M |
|---|---:|---:|
| prompt estricto | 222 (41,0 %) | 138 (25,5 %) |
| prompt laxo | 243 (44,9 %) | 151 (27,9 %) |
| instrucción estricta (834) | 428 (51,3 %) | 311 (37,3 %) |
| truncadas a 1280 | 65 (12 %) | 202 (37 %) |
| duración | 27 min | 28 min |

`COMPARABLE_DIAGNOSTIC`: ocho comprobaciones coinciden, misma clave de cohorte; la
plantilla resulta idéntica en ambos (misma familia). En prompt estricto, 118 prompts
los acierta solo el 1,5B y 34 solo el 0,5B. Sin intervalo: el split es completo y la
cifra es exacta para él. Ninguna respuesta vacía ni fallo de parseo; el EOS funciona
(476 y 339 respuestas terminan solas). Tamaño y cuantización difieren, así que no
aísla el efecto del tamaño. Buena parte de la distancia es degeneración: 53 de las
202 truncadas del 0,5B terminan repitiendo una línea, frente a 12 de 65.
Los registros siguen en `.codex-night/`; no se importaron al store del usuario.

Revisión con búsqueda normal (Codex, 2026-10-08): chat en inglés, Quality,
uso comercial, RTX 2060 detectada. Los dos registros se cargaron en un store
temporal, sin importar al store personal ni ejecutar evaluaciones. 40 candidatos,
12 planes, cero registros aplicables y cero movimientos: el plan GGUF del 0,5B
usaba Q6_K, no los bytes Q4_K_M medidos; el 1,5B no llegó a inspección profunda.
El orden activo era idéntico con y sin registros sobre el mismo pool. Esta muestra
no demuestra recall ni un primer movimiento real del shadow. No se cambian los
presupuestos ni se fuerzan modelos para obtenerlo: Evaluate candidates tendrá que
medir los artefactos que realmente llegan al pool. Traza local ignorada:
`.codex-night/ifeval-shadow-review-20261008.json`.

### Paso 5. Evaluate candidates y Recompare

Extender los flujos existentes: elegir suite según tarea, mostrar qué medirá y
permitir seleccionar uno a tres candidatos. Quality pide calidad; Fastest usa
benchmark existente; Balanced ofrece ambos. Mostrar evidencia reutilizable y
qué falta antes de proponer ejecuciones. Nunca ejecutar al entrar en Results.

Reutilizar preparación/verificación/download consentido, cancelación y persistencia
por candidato. No borrar modelos compartidos ni añadir desalojo automático aquí.
El usuario puede conservar el diagnóstico sin reordenar nada.

`Recompare using new evidence` reutiliza el pool y los requisitos guardados con
una nueva instantánea de evidencia; produce un resultado separado y explicable.
No relanza HF ni modifica la búsqueda original. Un fallo individual no invalida
los resultados completos de los demás candidatos.

Gate: tests fake de principio a fin Search -> Evaluate -> Recompare, cancelación,
reuse exacto y aislamiento entre artefactos. Validar TUI pequeña/grande con esperas
por estado, aprovechando los arreglos recientes de CI Windows.

**Incremento 5A (2026-10-08): pool guardado y API de Recompare.** El estado de una
búsqueda completada conserva `ranked_plans`, en el orden previo a diversidad.
`AdvisorService.recompare_quality(state)` lee una nueva instantánea del store de
calidad y devuelve un `ShadowReport` separado. Actualiza solo calidad: conserva
planes, requisitos, restricciones, viabilidad, velocidad y orden de fallback de
Search. No consulta HF, reestima memoria, prepara artefactos ni ejecuta modelos.
Una búsqueda nueva limpia el pool anterior, incluso si se cancela; un estado
antiguo sin pool sigue cargando pero necesita otra búsqueda para Recompare.
Los informes de recomendación exportados no cambian de esquema ni de contenido.

Probado offline con registros sintéticos: cambio de orden Quality dentro de una
cohorte, SHA distinto sin herencia, evidencia aislada sin promoción, ausencia o
registro corrupto sin pérdida de candidatos, fallback de código y otras
prioridades, y conservación de la búsqueda original. Esto **no completa el
paso 5**: la selección desde el pool guardado, el perfil IFEval del runner/TUI,
el botón Recompare y el test fake Search -> Evaluate -> Recompare quedan
pendientes. Ninguna descarga ni nueva medición GPU en este incremento.

Segundo incremento (2026-10-08): perfil `ifeval` en el puente del producto (grado
`full`, contexto 4096, dataset propio; el setup del piloto recibe `--suite`;
recordar el setup no pisa el dataset de HellaSwag). Con IFEval, los candidatos son
**los planes exactos que mostró la búsqueda**, sin re-rankear ni recorrer la
escalera de cuantización: evaluar otra cuantización mediría bytes que ningún plan
mostrado ejecuta, y su resultado no podría ordenar esta búsqueda. El ajuste se
comprueba a 4096. La regla «chat en inglés» sale de `requested_profile`, la misma
función que usa `assess_quality`, así que la TUI no puede ofrecer una suite que la
evaluación luego no acepte. En la TUI, IFEval solo aparece en búsquedas de chat en
inglés y es el perfil por defecto; cambiar de perfil rehace la selección; un
dataset escrito a mano no se pisa. Botón «Recompare this search with stored
quality» en el cuerpo (en la barra fija no cabía a 80 columnas): muestra la
propuesta aparte y deja intactas las recomendaciones. Capturas a 80 y 160
columnas en `.codex-night/ifeval-tui-captures/`.

Incremento de verificación (2026-10-09): test offline encadenado con
`AdvisorService.recommend` -> selección de los planes exactos -> cola secuencial
-> verificación SHA256 y persistencia reales -> Recompare. Solo las fronteras
externas (metadata, hardware, descarga y productor) son sintéticas. Comprueba un
movimiento Quality, permiso de descarga, reutilización del resultado guardado,
cancelación del segundo candidato sin perder el primero y aislamiento frente a
otro SHA. La búsqueda original y su exportación permanecen intactas; Recompare
no redescubre ni reestima.

El informe muestra por plan su artefacto, SHA y motivos de fallback: ausencia de
evidencia exacta, referencias fuera del contrato, conflicto de repeticiones,
medición aislada, protocolo distinto o separación por estrato. Tests TUI a 80x24
y 160x50 comprueban que el informe largo no oculta las acciones fijas.
Este gate offline de calidad queda cubierto; no implica activar el ranking ni
desbloquear Fastest/Balanced. Windows y la campaña real desde Search siguen
pendientes de validación.

Verificación de esta tanda: 59 tests específicos, Ruff y mypy (253 ficheros) en
verde. El resto de la suite pasa con 2580 tests y uno omitido, excluyendo tres
casos bloqueados: confirmación de borrado en Storage y los dos tamaños del
Estimate manual. Esos tests no se han modificado; la suite completa sin
exclusiones no queda certificada aquí.

### Paso 6. Revisar shadow y activar

Revisar una matriz offline de 3 perfiles de hardware x chat/código x las tres
prioridades: 18 combinaciones pequeñas, no 18 campañas GPU. Identificar los perfiles
sintéticos; solo hardware real medido cuenta como validación física.
Añadir casos de otros usos/idiomas para verificar fallback sin falsas afirmaciones.

Elegir 3-5 casos explicativos para revisión, incluidos ausencia/conflicto de evidencia
y un trade-off de Balanced. Para comprobar el camino nuevo con datos reales se
necesita al menos un par comparable completo y benchmarks aplicables. La campaña
más amplia y las gráficas del TFG pertenecen a la fase 2.

Ton revisa el informe shadow antes de activar el default. Esta es la única puerta
de aprobación de producto del plan; los pasos anteriores dejan diffs revisables.
Activación: las políticas nuevas pasan al flujo normal; conservar el baseline
necesario para fallback y regresiones, sin duplicar motores enteros. Actualizar
`ranking_criteria`, docs y schema/snapshots cuando el contrato exportado cambie.

Gate: top 5 real coherente con su explicación, viabilidad/restricciones intactas,
ninguna ejecución durante Search, fallback visible. Si no hay evidencia suficiente,
declarar ese caso cubierto por fallback; no afirmar una validación empírica inexistente.

**Preparación para revisión (2026-10-09), sin activar nada.**
`scripts/shadow_review_matrix.py` reproduce las capturas de chat y código del 07/10
en 3 perfiles (los tres sintéticos) x Quality/Speed/Balanced, más sondas de idioma:
27 casos sin descargas ni ejecución, con los registros reales del store. En todos,
el top 5 activo es idéntico con y sin calidad, y ninguno se mueve. La causa es la
cobertura: con Quality, el pool no contiene ninguno de los GGUF medidos, y los modelos
visibles suelen ser safetensors sin SHA. El camino sí se demuestra con un par controlado
real (`scripts/audit_quality_recompare.py`): Qwen2.5-1.5B Q5_K_M (41,0 %) pasa a #1 sobre
Qwen2.5-0.5B Q5_K_M (25,9 %) en la misma cohorte y estrato. LFM2.5 (80,2 %) no se ordena
frente a Qwen porque su licencia la sitúa en otro estrato. No hay trade-off de Balanced
demostrable mientras la velocidad no aplique. El informe con las opciones de activación
está en `.codex-night/step6-shadow-review-20261009.md`; la decisión es de Ton.

**Contraste adicional (2026-10-10), sin activación.**
`.codex-night/quality-recompare-coverage-20261009.json` reproduce la búsqueda
capturada con prioridad Quality y consulta las alternativas por separado.
Solo Qwen2.5-0.5B tiene una ruta GGUF en el pool guardado; los otros cuatro
modelos visibles tienen alternativas confirmadas por metadata (26, 7, 14 y 12
variantes), pero no pertenecen a ese pool ni tienen viabilidad de evaluación
comprobada. No añadirlas silenciosamente ni heredar resultados entre cuantizaciones.
El par Qwen/LFM sigue separado por licencia. Recompare muestra un motivo principal
por plan y deja SHA y limitaciones completas en desplegables; cambiar de suite
limpia esos detalles. Los tests focalizados verifican ambos tamaños de terminal.
La nueva pasada completa se interrumpió al reproducir el bloqueo de cierre de
Storage: no constituye otro gate en verde ni invalida las pasadas previas registradas.

**Activación de Quality (2026-10-10), aprobada por Ton.** Opción A del informe:
`apply_quality_policy` aplica la regla Quality de shadow-v1 al orden activo, solo con
prioridad Quality y tras la clave completa. `recommend` sigue entregando el orden base a
`on_ranked_plans`, así que Recompare y el fallback parten de él. Speed y Balanced siguen
en shadow hasta que la evidencia de velocidad aplique. Tras la auditoría de activación,
el informe separa `ranking.criteria` (orden base) de `ranking.quality_policy` (medición,
cohorte, digest del registro y posiciones del pool antes/después de Quality, antes de
diversidad). No presenta la calidad medida como otro desempate lexicográfico. Recompare
calcula sobre el pool base pero contrasta la propuesta con las recomendaciones visibles
guardadas: si ya tienen el orden Quality, no anuncia esos cambios como nuevos. El
contrato exportado se ha actualizado en el test de fidelidad. Tests: la regla sigue la
evidencia y no toca nada sin
ella ni con otras prioridades; `recommend` muestra el orden Quality y guarda el pool base
(el test falla si se quita la activación). La matriz repetida con la política activa da los
27 casos con el orden que marca cada política, y 0 cambios en búsquedas normales: la
cobertura queda declarada como fallback, como pide el gate. Pendiente para la campaña:
ampliar la cobertura GGUF del pool (lista aparte de alternativas medidas) y la
comparación de configuraciones BF16/Q8/Q5/Q4.

### Paso 7. Cierre de la fase funcional

Revisar el diff completo y ejecutar Python 3.12 pytest, Ruff, mypy, arquitectura,
compileall y diff check. Confirmar CI Linux/Windows; no sustituir esa confirmación
por «funciona en WSL». Actualizar estado real y pendientes acotados.

No marcar la fase completa si solo funciona shadow, falta la aprobación de
activación o se omitieron los gates reales. Si HumanEval queda bloqueado, Ton
decide explícitamente diferirlo y cerrar código con fallback; no cambiar el alcance
en silencio. Después se congela funcionalidad y empieza la campaña final del TFG.

## 6. Instrucciones para el agente ejecutor

Trabajar un paso por turno y leer solo los archivos que ese paso necesita.
Consultar AGENTS.md, estado Git y el diff antes de editar. No cambiar ramas,
stagear, hacer commits ni publicar. Trabajar en `master`, según la autorización
de Ton. Preservar cambios ajenos.

Llevar progreso local ignorado en `.codex-night/RECOMMENDATION_FINAL_PROGRESS.md`:
paso, decisión, archivos, comandos/resultados reales, bloqueos y siguiente paso.
No almacenar secretos ni inventar fixtures presentados como mediciones.

Validación durante desarrollo: tests focalizados del paso y Ruff; mypy si cambia
`src`. Full gates al final y antes de activar. No repetir la suite completa por
cada edición de documentación. Fuentes externas deben ser primarias, con pins.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --offline --python 3.12 pytest
UV_CACHE_DIR=/tmp/uv-cache uv run --offline --python 3.12 ruff check .
UV_CACHE_DIR=/tmp/uv-cache uv run --offline --python 3.12 mypy src
UV_CACHE_DIR=/tmp/uv-cache uv run --offline --python 3.12 pytest tests/test_architecture_dependencies.py
UV_CACHE_DIR=/tmp/uv-cache uv run --offline --python 3.12 python -m compileall -q src tests
git diff --check
git status --short
git diff --stat
git diff --cached --stat
```

Si offline falla por dependencias ausentes, informar del requisito; no afirmar que
pasó. Si un paso exige cambiar las decisiones de este plan, documentar el conflicto
antes de ampliar el alcance. No añadir benchmarks, tareas o servicios para llenar
el tiempo. Terminar cada turno con el paso completado, validación y siguiente paso.
