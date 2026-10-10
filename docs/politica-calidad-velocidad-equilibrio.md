# Política de Quality, Fastest y Balanced

Contrato propuesto para Recommendation vFinal, aprobado para implementación
por etapas. **El ranking nuevo todavía no está implementado ni activado.**
El plan ejecutable está en [recommendation-vfinal-plan.md](recommendation-vfinal-plan.md).

Versión propuesta: `recommendation-vfinal-v1`. Baseline capturado:
`engine-v2-7aac979`. Ton autoriza trabajar en `master`, con todos los cambios
sin stagear. La activación del nuevo default sigue requiriendo revisar el shadow.

## 1. Alcance

Las preferencias del wizard son Best quality, Fast responses y Balanced.
Cambian cómo se ordenan planes elegibles. Se preservan las restricciones de
licencia, idioma y tarea, el HFA, las fórmulas de memoria y la colocación.
Memory y el camino legacy sin hardware conservan su comportamiento.

Search consulta metadata y evidencia existente. Las evaluaciones Docker y los
benchmarks son acciones explícitas posteriores. Instalar un runtime afecta a
la preparación operativa; no convierte un modelo en mejor candidato.

La ordenación no suma calidad y velocidad en un score global. Los priors de
metadata/tamaño existentes permanecen como fallback identificado como heurístico.
No se introduce un predictor de velocidad ni se calibra el prior con mediciones.

## 2. Qué autoriza la evidencia

| Evidencia | Uso propuesto |
|---|---|
| Medición completa del artefacto exacto | Puede ordenar si es pertinente y comparable |
| Medición limited o smoke, incompleta o fallida | Diagnóstico; no ordena por calidad |
| Publicación con sujeto y condiciones suficientes | Puede ordenar solo en el alcance demostrado y entre sujetos comparables |
| Publicación con revisión o representación desconocida | Referencia visible; no demuestra calidad del artefacto |
| Sin evaluación aplicable | Fallback explícito; ausencia no es cero ni mala calidad |

La procedencia no establece por sí sola una jerarquía LOCAL > PUBLISHED > PRIOR.
Las evidencias coexisten. Una medición local no borra referencias publicadas.
Confirmar el linaje de un repack permite mostrar referencias al modelo base;
no transfiere sus resultados a la cuantización propuesta.

El catálogo actual conserva referencias con revisión y precisión desconocidas:
no autoriza ordenar GGUF por esas cifras. Nunca completar campos desconocidos
para hacer pasar las puertas de atribución o comparabilidad.

La calidad se proyectará por tarea en `QualityAssessment`: métricas, unidad,
dirección, identidad, protocolo, cobertura, procedencia, limitaciones y motivo
del fallback. Conservar referencias a los contratos existentes, sin crear otro
motor de capacidades ni duplicar registros completos en cada recomendación.

Completar una suite no implica calidad STRONG ni confianza HIGH. Inicialmente
mostrar valores y alcance sin categorías absolutas inventadas. Muestras
procesadas y respuestas correctas son cantidades diferentes.

## 3. Comparabilidad y datos ausentes

Para ordenar por calidad se exige identidad aplicable, tarea/idioma pertinentes,
registro completo reutilizable y mismo protocolo de comparación. En v1 se
requiere el split completo; un subset fijado sigue siendo limited.

La puerta actual de calidad comprueba suite, dataset, muestras, evaluador,
runtime y protocolo, incluyendo colocación. HellaSwag no ordena una petición
de código ni demuestra por sí solo capacidad general. Protocolos de chat/raw,
budgets o configuraciones incompatibles permanecen NOT COMPARABLE.

La política propuesta usa grupos disjuntos de comparación, no un comparador por
pares para `sort()`. La suite y su métrica primaria se fijan por tarea antes de
ver resultados; no se escoge el benchmark más favorable a cada candidato.

Dentro de los grupos que preservan viabilidad, requisitos confirmados y tarea:

1. Partir del orden determinista actual.
2. Identificar grupos con evidencia aplicable y comparable.
3. Reordenar únicamente las posiciones ocupadas por miembros de cada grupo.
4. Mantener el fallback para los demás y para registros aislados.
5. Aplicar después la diversidad y deduplicación existentes, explicando también
   las exclusiones y los cambios debidos a esa selección.

Así, estar medido no concede un bonus. Un candidato desconocido puede aparecer
entre dos evaluados por su posición base. Se debe mostrar que no existe una
comparación global completa. Sin evidencia nueva aplicable se conserva el orden
actual. No forzar un ganador ante evidencias incompatibles o conflictivas.

## 4. Quality

Ordenar cada grupo comparable por la métrica primaria pertinente, en su dirección
declarada. Empates conservan el orden base. Mostrar métricas secundarias por
separado; no promediarlas en una cifra de inteligencia general.

Suites previstas: IFEval para seguimiento de instrucciones y HumanEval para
resolución de problemas de código. Su soporte generativo todavía debe construirse
y verificarse. El piloto actual solo produce HellaSwag smoke y 100 muestras.
Otros usos e idiomas no cubiertos mantienen referencias y fallback explícito.

## 5. Fastest

Comparar rendimiento de planes completos sobre la máquina aplicable, con carga,
metodología, build y flags efectivos conocidos. La identidad de cada artefacto
debe coincidir con su propio plan. No trasladar una medición RTX 4090 a RTX 2060.

Métrica primaria inicial: generación `tg128`. Mostrar `pp512` por separado.
No escoger el máximo de velocidades medidas con distintas longitudes. No usar
la duración de una evaluación de calidad ni deducir TTFT de pp512.

Un plan con otro offload puede competir como plan completo si la observación
corresponde a él. Esto no relaja la puerta de colocación de calidad.
`--ctx-size` heredado en un record de llama-bench no significa que se ejecutara.

Sin mediciones comparables, conservar el fallback actual y etiquetarlo. No llamar
a sus desempates predicciones de throughput. Un registro sin campos necesarios
permanece histórico, sin forzar su uso en la política nueva.

## 6. Balanced

Aplicar frentes Pareto únicamente a grupos con ambas métricas comparables.
A domina B si no es peor en ninguna y es estrictamente mejor en al menos una.
Ordenar los frentes y conservar el orden base dentro de cada frente.

Pareto identifica dominancia entre las métricas observadas. Dentro de una misma
frontera puede haber un modelo muy rápido y otro de mayor calidad: v1 conserva
el desempate del baseline y lo declara en la explicación. No presenta ese
desempate como una elección de equilibrio derivada de los benchmarks.

No hay pesos, epsilon ni umbrales intuitivos. Con un eje desconocido no se
declara dominancia. El mínimo `min_generation_tps` no se convierte en una nueva
puerta de elegibilidad en esta fase; los SLO medidos pertenecen a otro alcance.

Un candidato con menor calidad y mayor velocidad puede seguir siendo Pareto.
Eliminarlo porque gana «poca» velocidad requeriría otra preferencia explícita.
Categorías HIGH/MEDIUM no sustituyen métricas comparables para decidir dominancia.

## 7. Incertidumbre y afirmaciones

Ordenar métricas de un protocolo no demuestra diferencias de capacidad general.
Mostrar cobertura, repeticiones y dispersión donde existan. No inventar intervalos
ni tolerancias para decidir empates. El bootstrap de comparaciones limited sigue
siendo exploratorio y no autoriza ordenar en esta fase.

Un resultado completo es exacto para ese split bajo su ejecución. No elimina
limitaciones de cobertura, dataset, generación o representación del modelo.
Los resultados CPU/GPU de la sonda de reparto justifican conservar las condiciones
de ejecución; no demuestran invariancia entre máquinas o GPUs distintas.

## 8. Shadow, recompare y activación

Construir los planes una vez y comparar ambas políticas sobre el mismo pool y
la misma instantánea de evidencia. Registrar versión, orden antes/después de
diversidad, IDs utilizados, grupos comparables, razones y fallback.

Evaluar candidatos es opt-in y secuencial. Recompare utiliza el pool y los
requisitos conservados con una nueva instantánea de evidencia. Produce otro
resultado vinculado al anterior; no modifica búsquedas o registros históricos
ni relanza discovery silenciosamente.

Tras revisar shadow se activará la política nueva. La fase no termina con shadow
como única implementación. Los gates, suites y regresiones están en el plan.

## 9. Baseline reproducible

`tests/snapshots/recommendation-policy-baseline.json` conserva inputs sintéticos,
IDs y órdenes observados del escenario de seis modelos ya utilizado en
`tests/test_recommendation_shortlist_regression.py`. Incluye las cuatro prioridades,
el orden de planes previo a diversidad y el top 5 final, con el plan insuficiente
excluido. No contiene resultados HF en vivo ni mediciones de calidad o velocidad.

Los tests verifican repetición e inversión del orden de entrada. El snapshot
documenta el fallback actual; no es evidencia de superioridad de esos modelos.
Las expectativas de ranking existentes solo cambiarán cuando una política activa
y revisada lo requiera. Las invariantes de HFA/elegibilidad, Memory y legacy se
conservan. No regenerar este baseline para ocultar cambios involuntarios.

## 10. Contrato de aplicabilidad v1

Definido el 2026-10-07 para implementar el siguiente paso. Estas puertas son
**parcialmente implementadas como diagnóstico, no como ranking**.
`QualityAssessment` selecciona `requested_profile` a partir de la tarea y los
idiomas explícitos normalizados (`en`), conserva el prior y distingue `absent`
de `reference_only`. El perfil solicitado es un objetivo, no una suite habilitada.
El contrato local actual solo admite scoring de continuaciones; incluso un
registro completo o renombrado como IFEval/HumanEval sigue siendo referencia.
Las publicaciones atribuidas conservan su procedencia y no reemplazan registros
locales. Datos inválidos se diagnostican, pero no crean evidencia.
`APPLICABLE`, `COMPARABLE`, `CONFLICTING`, los grupos de comparación y la
autorización de rendimiento siguen pendientes. Existen comprobaciones offline
de prerrequisitos de Fastest (10.4); no habilitan registros v1 para ordenar.
No hay cambio del orden activo ni shadow
capaz de reordenar todavía. No se inventa un contrato generativo para desbloquearlos.

### 10.1 Perfil elegido antes de mirar resultados

Cada política recibe un perfil versionado con tarea, idioma, suite, métrica
primaria, dirección, protocolo y alcance. No elige otra suite por tener más
candidatos evaluados o cifras mejores. Cambiar cualquiera de esas decisiones
crea una versión nueva; no redefine el significado de registros antiguos.

| Petición | Perfil inicial previsto | Métrica primaria | Alcance autorizado |
|---|---|---|---|
| `general_chat`, inglés | IFEval completo, zero-shot | `prompt_level_strict_acc`, fracción, mayor es mejor | Seguimiento de instrucciones; no calidad global de conversación |
| `coding`, inglés | HumanEval completo, zero-shot, una generación por problema | pass@1, fracción, mayor es mejor | Problemas de Python de ese dataset; no todos los lenguajes o proyectos |
| `reasoning` | Ninguna suite activa v1 | Ninguna | Referencias y fallback; no ordenar por HellaSwag o IFEval |
| `document_qa`, `summarization_extraction`, `writing_translation` | Ninguna suite activa v1 | Ninguna | Referencias y fallback; documentos empresariales se dejan para Biosfer |
| Cualquier petición en otro idioma o varios idiomas | Ninguna suite validada v1 | Ninguna | No extrapolar un benchmark inglés al conjunto de idiomas solicitado |

Se permiten métricas secundarias visibles, nunca sustitución oportunista de la
primaria. HellaSwag smoke/100 conserva su función diagnóstica; completar HellaSwag
tampoco habilitaría el perfil de instrucciones ni el de código.

La [configuración IFEval del pin actual](https://github.com/EleutherAI/lm-evaluation-harness/blob/ad8737ae7fad24cf64e50fc7fc31397bff586b9e/lm_eval/tasks/ifeval/ifeval.yaml)
declara generación, zero-shot y métricas strict/loose. La
[configuración HumanEval upstream](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/humaneval/humaneval.yaml)
declara código inseguro, una generación y pass@1. Upstream no es un pin ejecutable:
antes de habilitar HumanEval se fijarán commit, dataset, suite y aislamiento.
Estas consultas verifican las tareas, no el soporte efectivo del backend GGUF.

### 10.2 Dos puertas, no una jerarquía de fuentes

Primero se atribuye una evidencia al sujeto; después se comprueba su pertinencia
al perfil y su comparabilidad. Estados propuestos para el informe shadow:

| Estado | Significado | Uso |
|---|---|---|
| `ABSENT` | No existe evidencia atribuible | Fallback, nunca puntuación cero |
| `REFERENCE_ONLY` | Sujeto conocido, pero tarea, cobertura o configuración no aplicables/confirmadas | Mostrar fuente y bloqueos; no ordenar |
| `APPLICABLE` | Coincide con su propio sujeto y cumple el perfil | Puede entrar en un grupo; aún no tiene rival comparable |
| `COMPARABLE` | Al menos dos aplicables comparten la clave de comparación | Reordenar solo sus posiciones dentro del grupo |
| `CONFLICTING` | Afirmaciones incompatibles que el protocolo declarado no permite reconciliar | Conservar fuentes y usar fallback para la comparación afectada |

`INVALID` describe un error de lectura/validación, no la capacidad del modelo.
Queda como diagnóstico; no se convierte en evidencia ni exclusión del candidato.

Calidad local exige:

- Registro completo, exitoso, validado, split completo y métricas finitas coherentes
  con muestras. Un `full` autodeclarado no evita los validadores existentes.
- SHA256 exacto del artefacto del plan. Repo, familia o nombre no sustituyen el digest.
  El SHA identifica la representación evaluada, incluida su cuantización/tokenizer.
- Perfil de tarea/idioma explícito. Un nombre de suite desconocido no se clasifica
  por palabras del nombre ni por tags del modelo.
- Protocolo seleccionado y configuración efectiva de la evaluación verificables: contexto,
  prompts/template, few-shot, modo thinking, límites de generación/razonamiento,
  tools, seeds, evaluador, runtime/build y flags. Desconocido no equivale a default.
- Colocación y hardware de la evaluación documentados, con las condiciones
  requeridas por el grupo comparativo verificadas. V1 no asume invariancia entre
  GPUs; un simple nombre de hardware no confirma esas condiciones.

La aplicabilidad declara su alcance. `controlled_artifact` permite comparar
artefactos exactos bajo un protocolo común y afirmar, por ejemplo, «mejor IFEval
bajo el protocolo P, contexto 2048». Quality puede usar ese grupo aunque el plan
propuesto tenga contexto 4096, mostrando la diferencia de configuración.
`current_configuration` exige además correspondencia con el contexto, runtime,
hardware, colocación y condiciones efectivas que se atribuyen al plan actual.

Cambiar el contexto o la colocación propuestos no borra la evidencia controlada;
impide afirmar que se ha medido la nueva configuración. El alcance se conserva
en el resultado y en su explicación. Balanced v1 exige calidad aplicable a la
configuración de su medición de rendimiento; una diferencia en cualquiera de
las condiciones relevantes compartidas conserva el fallback de Balanced.
Los workloads propios de cada eje permanecen distintos: instrucciones evaluadas
para calidad y `tg128` para velocidad. No se exige igual prompt o presupuesto
entre ambos benchmarks. Se verifica su correspondencia en artefacto, runtime,
backend, colocación y demás condiciones compartidas declaradas por el perfil;
`--ctx-size` no aplicado por llama-bench no crea una igualdad o diferencia de
contexto. El contexto de calidad conserva su alcance explícito.

Publicaciones exigen el mismo alcance de tarea/métrica y protocolo declarado.
Para aplicar una cifra al plan se necesita revisión y representación explícitas
que correspondan a ese plan; un resultado BF16 del modelo base no evalúa un GGUF Q4.
El filtro de atribución existente sigue estricto. No se normalizan variantes ni
se completan condiciones para desbloquear el catálogo actual. Publisher e
independent permanecen distinguibles; una fuente local no borra otra publicada.

### 10.3 Clave de comparación de calidad

Dentro de un perfil, la clave reúne suite/config hash, dataset/revisión/split,
IDs de muestras, política de formato de prompts, few-shot, protocolo efectivo,
evaluador, runtime/build, flags y condiciones de hardware de la evaluación.
La comparación actual de HellaSwag conserva sus checks estrictos; las futuras
suites generativas necesitan su propio contrato auditado antes de habilitarse.

Cada artefacto conserva su tokenizer verificado. Se permiten tokenizers diferentes
entre modelos: no se exige igualdad de IDs de tokens ni de vocabularios.
Con prompts raw se exige el mismo texto por ejemplo y la misma política de BOS/EOS.
Para chat se fijan mensajes semánticos comunes y una regla versionada de aplicación
de la plantilla oficial de cada artefacto, registrando template/hash y texto final.
Ese modo solo se habilita tras comprobar la regla: no basta permitir cualquier
diferencia de prompt ni tratar raw y chat como equivalentes. Un límite común de
tokens se documenta como tal; no garantiza igual longitud de texto entre tokenizers.

La identidad del artefacto NO debe ser común entre rivales: cada evidencia debe
corresponder a su propio artefacto. Sus pesos, revisiones y cuantizaciones pueden
diferir porque precisamente se comparan esas representaciones; la explicación
lo dice y no convierte la diferencia en un efecto aislado de la familia.

No mezclar fuentes publicadas y locales en una misma clave v1. No mezclar
publicaciones de experimentos distintos solo porque coincida el benchmark.
Fuentes o sujetos con campos necesarios desconocidos quedan como referencias.
Si un sujeto tiene varios protocolos elegibles y no hay uno preseleccionado,
no escoger el favorable ni crear grupos solapados: queda en fallback.
Repetir idéntico contenido es idempotente; distintas ejecuciones se conservan.
Resultados diferentes en ejecuciones repetidas no son automáticamente conflicto.

El perfil v1 fija una ejecución por artefacto, vinculada a una campaña y seed
seleccionadas antes de observar scores. No se escoge retrospectivamente la última
o la mejor de varias ejecuciones. Si falta la selección previa necesaria, se
conservan las referencias y el fallback hasta fijar una campaña reproducible.

Un perfil que habilite repeticiones debe declarar de antemano cantidad y seeds
comunes, reglas de completitud y agregación. Se permite la media aritmética de
la misma métrica/split entre todas las repeticiones previstas, mostrando valores
individuales y dispersión. No se mezclan suites, protocolos o artefactos, ni se
cuenta el mismo split repetido como nuevas muestras independientes. Si falta una
repetición prevista, la campaña no autoriza esa comparación. Los registros fuente
siguen inmutables y el agregado es una proyección derivada con sus IDs.

### 10.4 Rendimiento aplicable

Perfil inicial de Fastest: microbenchmark individual `tg128`, acompañado por
`pp512`. Mayor throughput es mejor; cinco repeticiones y protocolo de warmup
versionado. El resultado describe esa carga, no latencia, batch de documentos,
contexto largo, TTFT, capacidad de servicio ni SLO del usuario.

Para ordenar: éxito, metodología soportada, artefacto con SHA exacto, misma
máquina verificable, runtime/build conocidos y carga/repeticiones iguales.
Se verifican comando y flags efectivos: backend, device, offload, threads,
batching y demás opciones relevantes del protocolo. Metadata heredada del plan
no prueba que una opción llegara al benchmark. `--ctx-size` heredado de
llama-bench se excluye como condición aplicada; no se inventa un contexto efectivo.

Cada observación debe corresponder a SU plan. Entre planes rivales pueden variar
artefacto, cuantización y offload: se compara el plan completo. V1 no compara
metodologías/runtime distintos como si midieran el mismo procedimiento.
Los helpers históricos de matching no bastan por sí solos: actualmente aceptan
ciertos matches de metadata y no comprueban todos estos campos efectivos.

Para varias ejecuciones de la misma configuración se elige la última válida
por fecha e ID, nunca la más rápida; se conservan IDs anteriores y dispersión.
Sin fecha/configuración suficientemente conocidas se mantiene el fallback.
Una medición individual no habilita ordenar por throughput multiusuario o batch:
en esos modos el orden sigue el fallback, con la medición como referencia.

**Auditoría offline del 2026-10-07.** `fastest_record_blockers` comprueba los
prerrequisitos del registro; `fastest_benchmark_blockers` añade correspondencia
con el plan, SHA exacto, dispositivo y workload individual. El matcher histórico
y el ranking activo siguen intactos. Estas funciones no autorizan comparabilidad:
los registros actuales no tienen un contrato auditado de parámetros efectivos.
Un comando coincidente o una duración de warmup no demuestra ese contrato.

Se leyeron los 14 registros existentes del store local, sin ejecutar modelos:

| Carencia | Registros |
|---|---:|
| Generación `tg64`, sin `tg128` | 9 |
| Build ausente de metadata, pero presente en el footer original | 8 |
| Metodología no declarada | 3 |
| Sin protocolo efectivo auditado de threads/batching/warmup/carga | 14 |

Las filas se solapan. Ninguno satisface todavía los prerrequisitos del perfil
nuevo; eso no invalida sus mediciones históricas ni cambia el fallback actual.
Esta auditoría solo revisa registros: no verificó su aplicabilidad a una búsqueda
o máquina actual. El informe local ignorado conserva IDs y bloqueos por registro
en `.codex-night/fastest-records-audit-20261007-03.json`.
Los 14 footers identifican el build; la comprobación lo lee sin completar metadata
persistida y rechaza footers ambiguos o contradictorios con el probe.

Se inspeccionó el productor y el código local de llama.cpp, commit
`689e227db485c6b33d061555e74034c93a867649`: la columna `backend` enumera backends
cargados, no acredita ejecución GPU; `dev`/`ngl` reportan configuración, no
buffers medidos. El parser actual conserva esos campos, pero no los parámetros
efectivos completos. No inferir defaults históricos del ejecutable instalado hoy.

Antes de habilitar Fastest se necesita captura estructurada del protocolo y
selección de la última ejecución **válida y aplicable**, con ID como desempate,
seguida de grupos comparables. El extractor actual del ranking usa el máximo
entre longitudes de generación; se conserva en este paso y no debe reutilizarse
para el perfil fijo `tg128`. No modificar ni rellenar registros antiguos.

**Revisión de la captura JSONL.** Los benchmarks nuevos piden `-oe jsonl` sin
cambiar la carga, y conservan tabla, footer y salida cruda. El protocolo es
opcional: los registros antiguos siguen con `protocol=None`. La captura mejora
la trazabilidad, pero **no resuelve todavía todas las condiciones efectivas**:
el constructor `test` del build auditado copia batch/microbatch y flash attention
de `cmd_params_instance`, sin leer el contexto creado. `llama-context.cpp`
puede limitar batch/microbatch al contexto y desactivar flash attention incluso
si se pidió `on`. No se aceptan esos valores como prueba de ejecución efectiva.
Warmup se registra como derivación del comando, no observación independiente;
su procedimiento requiere el build auditado. No se añade `-fa` automáticamente.
Los límites efectivos deben atribuirse a cada prueba `pp`/`tg`: los contextos
creados pueden diferir dentro de una misma ejecución. La unanimidad del JSONL
sobre el batch solicitado no resuelve esas diferencias.
JSONL malformado no debe impedir conservar una medición completada ni convertirse
en protocolo confiable por coerción de tipos. Los checks de build comparan hash
y número completos con el footer. Fastest continúa diagnóstico, sin cohortes ni
ordenación shadow habilitadas por estos campos solos.

### 10.5 Balanced, actualización y regresiones

Balanced usa la intersección de grupos comparables de calidad y rendimiento,
con los mismos sujetos/planes aplicables y calidad de alcance
`current_configuration` respecto a las condiciones del benchmark de velocidad.
Si solo se conoce un eje, no declara
Pareto ni cambia a Quality/Fastest silenciosamente. Mantiene sus posiciones base.
Con ambos ejes, aplica el Pareto ya definido, sin pesos ni bonus de procedencia.

Una búsqueda congela la versión de política y la instantánea de catálogo/records.
La siguiente búsqueda puede usar nuevos datos. Recompare crea un resultado nuevo;
no modifica búsquedas antiguas. Una nueva familia sin evidencia mantiene fallback.
Actualizar fuentes no exige tocar la policy; admitir una nueva suite o protocolo
sí exige revisión y versión, no un updater que descubra reglas automáticamente.

Regresiones mínimas antes de implementar shadow: SHA/revisión/variante incorrectos,
idioma o tarea no cubiertos, limited/fallo, protocolo/contexto/placement diferente,
fuente publicada sin precisión, runtime o máquina desconocidos, tg de otra longitud,
flags no aplicados, repeticiones incompatibles, conflicto y candidato aislado.
Sin un grupo comparable, IDs/orden/ScoreBreakdown deben conservar el baseline.
