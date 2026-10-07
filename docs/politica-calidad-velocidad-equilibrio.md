# Política de Quality, Fastest y Balanced

Borrador para revisión. **Las reglas nuevas de ordenación de este documento no
están implementadas.** Jaull sí almacena y muestra evidencia diagnóstica de
calidad, pero el ranking actual no la usa para ordenar. Este texto existe para
decidir si debe hacerlo y bajo qué reglas, antes de escribir código.

## 1. Qué decide este documento

Qué puede y qué no puede hacer la evidencia de calidad dentro del ranking, y
cómo se comportan las tres preferencias que el usuario elige en el wizard:
*Best quality*, *Fast responses* y *Balanced*.

**Qué no decide**: cómo se produce la evidencia (eso es el protocolo del
piloto), ni qué modelos entran en el catálogo (eso es trabajo de curación), ni
la estimación de memoria. El HFA y el motor de descubrimiento no se tocan.

## 2. Principios

**El ranking no tiene puntuación global y no la va a tener.** `score_all()` se
eliminó a propósito: el orden es lexicográfico por ejes visibles, y cada eje
dice de dónde sale. Cualquier regla de este documento que necesite sumar
calidad y velocidad en un número está mal planteada por construcción.

**La ausencia de evidencia no es mala calidad.** Es la frase que ya emite
`attach_capability_evidence` y es la regla más importante de todas. Un modelo
sin evaluación no es peor que uno evaluado: es desconocido. Penalizar la
ausencia convertiría el ranking en una medida de cuánto hemos mirado nosotros,
no de qué sirve al usuario.

**Un artefacto no es un modelo.** Es la tesis del proyecto y aquí se aplica
igual que en memoria: lo que un publisher midió es un modelo; lo que el usuario
va a ejecutar es un GGUF concreto, cuantizado por un tercero. Son afirmaciones
distintas y no se mezclan.

**Medir poco no es medir.** Un smoke de 3 ejemplos y una tirada limitada de 100
no son veredictos de calidad, por mucho que los hayamos producido nosotros.

## 3. Los tres grados de evidencia

| Grado | Qué es | Qué autoriza |
|---|---|---|
| **Medida** | Registro del piloto, `classification: full`, identidad exacta | Ordenar |
| **Publicada** | Entrada de catálogo sobre el modelo, con linaje confirmado | Mostrar |
| **Ninguna** | Solo prior de tamaño | Nada |

La separación entre las dos primeras no es de calidad del dato, es de **sujeto**.

Una medida publicada de MMLU puede ser mucho más informativa que un HellaSwag
nuestro de 100 muestras. Pero la publicada describe el modelo base y la nuestra
describe el artefacto exacto que se va a ejecutar. Por eso **coexisten**: la
medida no sustituye a la publicada ni al revés, y ninguna hereda el alcance de
la otra.

Las 11 entradas del catálogo son todas `publisher_reported` y ninguna declara
revisión ni precisión evaluada. Eso las deja en "mostrar", no en "ordenar",
independientemente de lo buenas que sean sus cifras.

## 4. Quality

**Decisión: Quality ordena por evidencia medida, acotada al suite, y solo entre
planes que tengan evidencia comparable.**

Tres condiciones, todas necesarias:

**Mismo suite.** "Mejor" no existe en abstracto. Un modelo mejor en HellaSwag
puede ser peor programando. Si el usuario pidió código y la única evidencia es
HellaSwag, la respuesta correcta no es ordenar por HellaSwag: es decir que no
hay evidencia para ese caso. El wizard ya pregunta el caso de uso, así que el
suite debe casar con él o la afirmación se acota explícitamente.

**Mismo protocolo.** `quality_comparison.py` ya retira las métricas si no
coinciden suite, dataset, muestras, evaluador, runtime y protocolo. Esa misma
puerta aplica aquí.

**Misma colocación.** La sonda de reparto lo midió: el mismo artefacto en CPU y
en GPU no coincidió en **ninguno** de los 146 logprobs, con hasta 0,88 nats de
diferencia por continuación. Dos modelos medidos bajo repartos distintos no se
comparan entre sí.

Si falta cualquiera de las tres, Quality **no ordena**: cae al criterio actual y
lo dice.

## 5. Fastest

**Decisión: Fastest ordena por rendimiento medido, nunca estimado.**

El rendimiento medido ya participa en el orden de *Fastest*: primero se ordena
por ejecutabilidad y después por throughput medido. `performance_evidence` se
muestra también como criterio explicativo. La decisión pendiente no es añadir
throughput al orden, sino acordar si ese orden actual representa bien
*Fastest*, especialmente cuando falta una medición y entran los desempates
existentes.

Sin medición, Fastest no inventa un orden a partir del tamaño. Un modelo más
pequeño suele ser más rápido, pero "suele" no es una medición, y el proyecto ya
decidió que las estimaciones se etiquetan como tales.

## 6. Balanced

**Decisión: Balanced es una puerta, no un peso.**

> Ordena por calidad entre los planes cuya velocidad medida supera el
> `min_generation_tps` del workload.

La alternativa —`w1 × calidad + w2 × velocidad`— reintroduce por la puerta de
atrás el score global que se eliminó, y además con pesos que nadie puede
justificar. ¿Por qué 0,6 y 0,4 y no 0,7 y 0,3? No hay respuesta, y una cifra sin
respuesta no debería decidir qué modelo usa alguien.

La puerta usaría un número que el usuario proporciona. `min_generation_tps`
existe en `WorkloadProfile` y `UserRequirements`, pero el wizard no lo pregunta
y la normalización actual no lo rellena; el ranking tampoco lo consume. Antes
de usarlo en *Balanced* habría que decidir cómo lo introduce el usuario y cómo
se conserva en el flujo de requisitos.

Con el ejemplo que manejábamos y un mínimo de 30 tok/s: C queda fuera por lento,
y entre A y B gana A por calidad. Sin inventar pesos.

**Si el usuario no fijó un mínimo**, Balanced no tiene puerta que aplicar y se
comporta como Quality, diciéndolo.

## 7. La ausencia, en detalle

Un plan sin evidencia de calidad **no baja** en el ranking. Mantiene su posición
por los ejes que sí tiene —ajuste de memoria, disponibilidad de runtime,
licencia, idioma— y su eje de calidad queda en `UNKNOWN`.

Esto importa más de lo que parece. La auditoría de 12 candidatos reales dio:

| | |
|---|---:|
| Sin entrada de catálogo | 8 |
| Linaje no confirmado | 3 |
| Candidato con entrada publicada atribuible | 1 |

Es decir: **con los datos de hoy, 11 de cada 12 planes no tendrían nada que
mostrar en este eje.** Una política que penalizara la ausencia reordenaría casi
toda la lista en función de nuestra cobertura de catálogo, que es una propiedad
nuestra y no del modelo.

## 8. Cuándo dos planes son indistinguibles

**Decisión: una diferencia menor que la incertidumbre no es una diferencia, y el
ranking debe poder decirlo.**

Como orientación, con una aproximación binomial independiente hacen falta del
orden de 1.250 muestras por modelo para que dos intervalos de Wilson al 95 %,
con accuracies de 0,70 y 0,75, dejen de solaparse. Esto no es un umbral universal
ni una prueba de diferencia: los intervalos separados son un criterio
conservador. Como ambos modelos responden a los mismos sample IDs, la
comparación debería considerar esa relación pareada y fijar un método de
incertidumbre antes de decidir cuándo son indistinguibles.

La comparación diagnóstica ya informa la diferencia pareada por muestra y, para
una evaluación limitada con variación observada, un intervalo bootstrap
percentil del 95 %. Ese intervalo es exploratorio; no es un criterio de
significancia ni una regla para ordenar modelos. Un split completo da la
puntuación exacta de ese benchmark y no necesita un intervalo de muestreo. En
ambos casos, el resultado no demuestra capacidad general.

Que dos intervalos se solapen, por sí solo, no demuestra igualdad. Cualquier
regla futura para decir **"no se distinguen"** debe quedar justificada por el
método y el protocolo.

Es el mismo criterio que la capa de memoria ya aplica cuando dice
`comparison unavailable`.

## 9. Lo que queda fuera

- **Concurrencia.** El experimento de 1/2/4 usuarios está medido y guardado, y
  no es prioridad de producto. El caso real es un usuario interactivo.
- **Combinar suites en un índice.** Promediar HellaSwag con MMLU produce un
  número que no significa nada. Si hay varios suites, se muestran varios.
- **Que la evidencia publicada ordene.** Hasta que una entrada declare revisión
  y precisión evaluadas, es referencia, no criterio.
- **Calibrar el prior de tamaño con las mediciones.** Mezclaría una heurística
  con una medición dentro del mismo número.

## 10. Qué haría falta para implementarlo

Sin escribir nada todavía, el trabajo que esta política implica:

1. Un eje `quality_evidence: AssessmentLevel` en `PlanAssessment`, en paralelo a
   `performance_evidence` y `resource_evidence`, que ya existen.
2. Que `ranking_criteria` consulte ese eje para la preferencia *Best quality*,
   respetando las tres condiciones del punto 4.
3. Que `min_generation_tps` llegue al ranking como puerta para *Balanced*.
4. Validar el intervalo bootstrap diagnóstico antes de usarlo en el ranking.
   Las respuestas por muestra ya están en el registro; no se persiste el
   intervalo.

Los registros ya conservan resultados por muestra, pero no un intervalo
precalculado. Puede derivarse de esos datos cuando se defina el método; no hace
falta añadir un campo persistido antes de tomar esa decisión.

Hay otra decisión previa al código: "ordenar solo cuando dos planes sean
comparables" define relaciones por pares, no necesariamente un orden total
transitivo. No se debe implementar como comparador por pares para `sort()`. Hay
que acordar cómo formar grupos comparables o cómo usar esa evidencia sin producir
órdenes contradictorios cuando una cadena de planes tiene protocolos distintos.

---

**Estado**: borrador para revisión; las reglas de ordenación propuestas no
están implementadas. Los números citados vienen de
`docs/quality-evaluation-pilot.md` (sonda de reparto), una auditoría local de 12
candidatos guardada en `.codex-night/lineage-audit-20261005.json` y
`src/jaull/recommendation/capability_catalog.json` (11 entradas). La auditoría
local no viaja en Git; su muestra no demuestra cobertura general y sus cifras
no son reproducibles por otra persona hasta conservar un informe revisable.
