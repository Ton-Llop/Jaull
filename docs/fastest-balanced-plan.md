# Fastest y Balanced: evidencia de rendimiento aplicable

**Estado: completada y activada el 2026-10-10** (commit `491b6fc`). Fastest y
Balanced ordenan con evidencia `llama_bench_v2` aplicable; ver [Estado](#estado).

Fase siguiente a `recommendation-vfinal-plan.md`. Al empezarla, Quality ya ordenaba
con calidad medida, mientras que Fastest y el eje de velocidad de Balanced seguían en
shadow porque ninguna medición de velocidad podía aplicarse. El objetivo era que un
benchmark nuevo pudiera superar esos bloqueos con rigor, no rebajar los bloqueos.

Pregunta que responde Fastest v1: entre los planes medidos con el mismo protocolo en
la misma máquina, ¿qué configuración genera más rápido? No: qué modelo es más rápido.

## Actualización acotada (2026-10-11): ruido entre ejecuciones

La campaña de tres rondas motivó sustituir "gana la última medición" por `shadow-v2`.
`assess_speed` agrega todas las llamadas válidas y distintas de la cohorte aplicable más
reciente: mediana de tg128, número de llamadas, rango observado e IDs de cada benchmark.
No hay ventana temporal ni exclusión por valor. Los benchmarks completos de intentos
interrumpidos también cuentan; la medición de 218,6 no se elimina por ser baja.

Con menos de tres llamadas el dato se muestra, pero no ordena. Revisión con Ton
(2026-10-11): cada pareja se compara directamente. La velocidad decide solo si los rangos
observados son disjuntos (tocarse cuenta como solape). En Balanced, la calidad se compara
con intervalos de Wilson al 95 % del IFEval: un plan va delante si ningún eje es
demostrablemente peor y uno es demostrablemente mejor. Velocidad solapada es un empate
operativo y la calidad claramente superior lo decide; si ningún eje se distingue, o es un
compromiso real (más rápido pero claramente peor), se conserva el orden base. No es
dominancia de Pareto estricta. Los solapes encadenados no se fusionan: A~B y B~C no
impiden que un par A–C disjunto decida. El orden final es el más cercano al base que
respeta cada par demostrable: el ganador sube al hueco del perdedor.

Limitación conocida: el rango mínimo–máximo solo crece; una ejecución atípica puede
impedir indefinidamente distinguir dos planes. Con 3–4 llamadas es una regla
conservadora y comprensible; un rango robusto queda como mejora posterior.

Evaluate (Speed y la parte de velocidad de Balanced) ejecuta las llamadas que faltan
hasta tres (0→3, 1→2, 2→1, 3→0; un plan con tres o más que se marque a mano recibe una
llamada extra), intercaladas entre planes y con 60 s de enfriamiento entre llamadas.
Cada llamada se guarda como su propio benchmark; si se cancela, se conservan las ya
guardadas y no se inventa ninguna.

La TUI y los informes muestran llamadas, rango e IDs; solo se marca un ganador de
velocidad si su rango está completamente por encima de los demás. El rango no es un
intervalo de confianza; tres llamadas son un mínimo conservador, no prueba estadística.
No cambia el productor `llama_bench_v2`, no se reescriben registros ni se calibra hardware.
Las decisiones y resultados de los apartados siguientes describen la versión inicial.

## Paso 0. Decisiones congeladas (2026-10-10)

| Decisión | Valor |
|---|---|
| Métrica primaria / secundaria | `tg128` tok/s / `pp512` tok/s; pp512 nunca sustituye a tg128 |
| Repeticiones | 5; se ordena por la media, empate = orden base |
| Flash attention | `-fa off`: lo único que se puede demostrar después. Más conservador que producción |
| Batching | `-b 512 -ub 512` explícitos |
| Threads | `-t` = núcleos físicos (`thread_policy=physical_cores`; 6 en el Ryzen 5 3600) |
| Caché KV | `-ctk f16 -ctv f16` |
| Carga | `-lm mmap` explícito (`-mmp` está obsoleto en el build actual) |
| Warmup | el de llama-bench, registrado como `ran` |
| Offload | el `-ngl` **del plan**, nunca `all` por defecto |
| Metodología | `llama_bench_v2`; los registros v1 quedan como referencia, sin reinterpretarse |

**Batching, con precisión.** llama.cpp recorta el batch al contexto de cada prueba
(tg128 tiene contexto 128, así que aplica 128, no 512). El recorte es determinista:
mismos `-p/-n/-b/-ub` implican el mismo batch aplicado en todos los modelos. Por eso
fijar los límites basta para comparar, aunque no se afirme el batch efectivo.

**Balanced, con honestidad.** Qwen2.5-0.5B (más rápido) frente a Qwen2.5-1.5B (más
calidad) es un trade-off: los dos quedan en el frente 1 y conservan el orden base. La
dominancia se demuestra con un test sintético; el caso real demuestra el trade-off. No
se fabrica un movimiento real.

## Pasos

1. **`llama_bench_v2`**: comando explícito con los valores de arriba; el JSONL de
   llama-bench debe devolver exactamente lo pedido o la medición falla. El registro
   guarda la política de threads. Se aceptan v1 y v2; se prefiere v2. Sin cambios en el
   ranking.
2. **Aplicabilidad**: los bloqueos de batching y flash attention se levantan solo si el
   registro es v2 y lo demuestra. La velocidad entra en `production_evidence` con su
   cohorte: máquina, build, protocolo, backend, dispositivo, batch, threads, KV, FA,
   warmup y carga. El SHA, el modelo, la cuantización y el offload no forman parte de
   la cohorte, pero cada registro debe coincidir con el artefacto y el offload de su plan.
3. **Campaña mínima en la RTX 2060**: los mismos GGUF que ya tienen IFEval (Qwen2.5-0.5B y
   1.5B Q5_K_M, LFM2.5-1.2B Q5_K_M), con el offload de su plan. Guardar los registros
   antes de mirar quién gana; no cambiar el protocolo después.
4. **Fastest en shadow**: misma regla que Quality, ordenando por tg128 descendente.
   Matriz de revisión: sin evidencia, una sola medida, dos comparables, protocolo o
   hardware distinto, SHA u offload distinto, un solo eje para Balanced, ambos ejes.
5. **Balanced**: Pareto con calidad y velocidad del **mismo SHA** del plan. IFEval y tg128
   miden cosas distintas; lo que tiene que coincidir es la configuración que caracterizan.
6. **Activación**, con aprobación de Ton: `production_evidence` expone la velocidad
   validada; el motor shadow ya contiene el resto.

## Estado

- **Paso 1 (2026-10-10).** `llama_bench_v2` en el runner. Probado con el binario real en la
  RTX 2060 (SmolLM2-135M): el JSONL devuelve exactamente los ajustes pedidos, sin
  discrepancias. Una medición que no siga la v2 se conserva como referencia.
- **Paso 2 (2026-10-10).** `ApplicableSpeed` y el estado `measured_plan`. Para v2, los
  bloqueos fijos de batching y flash attention se sustituyen por la comprobación real (ajustes
  devueltos, flags en argv y build). La v1 sigue solo como referencia. La cohorte digiere
  máquina, metodología, ajustes, backend, dispositivo y carga de trabajo. `production_evidence`
  entrega la velocidad al shadow. Con núcleos desconocidos nunca se confirma la v2. Aplica la
  referencia sin bloqueos más reciente; promediar repeticiones queda para cuando haya una regla
  de varianza.
- **Dispositivo implícito (paso 2).** Los planes de Search no llevan `--device`, y el bloqueo
  que exigía que coincidiera impedía que cualquier velocidad se aplicara. Ahora, con una sola
  GPU en la máquina del registro y el backend coincidente, el dispositivo queda determinado; con
  varias GPU sigue desconocido. Es la misma regla que ya usa la selección de IFEval.
- **Paso 3 (2026-10-10).** `scripts/fastest_campaign.py` reconstruye los planes de los GGUF con
  IFEval (búsqueda fijada a esos artefactos, hardware real), vuelve a calcular el SHA en disco y
  mide con el runtime y offload de cada plan. RTX 2060, `-ngl -1`, 5 repeticiones. Los tres
  quedan `measured_plan`:

  | GGUF Q5_K_M | tg128 | IFEval estricto |
  |---|---|---|
  | Qwen2.5-0.5B | 271,7 tok/s | 25,9 % |
  | LFM2.5-1.2B | 235,1 tok/s | 80,2 % |
  | Qwen2.5-1.5B | 164,5 tok/s | 41,0 % |

- **Paso 4, shadow con datos reales.** Speed y Balanced forman su primer grupo (el par Qwen).
  Speed no mueve porque el base ya pone delante al más rápido. Balanced reconoce el trade-off
  (0.5B más rápido, 1.5B más calidad), con los dos en el frente 1 y el orden base. LFM2.5 queda
  en otro estrato por la licencia en las tres prioridades, aunque domina al Qwen 1.5B en ambos
  ejes. Los movimientos de Fastest y la dominancia de Balanced están cubiertos por los tests
  sintéticos del shadow.
- **Paso 6, activación (2026-10-10), aprobada por Ton.** `apply_measured_policy` sustituye a
  `apply_quality_policy`: Quality, Fastest y Balanced ordenan con su evidencia aplicable;
  Memory no cambia. `assess_speed` es la única autoridad: `_throughput_rank` y
  `performance_evidence` leen solo su velocidad aplicable, nunca registros crudos. Los
  benchmarks v1 siguen visibles como referencia y ya no ordenan. El informe añade
  `speed_policy` y `balanced_policy` junto a `quality_policy`, y los snapshots se han
  regenerado (el informe canónico, Balanced, gana su bloque sin mediciones). Regresión:
  solo v1 ordena igual que sin benchmark; con v1 + v2 aplicable solo cuenta v2; una v2
  bloqueada no cuenta. Activación: Fastest mueve con dos medidas comparables; sin evidencia
  o con una aislada, no; Balanced mueve una dominancia, conserva un trade-off y necesita
  los dos ejes; Memory no cambia. Fuera de esta decisión: la memoria medida y la confianza
  siguen usando el benchmark emparejado, porque no son velocidad.

- **Revisión de Codex tras la activación (2026-10-10), tres arreglos.**
  1. La velocidad medida se colaba en el orden base: `_throughput_rank` (Speed) y
     `performance_evidence` (cola de todas las claves y firma de diversidad) leían la velocidad
     aplicable, de modo que una sola medición reordenaba antes de formar cohortes, incluso en
     Memory. Ninguno de los dos es ya un eje: la velocidad solo ordena dentro de los grupos de
     la política. Regresión desde el motor completo (`recommend`): una medición aislada no
     cambia nada en Speed, Balanced ni Memory; falla si se reintroduce el eje.
  2. La cohorte de velocidad incluye la versión del driver y de CUDA de cada GPU, y un registro
     con otro driver o CUDA que el de la máquina del plan queda bloqueado. La huella general de
     la máquina no cambia.
  3. El dispositivo implícito ya no se acepta por haber una GPU: el banner de llama.cpp del
     propio registro (`Device <i>: <nombre>`) debe nombrar al acelerador que eligió el plan,
     una sola vez. `CUDA1`, otra GPU o dos GPU idénticas quedan como referencia.
  Las tres mediciones de la 2060 siguen aplicando con estas reglas.

- **Evaluate candidates mide lo que pide la prioridad (2026-10-10).** El selector de
  la pantalla de evaluación arranca en la suite de la búsqueda: Quality → IFEval,
  Fastest → Speed (`benchmark_plan_speed`: `llama_bench_v2` con la carga Fastest sobre el
  offload y el dispositivo del propio plan, sin Docker), Balanced → IFEval y después
  velocidad sobre los mismos GGUF. Un candidato viene marcado solo si a esa suite le falta
  un resultado aplicable. Dos arreglos de emparejamiento salieron de la prueba real: la RAM
  total de WSL varía unos KiB entre arranques, así que la huella de evidencia la redondea a
  256 MiB (los manifiestos de casos conservan la cifra exacta); y un plan de Search dice
  `main` mientras la ejecución fija el commit, así que el mismo repo, fichero y SHA256
  cuentan como el mismo artefacto. Las tres mediciones de la 2060 vuelven a ser
  `measured_plan`. Pendiente: Qwen2.5-0.5B dio 271,7 y 245,0 tok/s en dos sesiones (~10 %);
  hoy aplica la más reciente, y una regla de agregación queda como trabajo futuro.
  Revisión de Codex, tres arreglos: Recompare recarga también la velocidad guardada (antes
  solo la calidad) sin tocar el pool; un fallo al leer o presentar un benchmark nuevo se
  informa y libera la pantalla; Balanced ejecuta solo el eje que le falta a cada candidato,
  y no repite IFEval donde la calidad ya aplica o está en conflicto.

## Cierre

- Fastest identifica el más rápido entre dos artefactos medidos y explica la evidencia.
- Balanced identifica dominancia o trade-off entre dos artefactos con los dos ejes.
- Sin evidencia suficiente, no inventa comparaciones y conserva el orden.
- Cada movimiento se rastrea hasta SHA, benchmark, hardware, build, protocolo, métrica y valor.

## Fuera de alcance

Predicción de tok/s, calibración del HFA, multiusuario o batch, comparar GPUs o
extrapolar entre máquinas, benchmark automático durante Search, descargas automáticas,
scores globales, autotuning de flags y Fastest entre llama.cpp y Transformers.

## Después: cobertura

Cuando las tres políticas estén activas, el cuello de botella será que los artefactos
medidos casi nunca están en el pool de Search. Afecta a las tres por igual y se resuelve
una sola vez, no política por política.
