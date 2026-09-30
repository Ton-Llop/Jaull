# Comparación de pruebas RTX 4060 y referencia RTX 2060

Revisión offline del 30 de septiembre de 2026. Se han leído records, comandos,
logs y manifests existentes; no se ha ejecutado otra inferencia ni benchmark.
Este documento analiza resultados, no modifica evidencia histórica ni calibra
el estimador. Los `+/-` de las tablas representan la desviación estándar reportada
por llama-bench, no un intervalo de confianza.

## Resumen

- La RTX 4060 completa Validate y Benchmark con el plan corregido de full offload.
- En la 4060, full offload obtiene 52,73 tok/s de generación frente a 44,22 con
  28/29 unidades: diferencia descriptiva de +19,24 %.
- La referencia 2060 obtiene 60,85 tok/s en full offload. La 4060 no mejora esa
  generación en estas pruebas, aunque sí aumenta el prefill. No son controles
  equivalentes de GPU: cambian runtime, OS, CPU y condiciones de ejecución.
- La observación de memoria útil procede de buffers reportados por llama.cpp.
  NVML por proceso sigue sin estar disponible; no se sustituyó por memoria global.
- El bundle final permite transportar Validate y Benchmark con integridad
  verificable. No certifica SLOs, capacidad multiusuario ni toda la VRAM del proceso.

## Baseline y fuentes

Modelo común: `bartowski/Qwen2.5-7B-Instruct-GGUF`, archivo
`Qwen2.5-7B-Instruct-Q4_K_M.gguf`, SHA-256
`65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423`.
La campaña 4060 fija la revision `8911e8a47f92bac19d6f5c64a2e2095bd2f7d031`.
Para la referencia 2060 se verifica el mismo SHA: no se presupone igualdad de
revision de repositorio si el documento histórico no la declara.

| Fuente | GPU / entorno | Runtime | Uso en esta revisión |
|---|---|---|---|
| [Control B001-R6](b001-r6-tensor-policy.md) | RTX 2060 6 GiB, Ryzen 5 3600, WSL2 | b10357 / `689e227db` | pp512/tg128, tres repeticiones |
| [B001-R10](b001-r10-experiment-record.md) | RTX 2060, WSL2 | b10357 | Persistencia de buffers, no throughput |
| [Campaña 4060 inicial](rtx4060-b001-r10-investigation.md) | RTX 4060 8 GiB, Windows | b10357 | Fallos y límites históricos |
| [4060, 28/29](../../validation/rtx4060-cnight-20260930-112007/bundle/) | RTX 4060, Windows | b11258 / `ba0ba54d9` | Validate y benchmark completos |
| [4060, plan nuevo](../../validation/rtx4060-fresh-validate-20260930-135047/README.md) | RTX 4060, Windows | b11258 | Validate sin traducción incorrecta de batch |
| [4060, bundle final](../../validation/rtx4060-full-offload-benchmark-20260930-164332/README.md) | RTX 4060, Windows | b11258 | Validate anterior y benchmark full offload |

## Evolución de la campaña 4060

1. **Campaña inicial b10357:** Validate terminó, pero los benchmarks no completaron
   generación. Uno terminó con exit 0 y filas de prefill solamente; otro agotó
   900 segundos. El `success=true` histórico del primero no lo convierte en
   benchmark completo. No se usa ninguno para comparar generación.
2. **b11258, plan 28/29:** Validate terminó en 5,60 s; el comando omitió el
   `--batch-size 1` que el plan declaraba. El benchmark pp512/tg128 sí terminó.
   Esa ejecución no probaba un batch efectivo de un token.
3. **Aplicación del flag explícito:** el Validate
   `exp-cf9bda77-e4cb-4f74-b48d-5a2cb4b1a39e` falló en 6,91 s. El log confirma
   `n_batch=1`, `n_ubatch=1` y `failed to add token ... n_tokens = 1`.
   Es evidencia del fallo con esa configuración; no demuestra OOM. El record
   conserva un trace vacío y carece de allocation estructurada; el diagnóstico
   depende del log original, no de una reconstrucción del record.
4. **Corrección semántica, commit `1d12f63`:** las secuencias que multiplican KV
   dejaron de convertirse en el límite de tokens por llamada del runtime.
   No cambiaron las fórmulas de memoria. Los planes históricos explícitos siguen
   respetándose, no se reescriben para convertir fallos en éxitos.
5. **Plan nuevo:** `exp-f05586ff-c90e-4239-821d-5baaf550bd2a` terminó en
   5,263 s, exit 0, CUDA confirmado, contexto efectivo 4096, 29/29 unidades,
   `n_batch=2048` y `n_ubatch=512`. No llevaba flags de batch de tokens.
6. **Benchmark final:** `bench-fe91bac7-a964-4bba-8e40-8646a8bd8e3c`, source
   `55b31d7`, terminó en 10,929 s mediante el runner existente. Incluye las dos
   filas solicitadas y tres repeticiones. Usa el mismo artefacto y plan full offload.

El paso 3 al 5 cambia tanto batch efectivo como offload (28/29 a 29/29).
Demuestra que el flujo corregido funciona, pero no aísla experimentalmente el
efecto del batch. Tampoco identifica la causa del stall inicial en b10357:
el éxito con otro build no es un control que cambie una sola variable.

## Throughput

Todos los controles siguientes usan pp512/tg128 y tres repeticiones.

| GPU / control | Unidades pedidas | Build | pp512 tok/s | tg128 tok/s |
|---|---:|---|---:|---:|
| 2060 B001-R6 | 24 | b10357 | 1159,45 +/- 39,76 | 19,41 +/- 3,42 |
| 2060 B001-R6 | 25 | b10357 | 1163,28 +/- 100,42 | 20,20 +/- 4,93 |
| 2060 B001-R6 | 28 | b10357 | 1472,07 +/- 68,58 | 45,22 +/- 0,62 |
| 2060 B001-R6 | -1 (full) | b10357 | 1558,16 +/- 44,68 | 60,85 +/- 0,17 |
| 4060, `bench-a07438ca` | 28 | b11258 | 2408,90 +/- 49,77 | 44,22 +/- 0,60 |
| 4060, `bench-fe91bac7` | -1 (full) | b11258 | 2555,13 +/- 37,53 | 52,73 +/- 0,04 |

Fuente numérica 2060: [mediciones B001-R6](../../validation/bundles/b001-r6-tensor-policy/).
Las dos fuentes 4060 son los BenchmarkRecords de sus bundles enlazados arriba.

### Comparación dentro de la 4060

Full frente a 28 unidades: prefill +6,07 %, generación +19,24 %, calculados como
`100 * (media_full / media_28 - 1)`. Son mediciones en momentos distintos:
cambian memoria disponible y ubicación del runtime; no hay campaña aleatorizada
ni barrido repetido. La diferencia observada es compatible con la importancia
del placement, pero no cuantifica exclusivamente el coste de una unidad en CPU.
La desviación baja de tg128 full describe esas tres repeticiones, no estabilidad
bajo otros workloads.

### Comparación entre máquinas

En full offload, la 4060 tiene +63,98 % de prefill y -13,34 % de generación frente
al control 2060 citado. Son diferencias entre **configuraciones observadas**, no
una estimación del efecto causal de cambiar GPU. OS, CPU, build, compilación y
defaults internos no están igualados. El mismo SHA, tamaños pp/tg y número de
repeticiones permiten describir los resultados, no atribuirlos solo al hardware.
No se infiere una causa de la menor generación en la 4060 ni se modifica ranking.

`--ctx-size 4096` no aparece en los comandos de llama-bench. Es procedencia del
plan, no un contexto aplicado en el benchmark. La duración del proceso Validate
no es TTFT ni se compara con tok/s. No hay prueba de concurrencia real ni SLO.

## Memoria del Validate final

La comparación guardada y recalculada coinciden. Valores en MiB (2^20 bytes):

| Componente GPU | Predicción física | Buffer reportado | Diferencia relativa | Interpretación |
|---|---:|---:|---:|---|
| Pesos | 4466,13 | 4168,09 | -6,67 % | Comparación directa de componentes |
| KV | 224,00 | 224,00 | 0,00 % | Coincidencia con el contexto efectivo |
| Overhead / compute | 958,61 | 136,01 | -85,81 % | Proxy, no equivalencia completa |
| Total | 5648,74 | 4528,10 | -19,84 % | Solo contra buffers enumerados por el runtime |

Fuente canónica: [ExperimentRecord final](../../validation/rtx4060-full-offload-benchmark-20260930-164332/bundle/records/experiment.json).
El total reportado es 4,42 GiB, frente a 5,52 GiB físicos previstos. La comparación
usa bytes físicos, no el budget con reserve y safety margin. Overhead también
cubre conceptos que compute no enumera, como contexto CUDA y allocations del
allocator. Por ello el -85,81 % no mide directamente el error de esa heurística.
No justifica reducir overhead, reserve ni márgenes.

`peak_vram_bytes=null`, `source=runtime_reported_allocation` y
`driver_confirmed=false` permanecen separados. La RAM RSS del Validate fue
4,51 GiB; su comparación es `methodologically_unavailable` por mmap. En el
benchmark fue 4,50 GiB, también sin NVML por proceso. No se convierten esos RSS
en una medición de la parte host del placement ni se inventa allocation
estructurada para el BenchmarkRecord.

En el Validate 4060 de 28/29 y en B001-R10 2060, los gates de comparación de VRAM
no permiten un porcentaje. Se conserva ese resultado: no se recalcula un error
numérico solo porque existen buffers observados. El Validate 2060 a 14 unidades
tampoco es un control de memoria equivalente al full offload 4060.

## Integridad y limitaciones de los bundles

Revalidación offline con el código actual:

| Entrega | Resultado | Límite |
|---|---|---|
| 4060 inicial, `bundle-final` | Error de tamaño en `case.json` | Transporte alteró bytes; no se reparó el manifest |
| 4060 b11258, 28/29 | `valid` | Comparación de memoria no disponible |
| 4060 Validate nuevo, sin benchmark | `valid` | Solo ejecución controlada |
| 4060 Validate + benchmark full | `valid` | Dos warnings de RAM/VRAM libre distinta |
| 2060 B001-R4 full / launch-policy | `partial` | Backend del experimento no observado en el record histórico |

El bundle final incluye exactamente un Validate y un benchmark. Sus records
coinciden con los originales; los raw outputs coinciden con los sidecars, y todos
los archivos declarados pasan sus hashes. No se cambiaron manifests históricos.

En el preflight del benchmark faltaba `llama-cli.exe`; su recuperación fue
denegada por Windows. Los otros 53 EXE/DLL coincidían con los hashes esperados,
incluidos llama-bench y sus DLLs. Esto no invalida el benchmark, pero la instalación
no está completa para nuevos Run/Validate. No se atribuye el bloqueo a Defender,
OneDrive o permisos sin evidencia de esa causa.

Los logs versionados contienen rutas locales, identificadores y salida del modelo.
La limpieza retiró copias externas exactas, no evidencia única ni archivos del
bundle final. Su validez es de integridad y consistencia, no de calibración general.

## Conclusiones y siguiente paso

La campaña actual queda cerrada como prueba real de ejecución y throughput en
la RTX 4060: plan recién generado, Validate exitoso, benchmark completo y entrega
portable validada. También protege la distinción entre secuencias KV y batch de
tokens. No cierra la investigación causal del stall antiguo, una comparación
controlada 2060/4060 ni la incertidumbre de toda la VRAM atribuida por el driver.

El siguiente experimento útil sería igualar build y metodología entre máquinas,
o medir en Linux NVIDIA con NVML por proceso disponible para separar buffers y
allocations no reportadas. No hace falta repetir la 4060 para conservar el resultado
actual; sí se necesitan nuevas mediciones para esas preguntas distintas. No se
cambian fórmulas, constantes ni recomendaciones basándose en esta campaña.

Para verificar la entrega final desde la raíz del repositorio:

```sh
UV_CACHE_DIR=/tmp/uv-cache uv run --python 3.12 jaull experiments case bundle validate validation/rtx4060-full-offload-benchmark-20260930-164332/bundle --json
```
