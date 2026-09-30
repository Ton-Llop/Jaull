> Publication note (2026-09-30): this is an anonymized derivative. Statements below about original bytes/hashes describe collection-time evidence, not this public copy. See validation/public-anonymization.json and the campaign PUBLICATION.md.

# RTX 4060 — Benchmark full offload junto al Validate existente

Benchmark real mediante `AdvisorService.run_benchmark` y el `LlamaBenchRunner` existente. No se repitió Validate ni se ejecutaron configuraciones alternativas. No se modificaron código, calibraciones, records, cases o manifests anteriores.

## Resultados

`success=true`, exit `0`, metodología `llama_bench_v1`. Duración del proceso: 10.928952799997205 s; no es TTFT ni duración individual de una petición.

- pp512: 2555.13 ± 37.53 tokens/s; repetitions=3
- tg128: 52.73 ± 0.04 tokens/s; repetitions=3

Ambas filas requeridas están presentes: pp512 y tg128, tres repeticiones cada una. Backend observado por las filas: CUDA, dispositivo CUDA0, ngl=-1. No se comparó throughput con los 5.263 segundos de Validate.

## Baseline, runtime y hardware

Repo `bartowski/Qwen2.5-7B-Instruct-GGUF`; revision `8911e8a47f92bac19d6f5c64a2e2095bd2f7d031`; archivo `Qwen2.5-7B-Instruct-Q4_K_M.gguf`; SHA recalculado correcto `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423`.

Windows nativo, RTX 4060, driver 616.92, llama.cpp b11258 / ba0ba54d9, CUDA0. Python 3.12.14, entorno .venv/rtx4060-py312; uv sync --locked --python 3.12 --check confirmó que no hacían falta cambios. Source commit del Benchmark: `55b31d7a2957345a89576d19a14eb30e73d80eac`; del Validate: `1d12f6306bfcac2cf13786945bf2b4f22582184c`. El hardware estable coincide, incluidas identidad GPU/UUID, CPU, OS, capacidad y driver.

Se comprobaron los 54 hashes esperados contra la evidencia previa. 53 archivos presentes coincidieron exactamente; llama-cli.exe está ausente. Se informó antes del Benchmark. El ZIP oficial previo está disponible y su hash coincidió al intentar recuperar únicamente el launcher; Windows denegó su creación con PermissionError. No se cambiaron permisos ni controles de seguridad y no se atribuye una causa definitiva a la ausencia. No hubo descarga nueva ni instalación global.

Esta diferencia se mantiene en bundle/evidence/preflight.json, runtime-mismatch.json y bundle/evidence/runtime-hashes.json. El Benchmark no usa llama-cli.exe: usa llama-bench.exe y sus DLLs, todos coincidentes con la evidencia guardada. Los probes de llama-bench confirmaron el build y el backend CUDA antes de ejecutar. No se afirmó que estén presentes los 54 archivos.

Disponibilidad observada: RAM 21055684608 → 20171747328 bytes; VRAM 6807072768 → 7450460160 bytes. Estas diferencias se informaron antes de ejecutar y quedan como warnings del case/bundle.

## Comando efectivo y límites

Comando lanzado por el flujo existente de Jaull, conservado en BenchmarkRecord y run-index.json:
```text
C:\Users\USER\OneDrive\Desktop\Jaull\.venv\rtx4060-runtime-b11258-recovery-20260930-121948\bin\llama-bench.exe -m C:\Users\USER\AppData\Local\jaull\models\bartowski\Qwen2.5-7B-Instruct-GGUF\Qwen2.5-7B-Instruct-Q4_K_M.gguf -dev CUDA0 -ngl -1 -p 512 -n 128 -r 3
```

Contexto 4096 y concurrencia 1 son procedencia del plan. El comando no pasa --ctx-size; no se afirma que llama-bench aplique ese contexto. Tampoco pasa --batch-size ni transforma batch_size=1 del estimador en batch de tokens. Se conserva el comportamiento por defecto de llama-bench sin atribuirle un valor no observado.

## Records, case y bundle nuevos

Validate usado sin cambios: `exp-f05586ff-c90e-4239-821d-5baaf550bd2a`. Su SHA de envelope al terminar sigue siendo `40e1b0b7fac3553688d70c5690f15eaabe599a55f841018839934294b07c0d36`, igual al README previo. Su sidecar real se conserva en bundle/evidence/logs/; los bytes coinciden con el original.

Benchmark: `bench-fe91bac7-a964-4bba-8e40-8646a8bd8e3c`. Original: `C:\Users\USER\AppData\Local\jaull\benchmarks\bench-fe91bac7-a964-4bba-8e40-8646a8bd8e3c.json`. Copia local: `C:\Users\USER\OneDrive\Desktop\Jaull\validation\rtx4060-full-offload-benchmark-20260930-164332\records\bench-fe91bac7-a964-4bba-8e40-8646a8bd8e3c.json`. Stdout y stderr reales conservados en el record y en `bundle/evidence/logs/bench-fe91bac7-a964-4bba-8e40-8646a8bd8e3c.runtime-log`.

Case nuevo: `case-671dd8f0-a401-43bc-bb2f-0ae9ebddc0e1`, store `C:\Users\USER\AppData\Local\jaull\cases\case-671dd8f0-a401-43bc-bb2f-0ae9ebddc0e1.json`. Bundle nuevo: `C:\Users\USER\OneDrive\Desktop\Jaull\validation\rtx4060-full-offload-benchmark-20260930-164332\bundle`.

Los comandos documentados case create, case validate, case export y case bundle validate terminaron exit 0. Case status=valid; bundle status=valid, sin razones de fallo, todos los archivos declarados verificados. Incluye exactamente el Validate indicado y el Benchmark nuevo, con cuatro archivos de evidencia referenciados. Comandos y outputs reales, incluida la validación, en case-commands.json.

Warnings preservados:
- benchmark bench-fe91bac7-a964-4bba-8e40-8646a8bd8e3c: available RAM differed at scan time (21055684608 vs 20171747328 bytes)
- benchmark bench-fe91bac7-a964-4bba-8e40-8646a8bd8e3c: available VRAM on GPU 0 differed at scan time (6807072768 vs 7450460160 bytes)

## Memoria y limitaciones

Benchmark peak process RSS: 4827664384 bytes. NVML process allocation: peak_vram_bytes=null. No se sustituye por VRAM del dispositivo. BenchmarkObservation no contiene runtime_reported_allocation estructurado; se conserva el raw output sin fabricar un campo equivalente.

En el Validate reutilizado, runtime_reported_allocation y NVML permanecen separados tal como se guardaron. Sus PredictionComparison y motivos de methodologically_unavailable no se modificaron. El overhead de runtime tiene semántica proxy; no se calibraron estimaciones a partir de este Benchmark. No se afirma cumplimiento de SLOs ni equivalencia entre latencia Validate y throughput Benchmark.

Bundle valid significa integridad y consistencia interna, no instalación completa del runtime ni atribución completa de VRAM. La ausencia del launcher y el intento de recuperación denegado no invalidan los hashes del llama-bench y las DLLs utilizados.

El primer preflight se detuvo por el launcher ausente. La recuperación falló antes de cualquier Benchmark; el segundo preflight también conservó un error del helper al intentar volver a escribir un diagnóstico existente. La comprobación específica del Benchmark pasó después; se ejecutó una única medición de pp512/tg128/r3. Diagnósticos retenidos, sin sustituir su resultado por éxito del preflight completo.

Logs versionados con rutas/identificadores y salida del modelo; no anonimizados. No se repitieron gates completos por tratarse de una ejecución de Benchmark sin cambios de código. AGENTS.md no está presente. No se sobrescribió el case/bundle anterior ni se adjuntó ningún benchmark antiguo.

## Git al terminar la medición (histórico)

git status --short:
```text
?? validation/rtx4060-full-offload-benchmark-20260930-164332/
```

git diff --stat: sin salida. git diff --check: exit 0. Diff staged también vacío. No git add, commit ni push.

## Revisión y limpieza posterior

El bundle es la entrega portable canónica. Desde WSL se comprobó que sus records
coinciden con el Validate original y el Benchmark guardado, que ambos conservan
el mismo plan y SHA del artefacto, y que el stdout/stderr del benchmark coincide
con su sidecar. La validación offline devuelve `valid` con los dos warnings de
memoria disponible ya indicados.

Se retiraron las cuatro copias externas de logs, preflight y hashes tras verificar
igualdad byte a byte con `bundle/evidence/`. También se retiraron `git-final.json`
(bookkeeping ya resumido arriba) y `bundle-validation.json` (resultado conservado
en `case-commands.json` y `case-index.json`, reproducible con el comando siguiente).
El bundle y su manifest no se modificaron. Los records originales, diagnósticos
de preflight y campañas anteriores permanecen intactos.

```sh
UV_CACHE_DIR=/tmp/uv-cache uv run --python 3.12 jaull experiments case bundle validate validation/rtx4060-full-offload-benchmark-20260930-164332/bundle --json
```
