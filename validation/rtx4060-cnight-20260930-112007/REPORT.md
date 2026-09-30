# Validación real RTX 4060 — c_night

Campaña local terminada: Validate y Benchmark correctos; case y bundle válidos. Sin staging, commit, push ni cambios de rama. Sin cambios de código, calibraciones, ranking o fórmulas. Evidencia histórica no modificada.

## Fuente y gates

Commit: `8423fd7b1acdd5a51bec6d9d103c564ab3d8033c`. Rama inicial/final: `c_night`. Estado inicial sucio por tres directorios históricos sin seguimiento; sin cambios versionados. Estado final añade solamente esta campaña sin seguimiento; diff normal y staged vacíos. Consultar initial-git.json y final-git.json.

Python 3.12.14, Windows nativo, entorno `.venv/rtx4060-py312`. `uv sync --locked --python 3.12 --check` confirmó que no eran necesarios cambios. download() llama a clear_sha256(target) antes de self.downloader. AGENTS.md no está presente; se leyó docs/experimental-validation.md.

| Gate | Resultado |
| --- | --- |
| pytest completo | 1909 passed, 122.63 s, exit 0 |
| ruff check . | pass, exit 0 |
| mypy src | pass, 237 archivos, exit 0 |
| compileall src | pass, exit 0 |
| architecture tests | 4 passed, exit 0 |
| git diff --check | pass, exit 0; repetido al finalizar |

## Hardware y runtime

Windows 11 AMD64, RTX 4060. NVML/nvidia-smi: VRAM total 8188 MiB, libre 5063 MiB, usada 2895 MiB al consultar; driver 616.92, API CUDA detectada 13.4. Disponibilidad variable. Fuentes completas: hardware.json, gpu.csv, scan.log y doctor logs locales.

Runtime existente: `C:/tools/llama.cpp/llama-cli.exe` y `C:/tools/llama.cpp/llama-bench.exe`. Ambos reportan 0.5.0-dev, build 11258, commit ba0ba54d9, Clang 20.1.8 Windows x86_64. CUDA0 confirmado por el probe de Jaull. Hashes de EXEs/DLLs en runtime-binary-hashes.json. No se descargó ni instaló runtime, driver o herramienta global. Se usó inspect_llama_bench con allow_empty_workload_probe=True; este build respondió a --version.

## Baseline y plan

Repo: `bartowski/Qwen2.5-7B-Instruct-GGUF`. Revision resuelta: `8911e8a47f92bac19d6f5c64a2e2095bd2f7d031`. Archivo: `Qwen2.5-7B-Instruct-Q4_K_M.gguf`, 4683074240 bytes. SHA256 recalculado y verificado: `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423`.

Contexto 4096, concurrencia 1, task general_chat, mode interactive, sin SLOs. Plan calculado por AdvisorService.plan_execution y prepare_execution_plan, sin overrides de GPU. Flags: --model (preview sustituido por archivo verificado en la ejecución), --ctx-size 4096, --n-gpu-layers 28, --batch-size 1, --device CUDA0. Warning: Local tensor refinement not applied: Runtime build is not verified.

Validate usa AdvisorService.run_experiment / ExperimentRunner con el prompt de la UI y capture_raw_logs=True. Benchmark usa AdvisorService.run_benchmark_matrix, solo backend CUDA, 28 unidades derivadas del mismo plan, pp512, tg128, 3 repeticiones, timeout 900 s. No se ejecutó ninguna configuración alternativa. Las APIs y comandos exactos están en actual-commands.json; helpers temporales en .venv, copias de texto en helpers/.

Comando Validate real:
```text
C:\tools\llama.cpp\llama-cli.exe --model C:\Users\PC\AppData\Local\jaull\models\bartowski\Qwen2.5-7B-Instruct-GGUF\Qwen2.5-7B-Instruct-Q4_K_M.gguf --ctx-size 4096 --n-gpu-layers 28 --device CUDA0 --no-display-prompt --color off --no-show-timings --simple-io --single-turn --verbose --prompt "Explain briefly what artificial intelligence is."
```

Comando Benchmark real:
```text
C:\tools\llama.cpp\llama-bench.exe -m C:\Users\PC\AppData\Local\jaull\models\bartowski\Qwen2.5-7B-Instruct-GGUF\Qwen2.5-7B-Instruct-Q4_K_M.gguf -dev CUDA0 -ngl 28 -p 512 -n 128 -r 3
```

El runner Validate omite --batch-size 1 del plan. llama-bench omite --ctx-size y --batch-size: 4096 es contexto de procedencia, no una condición aplicada por su comando. No se corrigió el código para cambiar esto.

## Resultados y rutas

ExperimentRecord: `exp-38e97695-954d-42ef-83e0-1bacecda01dc`, original `C:\Users\PC\AppData\Local\jaull\experiments\exp-38e97695-954d-42ef-83e0-1bacecda01dc.json`, copia `C:\Users\PC\OneDrive\Desktop\Jaull\validation\rtx4060-cnight-20260930-112007\records\exp-38e97695-954d-42ef-83e0-1bacecda01dc.json`. Raw log local: `C:\Users\PC\OneDrive\Desktop\Jaull\validation\rtx4060-cnight-20260930-112007\logs\exp-38e97695-954d-42ef-83e0-1bacecda01dc.runtime-log`.

Validate success=true, exit=0, duración 5.5956059 s; observed_backend=cuda. PredictionComparison.compatibility=correct_success. Esa duración no es TTFT ni una medida de throughput.

BenchmarkRecord: `bench-a07438ca-cf57-44ef-9d3d-7183ceb2a66e`, original `C:\Users\PC\AppData\Local\jaull\benchmarks\bench-a07438ca-cf57-44ef-9d3d-7183ceb2a66e.json`, copia `C:\Users\PC\OneDrive\Desktop\Jaull\validation\rtx4060-cnight-20260930-112007\records\bench-a07438ca-cf57-44ef-9d3d-7183ceb2a66e.json`. Raw log: `C:\Users\PC\OneDrive\Desktop\Jaull\validation\rtx4060-cnight-20260930-112007\logs\bench-a07438ca-cf57-44ef-9d3d-7183ceb2a66e.runtime-log`.

Benchmark success=true, exit=0, duración del proceso 12.3847817 s; metodología llama_bench_v1, tres repeticiones. pp512: 2408.90 ± 49.77 tokens/s. tg128: 44.22 ± 0.60 tokens/s. TTFT, model_load y warmup no disponibles en el record; no inferidos. Ambas filas requeridas presentes.

## Memoria y comparaciones

Validate RAM RSS pico: 4852645888 bytes. Benchmark RAM RSS pico: 4828770304 bytes. NVML process allocation: peak_vram_bytes=null en ambos records. No se atribuye una causa definitiva al null; WDDM puede no aportar memoria por proceso. La memoria total/libre del dispositivo no sustituye esta medición.

Validate runtime_reported_allocation CUDA0: 4632944312 bytes (modelo 4221441147, KV 226492416, compute 185010749). Esta fuente no está confirmada por el driver y se conserva separada de NVML. runtime_allocation.runtime_build=null; el build se captura aparte en runtime_capability.

RAM: methodologically_unavailable porque RSS incluye páginas del modelo mapeado y no mide la cuota host bajo offload. VRAM total y todos los componentes: methodologically_unavailable porque la colocación de pesos no pertenecientes a bloques es una cota, no una predicción puntual específica del dispositivo. Se conservan motivos y errores null; no se calculan porcentajes ni se convierten a cero. Compatibility=correct_success es la comparación disponible. No se compara duración de Validate con throughput Benchmark.

## Case y bundle

Case: `case-a8b66e98-447b-4faa-9d11-ab839d91b3b2`, original `C:\Users\PC\AppData\Local\jaull\cases\case-a8b66e98-447b-4faa-9d11-ab839d91b3b2.json`, copia `C:\Users\PC\OneDrive\Desktop\Jaull\validation\rtx4060-cnight-20260930-112007\records\case-a8b66e98-447b-4faa-9d11-ab839d91b3b2.json`. Bundle: `C:\Users\PC\OneDrive\Desktop\Jaull\validation\rtx4060-cnight-20260930-112007\bundle`.

case create, case validate, case export y case bundle validate ejecutados mediante uv, todos exit 0. Bundle status=valid, todos los archivos declarados verificados, 1 experiment, 1 benchmark, 10 archivos de evidencia. Comandos y salidas originales en case-commands.json y archivos case-*/bundle-validate.*.

Warning de runtime_build preservado: los textos de version difieren por los mensajes del loader CUDA/RPC/CPU capturados en llama-bench; ambos contienen build 11258 y commit ba0ba54d9. No se modificaron los records para normalizar esa diferencia.

## Fallos, warnings y límites

Sin fallos en gates, Validate, Benchmark ni case/bundle. Falló el inicio del helper sandbox antes de ejecutar comandos; las ejecuciones posteriores usaron escalación revisada automáticamente. AGENTS.md ausente. El primer doctor dio runtime_missing; la búsqueda encontró la instalación existente y el doctor con PATH local se conservó por separado. Hub avisó de peticiones sin autenticación. Persisten las limitaciones de flags, build no verificado por política y comparaciones de memoria descritas arriba. No se extrapola esta ejecución a otras GPUs o builds ni se afirma cumplimiento de SLOs.

Los logs quedan locales y pueden contener usuario, rutas, hostname y UUID. No se publicaron ni se añadieron a staging. El bundle se validó localmente; no se transfirió a otro equipo.
