# Plan nuevo y Validate — RTX 4060

Resultado: success=true, exit_code=0, duración 5.2630163 segundos. Solo Validate, sin Benchmark adicional.

## Fuente y entorno

Rama c_night, commit 1d12f6306bfcac2cf13786945bf2b4f22582184c. Git inicial limpio; final añade únicamente este directorio de evidencia sin seguimiento. Sin cambios versionados, staging, commits ni push. AGENTS.md ausente; docs/experimental-validation.md leído. Python 3.12.14, PowerShell/Windows nativo, entorno .venv/rtx4060-py312. uv sync --locked --python 3.12 --check confirmó que no había cambios necesarios. Gates completos no repetidos: alcance solicitado plan nuevo y solo Validate.

## Baseline

Repo bartowski/Qwen2.5-7B-Instruct-GGUF, revision 8911e8a47f92bac19d6f5c64a2e2095bd2f7d031; Qwen2.5-7B-Instruct-Q4_K_M.gguf. SHA recalculado correcto: 65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423. Contexto 4096, concurrencia 1, task general_chat, interactive, sin SLOs.

## Plan y ejecución

Se construyeron InferenceConfiguration, análisis, predicción, ArtifactVariant y ModelIdentity nuevos. AdvisorService.plan_execution y prepare_execution_plan(for_benchmark=False) calcularon el plan con el hardware actual. No se cargaron planes, requests, launch ni flags anteriores. No hubo overrides manuales de GPU o batch. El único dato histórico leído fue el manifiesto de hashes para comprobar que el runtime es idéntico.

Flags del nuevo plan: --model, --ctx-size 4096, --n-gpu-layers -1, --device CUDA0. El preview de --model se sustituye por la ruta real del artefacto verificado en la ejecución. No hay --batch-size ni --ubatch-size en el comando. El runtime confirmó offloaded 29/29 layers, n_batch=2048 y n_ubatch=512. batch_size=1 en InferenceConfiguration se conserva como número de secuencias con KV y no se transforma en tamaño del batch de tokens del runtime.

Comando aplicado completo en actual-invocation.json y en el backend_trace del ExperimentRecord; flags reales de control: --ctx-size 4096 --n-gpu-layers -1 --device CUDA0 --no-display-prompt --color off --no-show-timings --simple-io --single-turn --verbose --prompt "Explain briefly what artificial intelligence is."

Ejecución mediante AdvisorService.run_experiment y ExperimentRunner. Un wrapper temporal de HostExecutionBackend conserva el comando y raw output antes de la persistencia para retener diagnósticos incluso si fallase la creación del record; delega la ejecución real al backend existente, sin mocks ni otro runner. Helpers fuera de archivos versionados, copia de texto en helper.py.txt.

## Runtime y observaciones

Runtime local .venv/rtx4060-runtime-b11258-recovery-20260930-121948/bin, llama.cpp build 11258, commit ba0ba54d9; CUDA0 confirmado. Se verificaron de nuevo los 54 hashes EXE/DLL contra la campaña original. No se descargó ni sustituyó otro build en esta ejecución.

ExperimentRecord exp-f05586ff-c90e-4239-821d-5baaf550bd2a. Original en AppData/Local/jaull/experiments; copia local en records/exp-f05586ff-c90e-4239-821d-5baaf550bd2a.json. Rutas absolutas originales y raw sidecar en summary.json. Raw logs conservados en logs/, sin publicar. La duración 5.2630163 s es duración del proceso, no TTFT ni throughput.

RAM RSS pico: 4842053632 bytes. NVML process allocation: peak_vram_bytes=null. runtime_reported_allocation CUDA0: 4748056986 bytes, fuente distinta de NVML, driver_confirmed=false. No se reemplaza el null de NVML por la memoria del dispositivo.

PredictionComparison.compatibility=correct_success. RAM methodologically_unavailable por las páginas mapeadas del modelo en RSS bajo offload, con motivo y errores null originales. VRAM disponible según los gates del comparador: predicted_bytes=5923133600, measured_bytes=4748056986, error_percent=-19.838765986976895, fuente runtime_reported_allocation. No es VRAM atribuida por el driver ni una afirmación de error sobre toda la memoria CUDA.

Componentes: modelo y KV con semántica direct; KV 234881024 bytes tanto previstos como observados. Overhead tiene semántica proxy: la predicción incluye conceptos no enumerados por el compute buffer del runtime. Se conserva su advertencia original y no se usa ese porcentaje para calibrar fórmulas. Todos los valores y motivos originales están en experiment-result.json/summary.json.

Warning preservado: Local tensor refinement not applied: Runtime build is not verified. Build y backend fueron detectados; este warning se refiere al ámbito de verificación de la política de refinamiento de Jaull. No se cambiaron constantes, calibraciones o fórmulas.

Sin fallos de ejecución en este intento. Los intentos anteriores y sus records quedaron intactos. No se crearon case/bundle nuevos ni se vinculó el Benchmark anterior a este nuevo plan de full offload.
