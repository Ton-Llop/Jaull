# Validate repetido después del fix de batch

Resultado: fallo real de llama.cpp, conservado sin modificar. Solo Validate; Benchmark no repetido.

Fuente: c_night, commit 0af3b4fbdf23607e86de13e394961c422ccb1c24. Sin cambios de código, staging, commit, push ni cambio de rama. Los directorios nuevos son evidencia local sin seguimiento.

## Runtime recuperado

Descarga autorizada del release oficial https://github.com/ggml-org/llama.cpp/releases/tag/b11258, asset llama-b11258-bin-win-cuda-12.4-x64.zip. SHA256 del archivo descargado: 77c7b50e48aa96f4e5e21432ae83df11cb72eae5a4e38a4b0fbcafbde5458525, coincide con el digest publicado por GitHub. Runtime local en .venv/rtx4060-runtime-b11258-recovery-20260930-121948/bin. Dependencias CUDA faltantes en el ZIP copiadas desde la instalación existente después de verificar su SHA. Los 54 hashes EXE/DLL coinciden con la campaña anterior. No se instaló nada globalmente ni se modificó C:/tools.

Build 11258, commit ba0ba54d9; CUDA0 confirmado por Jaull. Los probes no son una ejecución sustitutiva. Se usó AdvisorService.run_experiment / ExperimentRunner.

## Baseline

bartowski/Qwen2.5-7B-Instruct-GGUF, revision 8911e8a47f92bac19d6f5c64a2e2095bd2f7d031, archivo Qwen2.5-7B-Instruct-Q4_K_M.gguf. SHA recalculado correcto: 65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423.

Flags del runtime seleccionado anterior conservados: contexto 4096, 28 unidades GPU, batch 1, CUDA0; concurrencia 1, task general_chat, workload interactive, sin SLOs. Hardware medido de nuevo; predicción recalculada y vinculada al launch anterior por AdvisorService.plan_execution, sin alterar los cálculos numéricos ni ajustar offload.

## Ejecución real

ExperimentRecord: exp-cf9bda77-e4cb-4f74-b48d-5a2cb4b1a39e. success=false, failure_reason=non_zero_exit, exit_code=3221226505, duración 6.9124532 segundos. Las rutas completas originales y copias constan en final-summary.json. Record y raw log guardados localmente en records/ y logs/.

Los logs confirman n_batch = 1 y n_ubatch = 1. Por tanto, el fix sí llegó al runtime. Después aparece: add: failed to add token 0 to the batch (error -1, n_tokens = 1). Se conserva el fallo; no se cambió el batch ni se ejecutó otra configuración.

Limitación de provenance: en este camino fallido ExperimentRunner deja executed_command vacío y observed_backend=null, aunque los probes y el raw output indican CUDA. No se rellenó ni modificó el record. El helper original calculó batch_size_1_applied=false a partir de ese command vacío: ese booleano es ausencia de evidencia en el trace, no prueba de que se omitiera el flag. final-summary.json documenta la confirmación real de batch 1 en los logs.

NVML process allocation y runtime_allocation permanecen null en el record fallido; no se sustituyen por VRAM del dispositivo ni por cálculos. Comparaciones RAM/VRAM methodologically_unavailable con motivos y nulls originales; no se convierten a porcentajes o cero.

## Intentos y límites

Se conserva el intento previo 121225, detenido antes de inferencia porque faltaba el launcher. También se conserva el intento 121948: el helper invocó ExperimentRunner pero falló al crear el record por incoherencia entre el runtime y el runtime_recommendation de la predicción. Ese intento no guardó ExperimentRecord y perdió el raw output antes de la persistencia opt-in; no se reconstruye. El helper se corrigió en una nueva copia temporal usando el planificador existente, sin modificar código del proyecto.

El helper final terminó exit 0 porque guardó correctamente un record fallido; eso no significa éxito de Validate. El resultado real está en observation.success=false y exit_code=3221226505. Gates completos no repetidos por el alcance solicitado Validate-only. AGENTS.md ausente. No se crearon case/bundle nuevos ni se publicó ningún log.
