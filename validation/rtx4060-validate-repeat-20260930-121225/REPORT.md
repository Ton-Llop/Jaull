# Repetición de Validate detenida antes de ejecutar

Rama c_night; source commit 0af3b4fbdf23607e86de13e394961c422ccb1c24. Fix de --batch-size presente. Git inicialmente limpio; al finalizar solo aparece este directorio de diagnóstico sin seguimiento. Sin staging ni cambios de código.

El SHA del baseline fue recalculado y coincide con 65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423. Revision 8911e8a47f92bac19d6f5c64a2e2095bd2f7d031, Qwen2.5-7B-Instruct-Q4_K_M.gguf. Entorno uv locked comprobado, sin cambios necesarios.

El archivo C:/tools/llama.cpp/llama-cli.exe ya no existe. Jaull devuelve binary_status=missing y readiness=not_ready, reason=runtime_missing. No se lanzó Validate y no existe un nuevo ExperimentRecord. No se atribuye una causa a la desaparición.

Se buscaron runtimes y archivos locales en .venv, C:/tools, Downloads y ubicaciones habituales. Las dos copias locales de llama-cli corresponden a b10357 y su SHA256 no coincide con el launcher b11258 usado anteriormente; no se sustituyeron. El llama-bench.exe existente tampoco coincide con el launcher original.

SHA original esperado de llama-cli.exe: e6f48860add2ff4c133b60a2f3d7ddd835fa8e85a55c3a045f46079614288436.

No se descargó ni instaló nada. Se requiere autorización explícita del usuario antes de descargar llama.cpp, según las reglas originales. Se propone recuperar el build b11258 desde su paquete oficial en una carpeta local del proyecto y comprobar los hashes originales antes de repetir solo Validate. No se propone cambiar de build ni alterar flags.

Diagnósticos locales: failure.txt, prepare.log, runtime-capability.json, runtime-readiness.json, runtime-search.log, artifact-hash-check.json y summary.json. No se publicaron logs ni se modificaron evidencias anteriores. Benchmark no repetido. Gates completos no repetidos por el alcance Validate-only.
