> Publication note (2026-09-30): this is an anonymized derivative. Statements below about original bytes/hashes describe collection-time evidence, not this public copy. See validation/public-anonymization.json and the campaign PUBLICATION.md.

# Repetición de Validate — RTX 4060

Source commit: `0af3b4fbdf23607e86de13e394961c422ccb1c24`, rama `c_night`. Estado inicial con el diagnóstico previo sin seguimiento; final añade este nuevo directorio. La ubicación del runtime cambia a .venv; los 54 EXEs/DLLs coinciden byte por byte con la ejecución anterior. Sin cambios versionados ni staging. No se repitió Benchmark ni se modificó evidencia anterior.

Mismo baseline: bartowski/Qwen2.5-7B-Instruct-GGUF / Qwen2.5-7B-Instruct-Q4_K_M.gguf, revision 8911e8a47f92bac19d6f5c64a2e2095bd2f7d031. SHA recalculado correcto: 65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423. Mismos binarios/DLLs del runtime, verificados por SHA.

Contexto 4096, concurrencia 1, task general_chat, interactive, sin SLOs; flags del runtime seleccionado anterior preservados. Hardware medido de nuevo; predicción recalculada con los inputs congelados previos y hardware actual.

Comando real mediante ExperimentRunner:
```text

```

success=False; exit=3221226505; duration_seconds=6.912453200000527; --batch-size 1 aplicado=False.

ID: `exp-cf9bda77-e4cb-4f74-b48d-5a2cb4b1a39e`. Original: `C:\Users\USER\AppData\Local\jaull\experiments\exp-cf9bda77-e4cb-4f74-b48d-5a2cb4b1a39e.json`. Copia: `C:\Users\USER\OneDrive\Desktop\Jaull\validation\rtx4060-validate-repeat-20260930-122455\records\exp-cf9bda77-e4cb-4f74-b48d-5a2cb4b1a39e.json`. Raw log local: `C:\Users\USER\OneDrive\Desktop\Jaull\validation\rtx4060-validate-repeat-20260930-122455\logs\exp-cf9bda77-e4cb-4f74-b48d-5a2cb4b1a39e.runtime-log`.

Observación y PredictionComparison originales conservados en experiment-result.json y summary.json, sin convertir indisponibilidad a porcentaje ni cero. runtime_reported_allocation y NVML process allocation se mantienen separados.

Gates completos no repetidos: alcance solicitado solo Validate. AGENTS.md sigue ausente. No se crearon case/bundle nuevos. Logs locales, no publicados.
