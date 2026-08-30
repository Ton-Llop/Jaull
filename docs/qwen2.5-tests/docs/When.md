# Qwen2.5-7B Q4_K_M · RTX 2060 · Hardware Fit validation

Fecha: 2026-08-30

## Entorno

- GPU: NVIDIA GeForce RTX 2060
- VRAM: 6 GiB
- RAM visible en WSL2: ~7.7 GiB
- Modelo: `bartowski/Qwen2.5-7B-Instruct-GGUF`
- Artifact: `Qwen2.5-7B-Instruct-Q4_K_M.gguf`
- Context length: 4096
- llama.cpp build: `b10357-689e227db`

## Metodología

La predicción de Jaull se calculó y congeló antes de ejecutar llama.cpp.

Se realizaron dos ejecuciones:

- baseline/predicted placement
- sweep de `-ngl` + prueba `--no-kv-offload`

Los outputs originales se conservan sin modificar como evidencia experimental.

## Resultado principal

Jaull predijo:

- placement: `GPU_OFFLOAD`
- `gpu_layers`: 18 / 28

llama.cpp consiguió:

- 29 / 29 capas offloaded
- ~4528 MiB de CUDA buffers en full offload
- ~56 tok/s de generación en el run observado

Con `-ngl 18`:

- ~3026 MiB CUDA buffers
- ~13 tok/s

Por tanto, el placement conservador de Jaull es ejecutable, pero la estimación actual de `gpu_layers` infrautiliza significativamente la GPU en este caso.

## Observación KV cache

Con `-ngl 18`:

- KV offload ON: ~3026 MiB CUDA buffers
- KV offload OFF: ~2890 MiB CUDA buffers
- diferencia observada: ~136 MiB

Jaull estimó un KV total de 224 MiB.

La distribución observada es compatible aproximadamente con:

`KV_GPU ≈ KV_total × offloaded_layers / runtime_layers`

No se modifica todavía ninguna fórmula a partir de un único experimento.

## Nota

`gpu_required_bytes` incluye reserve y safety margin, por lo que no debe compararse directamente con la memoria física asignada por llama.cpp.

La comparación física debe usar la métrica equivalente a `gpu_physical_bytes`.