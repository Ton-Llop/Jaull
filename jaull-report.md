# jaull recommendation report

Generated: 2026-08-04T09:12:30.970341+00:00

## Hardware

- CPU: AMD Ryzen 5 5500U with Radeon Graphics
- RAM: 7.4 GiB
- GPU: no NVIDIA GPU detected
- Platform: Linux 6.6.87.2-microsoft-standard-WSL2 (x86_64)

## Requirements

- Use case: general_chat
- Priority: balanced
- Languages: en
- Concurrency: One user
- Context: 4096 tokens
- Commercial use required: yes

## Recommendations

### 1. Qwen/Qwen2.5-7B-Instruct-AWQ — Best Effort

- Score: 70/100
- Memory fit: 85%
- Concurrency fit: 100%
- Capability: 78%
- Compatibility: compatible
- Confidence: low
- License: apache-2.0 (commercial_allowed)
- Precision: int4
- Context: 4096 tokens
- Runtime: transformers

**Why this model?**

- Strong match for general chat and assistant use.
- Model metadata lists EN.
- Fits in the detected memory at int4 precision.
- Fits the detected memory with room to spare.
- apache-2.0 license is generally suitable for commercial use.
- Suggested context of 4096 tokens.

**Limitations and warnings**

- int4 is a theoretical estimate: this repository ships no confirmed quantized artifact, so loading it at that precision requires a runtime that quantizes on the fly.
- Grouped-Query Attention: 4 KV heads for 28 attention heads.
- Confidence is low because part of the model metadata was missing.
- The selected precision is a theoretical estimate; no pre-quantized artifact was found in this repository.

**Series ladder**

- Qwen/Qwen2.5-0.5B-Instruct — 231M — comfortable
- Qwen/Qwen2.5-1.5B-Instruct — 793M — tight fit
- Qwen/Qwen2.5-3B-Instruct — 1.8B — tight fit
- Qwen/Qwen2.5-3B-Instruct-GGUF — 3B — fits
- Qwen/Qwen2.5-7B-Instruct — 4.3B — does not fit
- Qwen/Qwen2.5-Coder-7B-Instruct-GGUF — 7B — does not fit

### 2. Qwen/Qwen2-1.5B-Instruct — Higher quality but tighter fit

- Score: 56/100
- Memory fit: 22%
- Concurrency fit: 100%
- Capability: 64%
- Compatibility: tight
- Confidence: low
- License: apache-2.0 (commercial_allowed)
- Precision: int8
- Context: 4096 tokens
- Runtime: transformers

**Why this model?**

- Strong match for general chat and assistant use.
- Model metadata lists EN.
- Fits in the detected memory at int8 precision.
- apache-2.0 license is generally suitable for commercial use.
- Suggested context of 4096 tokens.

**Limitations and warnings**

- int8 is a theoretical estimate: this repository ships no confirmed quantized artifact, so loading it at that precision requires a runtime that quantizes on the fly.
- Grouped-Query Attention: 2 KV heads for 12 attention heads.
- Estimated memory leaves limited free VRAM; other applications may not fit.
- Confidence is low because part of the model metadata was missing.
- The selected precision is a theoretical estimate; no pre-quantized artifact was found in this repository.

### 3. TinyLlama/TinyLlama-1.1B-Chat-v1.0 — Alternative

- Score: 51/100
- Memory fit: 34%
- Concurrency fit: 100%
- Capability: 67%
- Compatibility: compatible
- Confidence: low
- License: apache-2.0 (commercial_allowed)
- Precision: int8
- Context: 4096 tokens
- Runtime: transformers

**Why this model?**

- Model metadata lists EN.
- Fits in the detected memory at int8 precision.
- Fits the detected memory with room to spare.
- apache-2.0 license is generally suitable for commercial use.
- Suggested context of 4096 tokens.

**Limitations and warnings**

- int8 is a theoretical estimate: this repository ships no confirmed quantized artifact, so loading it at that precision requires a runtime that quantizes on the fly.
- Grouped-Query Attention: 4 KV heads for 32 attention heads.
- Confidence is low because part of the model metadata was missing.
- The selected precision is a theoretical estimate; no pre-quantized artifact was found in this repository.

---

License information is reported from model metadata and is not legal advice; check the model's license yourself before commercial use.
