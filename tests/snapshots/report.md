# jaull recommendation report

Generated: <redacted>

## Hardware

- CPU: AMD Ryzen 5 3600
- RAM: 32.0 GiB
- GPU: NVIDIA RTX 4070
- VRAM: 12.0 GiB
- Platform: Linux 6.8 (x86_64)

## Requirements

- Use case: coding
- Priority: balanced
- Languages: es, en
- Concurrency: One user
- Context: 4096 tokens
- Workload mode: interactive
- Commercial use required: yes

## Confirmed recommendations

### 1. org/Coder-7B — Best Effort

- Compatibility: comfortable
- License: apache-2.0 (commercial_allowed)
- Precision: float16
- Context: 4096 tokens
- Artifact: native float16 — confirmed
- Parameter count: 7B (name inference, low confidence)
- Weights estimate: high (exact size)
- KV cache estimate: high (derived calculation)
- Runtime overhead estimate: low (assumed heuristic)

**Why this position?** Base criteria, compared in order, for `balanced`:

1. viability: placement confirmed (comfortable)
2. plan constraints: none
3. requirement confirmation: commercial use confirmed
4. suitability: strong
5. runnability: strong
6. capability: adequate
7. execution fitness: strong
8. executability: strong
9. memory headroom: strong
10. plan evidence confidence: medium
11. quantization preference: n/a
12. popularity: 10,000 downloads

**Balanced selection** (before measured evidence):
- A plan without known strong/adequate capability goes behind every capable plan that is at least as viable as it (confirmed memory fit, rejection state, runnability); capable plans keep their order. Comparable measured evidence is applied afterwards.
- Selection order: pool #1 -> #1 (before diversity).

**Measured-balanced policy** (after base criteria, before diversity):
- No applicable measurement; this plan keeps its selection-adjusted slot.
- Balanced order: pool #1 -> #1 (before diversity).

<details>
<summary>Technical details: confidence and diagnostics</summary>

- Weights estimate: high (exact size): fixture
- KV cache estimate: high (derived calculation): fixture
- Runtime overhead estimate: low (assumed heuristic): fixture
- Memory estimation confidence: high
- Memory confidence aggregation: Weakest component or compatibility assessment; not a measured error probability or overall model reliability.
- Plan evidence confidence: medium
- Plan evidence basis: Metadata confidence: medium; without matching local execution evidence, the plan uses metadata and memory-estimate confidence, not runtime installation.
- Performance evidence: No comparable benchmark for ranking
- Quality evidence: No applicable evaluation
- Diagnostic composite score: 61/100 (memory fit 75%, capability 70%)
- Score role: This is not what ordered the list. Position follows the plan criteria and applicable measured policy, then diversity; a higher score need not rank first.

</details>

**Why this model?**

- Strong match for programming tasks.
- Model metadata lists EN, ES.
- Float16 precision fits comfortably in the detected memory.
- Leaves comfortable free memory after loading.
- apache-2.0 license is generally suitable for commercial use.
- Suggested context of 4096 tokens.

---

License information is reported from model metadata and is not legal advice; check the model's license yourself before commercial use.
