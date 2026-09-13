# B001-R4: Qwen2.5-7B Q4_K_M on RTX 2060

This baseline records a controlled comparison of Qwen2.5-7B-Instruct Q4_K_M
on an RTX 2060 with a 4096-token scenario. It is evidence for the experimental
protocol, not a calibration target for the Hardware Fit Analyzer.

| Configuration | Generation throughput | Prompt throughput (512) | Device-wide VRAM peak delta |
| --- | ---: | ---: | ---: |
| Launch policy (`-ngl 24`) | 21.50 tok/s | 1263.54 tok/s | 3.87 GiB |
| Full offload (`-ngl -1`) | 59.36 tok/s | 1368.91 tok/s | 4.00 GiB |

The cases are stored under the local Jaull case store as:

- `case-b65dc3b3-fdff-4f56-bcc1-61004d482be8` for the launch policy.
- `case-ab4553e2-c4dc-40e4-b612-0cb754b70446` for full offload.

Both records pin the artifact
`bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf`, its
SHA-256, hardware fingerprint, llama.cpp build `689e227db (10357)`, a 512 MiB
device reserve and a 10% safety margin. The benchmark carries `ctx=4096` as
scenario provenance; llama-bench itself does not enforce that context size.

The device-memory captures use `nvidia-smi` and represent total device memory,
including unrelated consumers. They are not process-attributed CUDA allocation
and must not be compared directly with HFA planning budgets. The cases remain
`partial` because this llama-cli build did not report a reliable marker for the
backend that actually executed the inference. CUDA availability and observed
device activity are retained as evidence, not converted into an inferred
backend field.

Export either case as a portable offline snapshot:

```bash
jaull experiments case export <case-id> <destination>
jaull experiments case bundle validate <destination> --json
```

The bundle index verifies the exported bytes. When a case reference already
declares an evidence checksum or size, export verifies that historical claim
before copying; the bundle does not replace it with a newly calculated identity.

The next physical baseline should run from a clean commit and execute three
alternating launch-policy/full-offload pairs before comparing another GPU.
