# llama.cpp Setup for Jaull

Jaull uses [`llama.cpp`](https://github.com/ggml-org/llama.cpp) as its local GGUF execution runtime.

This document describes CUDA, Vulkan and HIP/ROCm setup options for `llama-cli`.
The CUDA commands below are the validated development setup; Vulkan and HIP remain
experimental until they are exercised on the corresponding hardware.

## 1. Clone llama.cpp

```bash
mkdir -p ~/tools
cd ~/tools

git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```

## 2. Check NVIDIA / CUDA

Make sure WSL/Linux can see the NVIDIA GPU:

```bash
nvidia-smi
```

Check that the CUDA compiler is installed:

```bash
nvcc --version
```

Both commands should work before compiling `llama.cpp` with CUDA support.

## 3. Build llama-cli with CUDA

Configure the build:

```bash
cd ~/tools/llama.cpp

cmake -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON
```

Build `llama-cli`:

```bash
cmake --build build-cuda \
  --config Release \
  --target llama-cli \
  -j 2
```

The resulting binary is:

```text
~/tools/llama.cpp/build-cuda/bin/llama-cli
```

Verify it:

```bash
~/tools/llama.cpp/build-cuda/bin/llama-cli --version
```

### Optional: add llama-cli to PATH

For the current shell:

```bash
export PATH="$HOME/tools/llama.cpp/build-cuda/bin:$PATH"
```

For Zsh permanently:

```bash
echo 'export PATH="$HOME/tools/llama.cpp/build-cuda/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

Then:

```bash
which llama-cli
llama-cli --version
```

### Vulkan build (GGUF)

On a Linux host with a working Vulkan driver, build a separate runtime:

```bash
cd ~/tools/llama.cpp

cmake -B build-vulkan \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON

cmake --build build-vulkan \
  --config Release \
  --target llama-cli \
  -j 2
```

Check the runtime device list before using it through Jaull:

```bash
~/tools/llama.cpp/build-vulkan/bin/llama-cli --list-devices
```

Jaull only pins `--device Vulkan0` (or another reported identifier) after this
probe exposes a ready Vulkan device. `vulkaninfo` alone is hardware discovery;
it does not prove that a particular `llama-cli` binary was built with Vulkan.

### HIP/ROCm build (GGUF)

On a Linux AMD/ROCm host, build a HIP runtime with the ROCm toolchain installed:

```bash
cd ~/tools/llama.cpp

cmake -B build-hip \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON

cmake --build build-hip \
  --config Release \
  --target llama-cli \
  -j 2
```

Then verify the runtime probe:

```bash
~/tools/llama.cpp/build-hip/bin/llama-cli --list-devices
```

Jaull treats `HIP0`/`ROCm0` reported by llama.cpp as its HIP backend and passes
the runtime identifier to the launch plan. The identifier is not interchangeable
with a PyTorch device string: a ROCm PyTorch build normally still uses
`cuda:0`/`cuda` inside Transformers.

## 4. Download a GGUF with Jaull

From the Jaull repository:

```bash
uv run python -c '
from jaull.advisor.service import AdvisorService

advisor = AdvisorService.default()

artifact = advisor.resolve_artifact(
    "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    quantization="Q4_K_M",
)

artifact = advisor.download_artifact(artifact)
artifact = advisor.verify_artifact(artifact, full=False)

print(artifact.local_path)
'
```

Jaull stores downloaded models under:

```text
~/.local/share/jaull/models/
```

Example:

```text
~/.local/share/jaull/models/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

## 5. Test llama.cpp directly

```bash
~/tools/llama.cpp/build-cuda/bin/llama-cli \
  --model "$HOME/.local/share/jaull/models/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
  --ctx-size 4096 \
  --n-gpu-layers 99 \
  --single-turn \
  --prompt "Hello! Explain in one sentence what GGUF is."
```

`--single-turn` is important for programmatic execution: `llama-cli` generates one response and exits instead of remaining in interactive chat mode.

## 6. Run through Jaull

```bash
uv run jaull run \
  --model "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF" \
  --quantization "Q4_K_M" \
  --prompt "Hello! Explain in one sentence what GGUF is." \
  --llama-cli "$HOME/tools/llama.cpp/build-cuda/bin/llama-cli" \
  --ctx-size 4096 \
  --n-gpu-layers 99 \
  --timeout-seconds 30
```

The execution flow is:

```text
Hugging Face
    ↓
ArtifactService
    ↓
GGUF download + verification
    ↓
LlamaCppRunner
    ↓
HostExecutionBackend
    ↓
llama-cli
    ↓
CUDA / local hardware
    ↓
Generated text
```

## Notes

- `llama-cli` is not installed by cloning the repository; it must be compiled.
- CUDA builds require both a working NVIDIA GPU setup and the CUDA Toolkit (`nvcc`).
- Jaull currently executes GGUF artifacts through `llama-cli`.
- Vulkan and HIP/ROCm GGUF execution requires a matching llama.cpp build and a
  successful `--list-devices` probe; no backend is inferred from `vulkaninfo` alone.
- Native Transformers ROCm execution is a separate path and depends on a ROCm-enabled
  PyTorch installation. AWQ/GPTQ/bitsandbytes ROCm support is not claimed here.
- `--single-turn` is added by `LlamaCppRunner` so executions terminate cleanly.
- `n_gpu_layers=0` means CPU-only execution.
- GPU offloading can later be selected automatically by Jaull instead of being manually specified.
