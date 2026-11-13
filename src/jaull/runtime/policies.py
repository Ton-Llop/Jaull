"""Documented constants for the RuntimeSelector."""

from __future__ import annotations

MiB = 1024 * 1024

# --------------------------------------------------------------------------
# Architectures that vLLM has first-class support for. Not exhaustive; the
# real support matrix in vLLM is broader. When in doubt, we fall back to
# recommending Transformers (which is architecturally portable).
# --------------------------------------------------------------------------
VLLM_SUPPORTED_ARCHITECTURES: frozenset[str] = frozenset(
    {"llama", "qwen2", "mistral", "phi3", "gemma", "gemma2", "gemma3"}
)

# --------------------------------------------------------------------------
# llama.cpp heuristics for --n-gpu-layers.
# --------------------------------------------------------------------------
LLAMA_CPP_HEADROOM_BYTES = 256 * MiB  # kernel workspaces + KV growth
LLAMA_CPP_DEFAULT_LAYERS_WHEN_UNKNOWN = 20  # conservative fallback with warning

# Context length assumed when a launch plan is built without a memory estimate
# (e.g. `jaull run --n-gpu-layers N`, which skips estimation). Mirrors the
# defensive default in jaull.runtime.llama_cpp_runner.
LLAMA_CPP_DEFAULT_CONTEXT_SIZE = 4096

# --------------------------------------------------------------------------
# Managed llama.cpp installs (Setup). One pinned upstream build, the official
# prebuilt archives and the SHA-256 the upstream release publishes for each.
# CUDA uses the 12.x archives: 13.x needs a newer driver than the RTX 2060 test
# machine is guaranteed to have. HIP is published upstream but not validated,
# so it is not offered.
# --------------------------------------------------------------------------
LLAMA_CPP_PINNED_BUILD = "b11889"
LLAMA_CPP_RELEASE_URL = "https://github.com/ggml-org/llama.cpp/releases/download"
LLAMA_CPP_MANAGED_DIRNAME = "llama.cpp"
# Transport limits for the archive download (CUDA archives are several hundred MiB).
LLAMA_CPP_DOWNLOAD_CONNECT_TIMEOUT_SECONDS = 60.0
LLAMA_CPP_DOWNLOAD_READ_TIMEOUT_SECONDS = 600.0

# (os, arch, backend) -> ((file name, sha256), ...). The first entry is the
# llama.cpp archive; later ones are runtime libraries unpacked beside it.
LLAMA_CPP_RELEASE_ASSETS: dict[tuple[str, str, str], tuple[tuple[str, str], ...]] = {
    ("windows", "x86_64", "vulkan"): (
        ("llama-b11889-bin-win-vulkan-x64.zip",
         "3874d6cb33c3540a68191c1ca31b774bd72914e8ca65e51dfe23e3dae8201f00"),
    ),
    ("windows", "x86_64", "cpu"): (
        ("llama-b11889-bin-win-cpu-x64.zip",
         "bf563dffb502a230ae06c9dc455b54cbc079e7e0bc270cd4d6191209b28d175d"),
    ),
    ("windows", "x86_64", "cuda"): (
        ("llama-b11889-bin-win-cuda-12.4-x64.zip",
         "ce07031c3e4ab2a98d8b6addf1937b9f60d743208a569d3aad116cd25d6a41bd"),
        ("cudart-llama-bin-win-cuda-12.4-x64.zip",
         "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6"),
    ),
    ("linux", "x86_64", "vulkan"): (
        ("llama-b11889-bin-ubuntu-vulkan-x64.tar.gz",
         "1b0bce67a1c3dd683d5825b082de9d70ee718dcf186fdc67e0e00872b2626c89"),
    ),
    ("linux", "x86_64", "cpu"): (
        ("llama-b11889-bin-ubuntu-x64.tar.gz",
         "2076290f3b80b7642e2dc165c01fbf1be386a56b4612ba45576d20527834d350"),
    ),
    ("linux", "x86_64", "cuda"): (
        ("llama-b11889-bin-ubuntu-cuda-12.8-x64.tar.gz",
         "85bf5c3a64bb6b16219bc5381d952046460533ce5394a1e141b25b0e425f5409"),
        ("cudart-llama-b11889-bin-ubuntu-cuda-12.8-x64.tar.gz",
         "fea3678dcd01f05f8e76a5a5f8384c3376337c3ca9af7762410762d5f3afceb2"),
    ),
}
