"""Historical audited identities, as test data rather than an execution whitelist."""

ARTIFACT_PINS = {
    "9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0": {
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "revision": "52e7645ba7c309695bec7ac98f4f005b139cf465",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "format": "gguf",
        "quantization": "Q4_K_M",
        "size_bytes": 668788096,
    },
    "b46661073c18e5b56a41fa320975f866a00def1ff08feef4718e013258896f8c": {
        "repo_id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "revision": "91cad51170dc346986eccefdc2dd33a9da36ead9",
        "filename": "qwen2.5-1.5b-instruct-q5_k_m.gguf",
        "format": "gguf",
        "quantization": "Q5_K_M",
        "size_bytes": 1285494304,
    },
}
