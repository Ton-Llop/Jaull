"""Group a GGUF file's tensors by name and size, reading only the header.

Meant to explain why llama.cpp loads fewer bytes on the GPU than the file size: it lists which
groups of tensors (token embeddings, output, blocks, vision or projector tensors) make up the file.
Standard library only; it does not load the weights.

    python3 gguf_tensor_groups.py /path/to/model.gguf [--top 12]
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

MIB = 1024 * 1024
SCALARS = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
STRING, ARRAY = 8, 9


def read_u32(f) -> int:
    return struct.unpack("<I", f.read(4))[0]


def read_u64(f) -> int:
    return struct.unpack("<Q", f.read(8))[0]


def read_string(f) -> str:
    return f.read(read_u64(f)).decode("utf-8", "replace")


def skip_value(f, kind: int) -> None:
    if kind in SCALARS:
        f.seek(SCALARS[kind], 1)
    elif kind == STRING:
        f.seek(read_u64(f), 1)
    elif kind == ARRAY:
        item, count = read_u32(f), read_u64(f)
        if item in SCALARS:
            f.seek(SCALARS[item] * count, 1)
        else:
            for _ in range(count):
                skip_value(f, item)
    else:
        raise ValueError(f"unknown GGUF value type {kind}")


def read_header(path: Path) -> tuple[dict[str, str | int], list[tuple[str, int]], int]:
    """Return (a few metadata values, [(tensor name, offset)], start of the tensor data)."""
    wanted = {"general.architecture", "general.name", "general.alignment"}
    meta: dict[str, str | int] = {}
    with path.open("rb") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit(f"{path} is not a GGUF file")
        version = read_u32(f)
        if version < 2:
            raise SystemExit(f"GGUF version {version} is not supported")
        tensor_count, kv_count = read_u64(f), read_u64(f)
        for _ in range(kv_count):
            key, kind = read_string(f), read_u32(f)
            if key in wanted and kind == STRING:
                meta[key] = read_string(f)
            elif key in wanted and kind == 4:
                meta[key] = read_u32(f)
            else:
                skip_value(f, kind)
        tensors = []
        for _ in range(tensor_count):
            name = read_string(f)
            dims = read_u32(f)
            f.seek(8 * dims, 1)
            f.seek(4, 1)  # tensor type
            tensors.append((name, read_u64(f)))
        align = int(meta.get("general.alignment", 32))
        data_start = (f.tell() + align - 1) // align * align
    return meta, tensors, data_start


def group_of(name: str) -> str:
    if name.startswith("blk."):
        return "blk (transformer blocks)"
    first = name.split(".")[0]
    return {"token_embd": "token_embd (input embeddings)", "output": "output (lm head)"}.get(
        first, first,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("gguf", type=Path)
    parser.add_argument("--top", type=int, default=12, help="largest tensors to list")
    parser.add_argument("--blocks", action="store_true",
                        help="also total each tensor kind across blocks and list per-block sizes")
    args = parser.parse_args()

    meta, tensors, data_start = read_header(args.gguf)
    file_size = args.gguf.stat().st_size
    order = sorted(range(len(tensors)), key=lambda i: tensors[i][1])
    sizes: dict[str, int] = {}
    for rank, index in enumerate(order):
        name, offset = tensors[index]
        end = tensors[order[rank + 1]][1] if rank + 1 < len(order) else file_size - data_start
        sizes[name] = end - offset

    groups: dict[str, list[int]] = {}
    for name, size in sizes.items():
        groups.setdefault(group_of(name), []).append(size)

    print(f"{args.gguf.name}: architecture={meta.get('general.architecture', '?')}, "
          f"{len(tensors)} tensors, file {file_size / MIB:,.1f} MiB, "
          f"header {data_start / MIB:.1f} MiB")
    print(f"{'group':34} {'tensors':>8} {'MiB':>10}")
    for group, items in sorted(groups.items(), key=lambda item: -sum(item[1])):
        print(f"{group:34} {len(items):>8} {sum(items) / MIB:>10,.1f}")
    total = sum(sizes.values())
    print(f"{'all tensors':34} {len(sizes):>8} {total / MIB:>10,.1f}")
    print(f"{'file minus tensors (header etc.)':34} {'':>8} {(file_size - total) / MIB:>10,.1f}")
    print(f"\nlargest {args.top} tensors (sizes include up to 31 bytes of alignment padding):")
    for name, size in sorted(sizes.items(), key=lambda item: -item[1])[: args.top]:
        print(f"  {size / MIB:>9,.1f} MiB  {name}")
    if args.blocks:
        print_blocks(sizes)


def print_blocks(sizes: dict[str, int]) -> None:
    """Totals per tensor kind (the part after 'blk.N.') and per block, for the blocks only."""
    kinds: dict[str, list[int]] = {}
    per_block: dict[int, int] = {}
    for name, size in sizes.items():
        parts = name.split(".", 2)
        if parts[0] != "blk" or len(parts) < 3:
            continue
        kinds.setdefault(parts[2], []).append(size)
        per_block[int(parts[1])] = per_block.get(int(parts[1]), 0) + size
    print("\ntensor kinds inside the blocks (count, total MiB):")
    for kind, items in sorted(kinds.items(), key=lambda item: -sum(item[1])):
        print(f"  {len(items):>4}  {sum(items) / MIB:>9,.1f}  {kind}")
    print("\nper block, MiB (index: size):")
    line = ""
    for index in sorted(per_block):
        line += f"{index}:{per_block[index] / MIB:.0f}  "
        if len(line) > 90:
            print(" ", line.rstrip())
            line = ""
    print(" ", line.rstrip())


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
