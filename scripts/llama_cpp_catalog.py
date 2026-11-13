"""Print the pinned llama.cpp catalog for an upstream build, to paste into runtime/policies.py.

Moving the pinned build is the author's decision after testing it on real hardware;
this only writes out the file names and the SHA-256 GitHub publishes for them.

    uv run python scripts/llama_cpp_catalog.py b12360
"""

from __future__ import annotations

import argparse
import sys

import httpx

from jaull.runtime import policies
from jaull.runtime.llama_cpp_release import UnsupportedPlatformError
from jaull.runtime.llama_cpp_updates import newest_build, package_from_release

_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{tag}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tag", help="Upstream build tag, e.g. b12360.")
    tag = parser.parse_args().tag

    timeout = policies.LLAMA_CPP_UPDATE_CHECK_TIMEOUT_SECONDS
    response = httpx.get(_API.format(tag=tag), timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    release = newest_build([payload])
    if release is None:
        sys.stderr.write(f"{tag} is not a b<number> build tag.\n")
        return 2

    from jaull.domain.hardware import ComputeBackend

    print(f'LLAMA_CPP_PINNED_BUILD = "{release.tag}"')
    print(f'LLAMA_CPP_PINNED_BUILD_PUBLISHED = "{payload["published_at"]}"')
    print("LLAMA_CPP_RELEASE_ASSETS: dict[tuple[str, str, str], tuple[tuple[str, str], ...]] = {")
    for os_name, arch, backend in policies.LLAMA_CPP_RELEASE_ASSETS:
        try:
            package = package_from_release(release, os_name, arch, ComputeBackend(backend))
        except UnsupportedPlatformError as exc:
            sys.stderr.write(f"{exc}\n")
            return 1
        print(f'    ("{os_name}", "{arch}", "{backend}"): (')
        for asset in package.assets:
            print(f'        ("{asset.name}",\n         "{asset.sha256}"),')
        print("    ),")
    print("}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
