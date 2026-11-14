from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def _isolated_user_dirs(tmp_path_factory: pytest.TempPathFactory,
                        monkeypatch: pytest.MonkeyPatch) -> None:
    """No test reads or writes the developer's own Jaull data or caches.

    Stores built without an explicit root resolve through ``jaull.paths``, which
    reads these variables first. On a machine with real runs, reading them made
    tests slow enough to time out (a guided search went from 9 s to 16 s) and
    their results machine-bound. A test that sets its own value still wins.
    """
    base = tmp_path_factory.mktemp("jaull-user")
    monkeypatch.setenv("XDG_DATA_HOME", str(base / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(base / "cache"))
    monkeypatch.setenv("LOCALAPPDATA", str(base / "local"))


@pytest.fixture(autouse=True)
def _isolated_drm(tmp_path_factory: pytest.TempPathFactory,
                  monkeypatch: pytest.MonkeyPatch) -> None:
    """No test sees the GPUs of the machine it runs on through Linux DRM sysfs.

    The kernel listing adds any card no other probe reported, so on a Linux PC
    with a GPU the hardware tests would otherwise depend on that PC.
    """
    monkeypatch.setattr("jaull.hardware.linux_drm.DRM_ROOT",
                        tmp_path_factory.mktemp("drm-empty"))
