"""Shared fixtures: stub-engine environment + bridge cache hygiene.

The bridge memoizes ``--help`` output (per argv) and staleness verdicts (30 s TTL)
in-process; tests flip DSP_* env vars per test, so both caches are cleared around
every test via the autouse fixture below.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cxx_bridge import engine_cli

TESTS_DIR = Path(__file__).resolve().parent
STUB_ENGINE_PATH = TESTS_DIR / "stub_engine.py"


@pytest.fixture(autouse=True)
def _fresh_bridge_caches():
    engine_cli.clear_caches()
    yield
    engine_cli.clear_caches()


@pytest.fixture
def stub_engine_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the bridge at tests/stub_engine.py and sandbox data/ under tmp_path.

    Returns the per-test data directory (dumps/logs/plots/cache live beneath it).
    """
    data_dir = tmp_path / "data"
    monkeypatch.setenv("DSP_ENGINE_CLI", str(STUB_ENGINE_PATH))
    monkeypatch.setenv("DSP_DATA_DIR", str(data_dir))
    return data_dir
