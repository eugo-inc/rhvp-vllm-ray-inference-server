"""Fixtures shared by the chart tests: scripts/render-charts.sh runs once per test session (helm inside the
Validated Patterns utility container). They are skipped when neither docker nor podman is available, or when
charts.validatedpatterns.io cannot be reached.

pytest loads this file for every test run under tests/, also the notebook checks, whose environment has no PyYAML;
so chart_helpers (which needs it) is imported inside the fixtures only."""

from __future__ import annotations

import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def render(tmp_path_factory) -> Path:
    from chart_helpers import REPO, container_engine

    if container_engine() is None:
        pytest.skip("needs docker or podman")
    try:
        urllib.request.urlopen("https://charts.validatedpatterns.io/index.yaml", timeout=30).close()
    except (urllib.error.URLError, TimeoutError) as err:
        pytest.skip(f"charts.validatedpatterns.io unreachable: {err}")
    out = tmp_path_factory.mktemp("render")
    proc = subprocess.run(
        [str(REPO / "scripts/render-charts.sh")],
        env={**os.environ, "RENDER_DIR": str(out)},
        capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    return out


@pytest.fixture(scope="session")
def server(render) -> list[dict]:
    from chart_helpers import docs

    return docs(render / "charts/ray-serve-llm.yaml")
