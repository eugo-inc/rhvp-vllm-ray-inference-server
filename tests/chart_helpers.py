"""Shared by the chart tests (tests/conftest.py, test_charts.py, test_server_chart.py, test_scaleout_contract.py):
paths, the utility container pin, and small helpers for rendered YAML. Not a test module."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
# Same pin as scripts/render-charts.sh: the v1.0.5 release digest (a latest digest is deleted on rebuild).
UTILITY_IMAGE = (
    "quay.io/validatedpatterns/utility-container@sha256:"
    "556083f92b08642868ca64ade6cf9553295bd394f7a54d5e10ca12d143907214"
)
GLOBAL = yaml.safe_load((REPO / "values-global.yaml").read_text())
READY_FILE = ["test", "-f", "/tmp/r/rhvp-ready"]


def container_engine() -> str | None:
    for engine in ("podman", "docker"):
        if shutil.which(engine):
            probe = subprocess.run([engine, "info"], capture_output=True, timeout=60)
            if probe.returncode == 0:
                return engine
    return None


def docs(path: Path) -> list[dict]:
    return [d for d in yaml.safe_load_all(path.read_text()) if d]


def one(items: list[dict], kind: str, name: str | None = None) -> dict:
    found = [d for d in items if d["kind"] == kind and (name is None or d["metadata"]["name"] == name)]
    assert len(found) == 1, f"expected one {kind} {name or ''}, found {len(found)}"
    return found[0]


def kinds(items: list[dict]) -> set[str]:
    return {d["kind"] for d in items}


def container(deploy: dict, name: str) -> dict:
    [c] = [c for c in deploy["spec"]["template"]["spec"]["containers"] if c["name"] == name]
    return c


def env_map(c: dict) -> dict:
    return {e["name"]: e.get("value", e.get("valueFrom")) for e in c.get("env", [])}


def quantity(value: str) -> float:
    units = {"Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40, "k": 1e3, "M": 1e6, "G": 1e9}
    match = re.fullmatch(r"([0-9.]+)([A-Za-z]*)", str(value))
    assert match, value
    return float(match.group(1)) * units.get(match.group(2), 1)
