# SPDX-License-Identifier: Apache-2.0
"""The comparison notebook (compare_images.ipynb): generated and unchanged, no outputs, the layout, the wording rules,
the synthetic results file against the schema, and headless runs in REPLAY mode, on a results volume, and with no
results at all.

Run with: python -m pytest -q tests/test_compare_notebook.py
"""

import json
import re
import shutil
import sys
from pathlib import Path

import jsonschema
import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
NOTEBOOK = NOTEBOOKS / "compare_images.ipynb"
SYNTHETIC = NOTEBOOKS / "compare-recordings" / "synthetic.json"
sys.path.insert(0, str(NOTEBOOKS))
sys.path.insert(0, str(ROOT / "tests"))

from rhvp_compare import cells, example, words  # noqa: E402
from test_notebook_rules import CAUSE, HARDWARE, VENDOR, _strings, _visible  # noqa: E402

SCHEMA = json.loads((NOTEBOOKS / "rhvp_compare" / "results.schema.json").read_text())


def compare_word_strings() -> list[str]:
    out = [words.__doc__ or ""]
    for name in dir(words):
        if not name.startswith("_"):
            out.extend(_strings(getattr(words, name)))
    return out


def test_generated_without_outputs_and_with_stable_ids():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    assert [c.id for c in nb.cells] == [c.id for c in cells.cells()]
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert cell.outputs == [] and cell.execution_count is None
    assert nb.cells[0].source.startswith(f"# {words.TITLE}")
    code = [c.source for c in nb.cells if c.cell_type == "code"]
    assert code == ["from rhvp_compare import show\n\nshow.runs()", "show.verdicts()", "show.startup()",
                    "show.energy()", "show.provenance()"]


def test_wording_rules():
    texts = compare_word_strings() + [c.source for c in cells.cells()]
    for text in texts:
        assert not CAUSE.search(text), text
        assert not VENDOR.search(text), text
        assert not HARDWARE.search(text), text
    assert "GPU-only power" in words.SECTION_TEXT[3]


def test_synthetic_results_are_flagged_and_valid():
    data = json.loads(SYNTHETIC.read_text())
    jsonschema.validate(data, SCHEMA)
    assert data["run"]["synthetic"] is True
    assert all(s["image"].startswith("registry.example/") for s in data["plan"]["sides"])
    assert data == json.loads(example.text(example.synthesize()))
    verdicts = {row["id"]: row["verdict"] for row in data["analysis"]["metrics"]}
    assert verdicts["c8.itl_ms_median"] == "FASTER" and verdicts["startup.ready_s"] == "TIE"
    assert verdicts["c32.tokens_per_joule"] == "MORE_EFFICIENT"


def _execute(tmp_path, env: dict[str, str]):
    from nbclient import NotebookClient

    work = tmp_path / "notebooks"
    shutil.copytree(NOTEBOOKS, work, ignore=shutil.ignore_patterns("__pycache__", ".ipynb_checkpoints"))
    kernels = tmp_path / "jupyter" / "kernels" / "rhvp-test"
    kernels.mkdir(parents=True)
    (kernels / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-Xfrozen_modules=off", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "rhvp test", "language": "python"}), encoding="utf-8")
    patch = pytest.MonkeyPatch()
    patch.setenv("JUPYTER_PATH", str(tmp_path / "jupyter"))
    for name in ("RHVP_COMPARE_REPLAY", "RHVP_COMPARE_RESULTS"):
        patch.delenv(name, raising=False)
    for name, value in env.items():
        patch.setenv(name, value.replace("{work}", str(work)))
    try:
        nb = nbformat.read(work / NOTEBOOK.name, as_version=4)
        NotebookClient(nb, kernel_name="rhvp-test", timeout=120, resources={"metadata": {"path": str(work)}}).execute()
    finally:
        patch.undo()
    shown = {}
    for cell in nb.cells:
        for output in cell.get("outputs", []):
            assert output.output_type in ("display_data", "update_display_data"), (cell.id, output)
            shown[cell.id] = shown.get(cell.id, "") + output.data["text/html"]
    return shown


def test_replay_run(tmp_path):
    shown = _execute(tmp_path, {"RHVP_COMPARE_REPLAY": "{work}/compare-recordings/synthetic.json"})
    assert set(shown) == {"runs", "s1-show", "s2-show", "s3-show", "s4-show"}
    for cell_id, markup in shown.items():
        text = _visible(markup)
        assert text.startswith(words.REPLAY_STRIP), cell_id
        assert words.SYNTHETIC_STRIP in text, cell_id
        assert not VENDOR.search(text) and not HARDWARE.search(text) and not CAUSE.search(text), cell_id
    verdicts = _visible(shown["s1-show"])
    assert "inter-token latency, median, 8 concurrent" in verdicts and "FASTER" in verdicts and "TIE" in verdicts
    assert re.search(r"0\.\d{3} \[0\.\d{3}, 0\.\d{3}\]", verdicts)
    assert "loading the weights" in _visible(shown["s2-show"])
    # Secondary verdicts (time to ready; tokens per joule below the top level) are marked descriptive.
    assert words.SECONDARY_NOTE in _visible(shown["s2-show"]) and words.SECONDARY_NOTE in _visible(shown["s3-show"])
    assert words.SECONDARY_TITLE in verdicts
    assert "DCGM" in _visible(shown["s3-show"]) and "MORE_EFFICIENT" in _visible(shown["s3-show"])
    provenance = _visible(shown["s4-show"])
    assert "registry.example/inference-server@sha256:" in provenance and "A/A pairs" in provenance


def test_run_from_a_results_volume_and_from_an_empty_one(tmp_path):
    volume = tmp_path / "volume"
    (volume / "synthetic-example").mkdir(parents=True)
    shutil.copy(SYNTHETIC, volume / "synthetic-example" / "results.json")
    shown = _execute(tmp_path / "a", {"RHVP_COMPARE_RESULTS": str(volume)})
    text = _visible(shown["s1-show"])
    assert words.REPLAY_STRIP not in text and words.SYNTHETIC_STRIP in text and "FASTER" in text
    empty = tmp_path / "empty"
    empty.mkdir()
    shown = _execute(tmp_path / "b", {"RHVP_COMPARE_RESULTS": str(empty)})
    assert all(words.NO_RESULTS in _visible(markup) for markup in shown.values())
