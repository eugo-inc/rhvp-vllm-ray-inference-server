# SPDX-License-Identifier: Apache-2.0
"""The comparison notebook (compare_images.ipynb): generated and unchanged, no outputs, the layout, the wording rules,
the synthetic results file against the schema, and headless runs in REPLAY mode, on a results volume, and with no
results at all; and a run in mode scaleout, in the notebook and in results.md.

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

from rhvp_compare import cells, example, report, show, summary, words  # noqa: E402
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
                    "show.energy()", "show.scale_out()", "show.provenance()"]


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
    assert set(shown) == {"runs", "s1-show", "s2-show", "s3-show", "s4-show", "s5-show"}
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
    assert words.SCALE_OFF in _visible(shown["s4-show"])  # a pairs run has no scale round
    provenance = _visible(shown["s5-show"])
    assert "registry.example/inference-server@sha256:" in provenance and "A/A pairs" in provenance
    assert "concurrent streams" in provenance


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


# ---- mode scaleout ----

def test_a_synthetic_scale_out_run_is_valid():
    data = example.synthesize_scale_out()
    jsonschema.validate(data, SCHEMA)
    assert data["run"]["synthetic"] is True and data["plan"]["mode"] == "scaleout"
    assert data["plan"]["load"]["concurrencies"] == [] and data["plan"]["energy"]["source"] == "none"
    scored = [r for r in data["rounds"] if r["kind"] == "scored"]
    assert all(len(r["launches"]) == 2 and r["scale"]["new_replica_share"] > 0 for r in scored)
    assert all(len(r["launches"]) == 1 and "scale" not in r for r in data["rounds"] if r["kind"] == "warmup")
    verdicts = {row["id"]: row["verdict"] for row in data["analysis"]["metrics"]}
    assert verdicts["scale.first_answer_s"] == "FASTER" and verdicts["surge.slo_attainment"] == "FASTER"
    assert verdicts["scale.load_triggered_s"] == "INCONCLUSIVE"  # the made-up servers log no rhvp-scale line


def test_replay_run_in_mode_scaleout(tmp_path):
    path = tmp_path / "scale-out.json"
    path.write_text(example.text(example.synthesize_scale_out()), encoding="utf-8")
    shown = _execute(tmp_path / "run", {"RHVP_COMPARE_REPLAY": str(path)})
    for cell_id, markup in shown.items():
        text = _visible(markup)
        assert words.SYNTHETIC_STRIP in text, cell_id
        assert not VENDOR.search(text) and not HARDWARE.search(text) and not CAUSE.search(text), cell_id
    verdicts = _visible(shown["s1-show"])
    assert "new replica: time to first answer" in verdicts and "surge: share of requests within the latency goals" \
        in verdicts
    assert re.search(r"\b\d{2}\.\d% \b", verdicts)  # a share is shown as a percentage
    assert words.ENERGY_OFF in _visible(shown["s3-show"])
    scale = _visible(shown["s4-show"])
    assert "Surge: 6 requests per second for 120 s" in scale and "30 s after the surge's first request" in scale
    assert "time to first token at most 500 ms and inter-token latency at most 50 ms" in scale
    assert words.SCALE_SHARE in scale and words.SCALE_REQUESTS_NOTE in scale and words.SCALE_SURGE_RULE in scale
    assert words.SCALE_GPU_SHARED in scale and words.SCALE_GPU_NONE not in scale
    assert words.SCALE_EVENTS.format(a=0, na=8, b=0, nb=8) in scale
    assert "loading the weights" in scale  # the second replica's start phases
    assert words.SCALE_OFF not in scale and words.SCALE_LATE.split(":")[0] not in scale
    provenance = _visible(shown["s5-show"])
    assert "6 requests per second for 120 s in each round" in provenance and "concurrent streams" not in provenance


def _shown(monkeypatch, results: dict, call) -> str:
    """The visible text of one show.* call on `results`, without a kernel."""
    out = []
    monkeypatch.setattr(show, "pick", lambda run=None: (results, [results], []))
    monkeypatch.setattr(show.draw, "show", lambda *parts: out.extend(parts))
    call()
    return _visible("".join(out))


def test_a_surge_left_out_and_a_late_replica_are_named(monkeypatch):
    results = example.synthesize_scale_out()
    late = next(r for r in results["rounds"] if r["kind"] == "scored" and r["side"] == "B")
    late["scale"]["new_replica"]["joined_during_surge"] = False
    failed = next(r for r in results["rounds"] if r["kind"] == "scored" and r["side"] == "A")
    failed["levels"][0].update(usable=False, note="some requests failed",
                               requests={"successful": 40, "errored": 3, "incomplete": 0})
    text = _shown(monkeypatch, results, show.scale_out)
    assert words.SCALE_LATE.format(rounds=late["index"]) in text
    assert words.LEVEL_SURGE.format(index=failed["index"], side="A", errored=3, successful=40) in text
    assert "None streams" not in text
    markdown = report.markdown(results)
    assert "The second replica" in markdown and "| B | 8 |" in markdown


def test_a_surge_with_failed_requests_is_not_said_to_be_left_out(monkeypatch):
    # Its share within the goals and goodput still count (failed requests as misses); only its latency medians go.
    results = example.synthesize_scale_out()
    failed = next(r for r in results["rounds"] if r["kind"] == "scored" and r["side"] == "A")
    failed["levels"][0].update(usable=False, requests={"successful": 40, "errored": 3, "incomplete": 0})
    text = _shown(monkeypatch, results, show.verdicts)
    assert "Load levels left out" not in text
    assert words.RUN_SURGES.split(":")[0] in text and "Only their latency medians are left out" in text
    markdown = report.markdown(results)
    assert "Load levels left out of the comparison" not in markdown
    assert f"- round {failed['index']} (side A, scored): 3 requests failed, 40 succeeded" in markdown
    assert "Only their latency medians are left out of the comparison" in markdown
    assert "counts as one that missed the goals in the share within the goals and in goodput" in markdown


def _surges(results):
    return [lv for r in results["rounds"] if r["kind"] == "scored" for lv in r["levels"]]


@pytest.mark.parametrize("case", ["all_met", "all_failed", "mixed"])
def test_a_surge_rate_that_did_not_separate_the_sides_is_named(monkeypatch, case):
    results = example.synthesize_scale_out()
    for i, level in enumerate(_surges(results)):
        if case == "all_met":
            level.update(slo_attainment=1.0, requests={"successful": 300, "errored": 0, "incomplete": 0})
        elif case == "all_failed" or (case == "mixed" and i % 2):
            level.update(usable=False, requests={"successful": 250, "errored": 50, "incomplete": 0})
    note = {"all_met": words.SCALE_RATE_ALL_MET, "all_failed": words.SCALE_RATE_ALL_FAILED}.get(case)
    assert summary.surge_rate_fit(results) == (None if case == "mixed" else case)
    text = _shown(monkeypatch, results, show.scale_out)
    markdown = report.markdown(results)
    for shown in (words.SCALE_RATE_ALL_MET, words.SCALE_RATE_ALL_FAILED):
        head = shown.split(" (README")[0]
        assert (head in text) == (shown == note) and (head in markdown) == (shown == note)


def test_the_surge_rate_note_needs_a_finished_surge_on_each_side():
    results = example.synthesize_scale_out()
    for r in results["rounds"]:
        if r["kind"] == "scored":
            r["levels"][0].update(slo_attainment=1.0, requests={"successful": 300, "errored": 0, "incomplete": 0})
            if r["side"] == "B":
                r["status"] = "failed"
    assert summary.surge_rate_fit(results) is None


def test_the_scale_out_cell_with_no_forced_step_and_no_gpu(monkeypatch):
    results = example.synthesize_scale_out()
    for r in results["rounds"]:
        r.pop("scale", None)
    for side in results["plan"]["sides"]:
        side["gpu"] = False
    text = _shown(monkeypatch, results, show.scale_out)
    assert words.SCALE_NONE in text and words.SCALE_SHARE not in text
    assert words.SCALE_GPU_NONE in text and words.SCALE_GPU_SHARED not in text


def test_sides_with_different_compile_caches_are_named_next_to_the_primary_metrics(monkeypatch):
    # A warm cache alone shortens a start, and the start-up times are primary metrics: a run whose two sides differ
    # in compileCache says so in results.md and the notebook, next to the verdicts, and both show each side's setting.
    for results in (example.synthesize(), example.synthesize_scale_out()):
        results["plan"]["sides"][0]["compile_cache"] = "cold"
        results["plan"]["sides"][1]["compile_cache"] = "warm"
        markdown = report.markdown(results)
        assert "| Side | Name | Image | Digest | Server version | Compile cache |" in markdown
        assert re.search(r"^\| A \|.*\| cold \|$", markdown, re.MULTILINE)
        assert re.search(r"^\| B \|.*\| warm \|$", markdown, re.MULTILINE)
        primary, rest = markdown.split("Primary metrics", 1)[1].split("Secondary metrics", 1)
        assert "different compile caches: side A cold, side B warm" in primary, primary
        assert "different compile caches" not in rest
        differ = words.CACHE_DIFFER.format(a="cold", b="warm")
        assert differ in _shown(monkeypatch, results, show.verdicts)
        assert differ in _shown(monkeypatch, results, show.startup)
        provenance = _shown(monkeypatch, results, show.provenance)
        assert "Compile cache" in provenance and "warm" in provenance
    scale = _shown(monkeypatch, results, show.scale_out)
    assert words.CACHE_DIFFER.format(a="cold", b="warm") in scale


def test_sides_with_the_same_compile_cache(monkeypatch):
    results = example.synthesize()
    assert "compile caches" not in report.markdown(results)  # both cold, the default: nothing to say
    assert "compile caches" not in _shown(monkeypatch, results, show.verdicts)
    for side in results["plan"]["sides"]:
        side["compile_cache"] = "warm"
    assert "Both sides started their scored rounds with warm compile caches" in report.markdown(results)
    assert words.CACHE_BOTH_WARM in _shown(monkeypatch, results, show.verdicts)
