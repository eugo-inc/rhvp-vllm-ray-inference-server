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
    verdicts = {row["case"]: row["verdict"] for row in data["analysis"]["comparisons"]}
    assert verdicts["c8/itl_ms_median"] == "FASTER" and verdicts["c1/ttft_ms_median"] == "TIE"
    assert verdicts["c32/tokens_per_joule[gpu]"] == "BETTER" and verdicts["c32/output_tokens_per_s"] == "BETTER"
    assert data["analysis"]["rule"]["plain"] == "corrected for testing 4 cases at once"
    assert len(data["launches"]) == len(data["plan"]["launches"]) == 34


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
    assert "UNRESOLVED" in verdicts and words.REASON_TEXT["not_significant"] in verdicts
    assert re.search(r"\b[01]\.\d{3}\b", verdicts) and re.search(r"±\d+\.\d%", verdicts)  # B/A and its bar
    assert "corrected for testing 4 cases at once" in verdicts
    assert "loading the weights" in _visible(shown["s2-show"])
    # Secondary verdicts (time to ready; tokens per joule below the top level) are marked descriptive.
    assert words.SECONDARY_NOTE in _visible(shown["s2-show"]) and words.SECONDARY_NOTE in _visible(shown["s3-show"])
    assert words.SECONDARY_TITLE in verdicts
    energy = _visible(shown["s3-show"])
    assert "DCGM" in energy and "BETTER" in energy and "GPU-only power" in energy
    assert words.SCALE_OFF in _visible(shown["s4-show"])  # a levels run has no scale round
    provenance = _visible(shown["s5-show"])
    assert "registry.example/inference-server@sha256:" in provenance and "A~same" in provenance
    assert words.TWIN_NOTE in provenance and "concurrent streams" in provenance


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


def test_a_results_file_of_the_earlier_rule_is_skipped_with_a_notice(tmp_path, monkeypatch):
    for name, schema in (("old-run", "rhvp-image-compare/1"), ("new-run", None)):
        (tmp_path / name).mkdir()
        data = json.loads(SYNTHETIC.read_text())
        if schema:
            data["schema"] = schema
        (tmp_path / name / "results.json").write_text(json.dumps(data))
    monkeypatch.delenv("RHVP_COMPARE_REPLAY", raising=False)
    monkeypatch.setenv("RHVP_COMPARE_RESULTS", str(tmp_path))
    found, problems = show.load_all()
    assert len(found) == 1 and found[0]["schema"] == "rhvp-image-compare/2"
    assert problems == [words.OLD_SCHEMA.format(schema="rhvp-image-compare/1")]


# ---- mode scaleout ----

def _scored(results: dict, of: str | None = None, twin: bool = False) -> list[dict]:
    """The scored launches of `results`; with `of`, those of that side (its twin's too with `twin`)."""
    return [x for x in results["launches"] if x["kind"] == "scored" and (
        of is None or x["side"] == of or (twin and x["of"] == of))]


def test_a_synthetic_scale_out_run_is_valid():
    data = example.synthesize_scale_out()
    jsonschema.validate(data, SCHEMA)
    assert data["run"]["synthetic"] is True and data["plan"]["mode"] == "scaleout"
    assert data["plan"]["load"]["concurrencies"] == [] and data["plan"]["energy"]["source"] == "none"
    assert data == json.loads(example.text(example.synthesize_scale_out()))
    assert all(len(x["servers"]) == x["starts"] == 2 and x["scale"]["new_replica_share"] > 0 for x in _scored(data))
    assert len(_scored(data)) == 32 and {x["side"] for x in _scored(data)} == {"A", "A~same", "B", "B~same"}
    warm = [x for x in data["launches"] if x["kind"] == "warmup"]
    assert len(warm) == 2 and all(len(x["servers"]) == x["starts"] == 1 and "scale" not in x for x in warm)
    rows = {row["case"]: row for row in data["analysis"]["comparisons"]}
    assert rows["scale/first_answer_s"]["verdict"] == "FASTER" and rows["surge/slo_attainment"]["verdict"] == "BETTER"
    # The made-up servers log no rhvp-scale line, so that case has no figures at all.
    assert rows["scale/load_triggered_s"]["verdict"] == "NO VERDICT"


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
    assert "corrected for testing 2 cases at once" in verdicts
    assert words.ENERGY_OFF in _visible(shown["s3-show"])
    scale = _visible(shown["s4-show"])
    assert "Surge: 6 requests per second for 120 s" in scale and "30 s after the surge's first request" in scale
    assert "time to first token at most 500 ms and inter-token latency at most 50 ms" in scale
    assert words.SCALE_SHARE in scale and words.SCALE_REQUESTS_NOTE in scale and words.SCALE_SURGE_RULE in scale
    assert words.SCALE_GPU_SHARED in scale and words.SCALE_GPU_NONE not in scale
    assert "For a MIG device the UUID is its parent GPU's." in scale
    uuid = "GPU-00000000-0000-0000-0000-000000000000"
    assert f"replica 1: 0 ({uuid}); replica 2: 0 ({uuid})" in scale  # each replica's request, with its UUIDs
    assert words.SCALE_EVENTS.format(a=0, na=8, b=0, nb=8) in scale  # each side's own 8 launches, not its twin's
    assert "loading the weights" in scale  # the second replica's start phases
    assert words.SCALE_OFF not in scale and words.SCALE_LATE.split(":")[0] not in scale
    provenance = _visible(shown["s5-show"])
    assert "6 requests per second for 120 s in each scored launch" in provenance
    assert "concurrent streams" not in provenance and "66 server starts" in provenance


def _shown(monkeypatch, results: dict, call) -> str:
    """The visible text of one show.* call on `results`, without a kernel."""
    out = []
    monkeypatch.setattr(show, "pick", lambda run=None: (results, [results], []))
    monkeypatch.setattr(show.draw, "show", lambda *parts: out.extend(parts))
    call()
    return _visible("".join(out))


def test_a_surge_left_out_and_a_late_replica_are_named(monkeypatch):
    results = example.synthesize_scale_out()
    late = _scored(results, "B")[0]
    late["scale"]["new_replica"]["joined_during_surge"] = False
    failed = _scored(results, "A")[0]
    failed["levels"][0].update(usable=False, note="some requests failed",
                               requests={"successful": 40, "errored": 3, "incomplete": 0})
    text = _shown(monkeypatch, results, show.scale_out)
    assert words.SCALE_LATE.format(launches=late["index"]) in text
    assert "None streams" not in text
    markdown = report.markdown(results)
    assert "The second replica" in markdown and "| B | 8 |" in markdown  # side B's own 8 launches
    head = _shown(monkeypatch, results, show.verdicts)
    assert words.RUN_SURGE.format(index=failed["index"], side="A", errored=3, successful=40) in head


def test_a_surge_with_failed_requests_is_not_said_to_be_left_out(monkeypatch):
    # Its share within the goals and goodput still count (failed requests as misses); only its latency medians go.
    results = example.synthesize_scale_out()
    failed = _scored(results, "A")[0]
    failed["levels"][0].update(usable=False, requests={"successful": 40, "errored": 3, "incomplete": 0})
    text = _shown(monkeypatch, results, show.verdicts)
    assert "Load levels left out" not in text
    assert words.RUN_SURGES.split(":")[0] in text and "Only their latency medians are left out" in text
    markdown = report.markdown(results)
    assert "Load levels left out of the comparison" not in markdown
    assert f"- launch {failed['index']} (A): 3 requests failed, 40 succeeded" in markdown
    assert "Only their latency medians are left out of the comparison" in markdown
    assert "counts as one that missed the goals in the share within the goals and in goodput" in markdown


def _surges(results):
    return [lv for x in _scored(results) for lv in x["levels"]]


@pytest.mark.parametrize("case", ["all_met", "all_failed", "all_missed", "one_failed_each", "mixed"])
def test_a_surge_rate_that_did_not_separate_the_sides_is_named(monkeypatch, case):
    results = example.synthesize_scale_out()
    for i, level in enumerate(_surges(results)):
        if case == "all_met":
            level.update(slo_attainment=1.0, requests={"successful": 300, "errored": 0, "incomplete": 0})
        elif case == "all_failed" or (case == "mixed" and i % 2):  # half of the judged requests failed
            level.update(usable=False, slo_attainment=0.2,
                         requests={"successful": 150, "errored": 150, "incomplete": 0})
        elif case == "all_missed":  # as in a real overloaded run: about 17% failed, nearly every finished one late
            level.update(usable=False, slo_attainment=0.008, requests={"successful": 3700, "errored": 750,
                                                                         "incomplete": 0})
        elif case == "one_failed_each":  # one failed request per surge says nothing about the rate
            level.update(usable=False, requests={"successful": 299, "errored": 1, "incomplete": 0})
    expected = case if case in words.SCALE_RATE_NOTES else None
    assert summary.surge_rate_fit(results) == expected
    text = _shown(monkeypatch, results, show.scale_out)
    markdown = report.markdown(results)
    for fit, shown in words.SCALE_RATE_NOTES.items():
        head = shown.split(" (README")[0]
        assert (head in text) == (fit == expected) and (head in markdown) == (fit == expected), fit


def test_results_md_shows_each_replicas_gpu_request_with_its_uuids():
    results = example.synthesize_scale_out()
    _scored(results)[0]["servers"][1].update(gpus="1", gpu_uuids=["GPU-1111"])
    _scored(results, "B")[0]["servers"][1].update(gpus="1", gpu_uuids=None)
    markdown = report.markdown(results)
    # First seen first: launch 3 (A), then launch 4 (A~same), then launch 5 (B).
    assert ("replica 1: `0` (`GPU-00000000-0000-0000-0000-000000000000`); replica 2: `1` (`GPU-1111`) or `0` "
            "(`GPU-00000000-0000-0000-0000-000000000000`) or `1` (UUID not read).") in markdown
    assert "with the UUIDs nvidia-smi read through each request before the first launch" in markdown
    assert "For a MIG device the UUID is its parent GPU's." in markdown


def test_the_surge_rate_thresholds_are_the_ones_the_notes_state():
    assert summary.SURGE_FAILED_SHARE == 0.5 and "at least half of the requests" in words.SCALE_RATE_ALL_FAILED
    assert summary.SURGE_MET_SHARE_LOW == 0.05 and "at most 5% of the requests" in words.SCALE_RATE_ALL_MISSED
    results = example.synthesize_scale_out()
    for level in _surges(results):  # just under half failed in every surge, and more than 5% met the goals
        level.update(usable=False, slo_attainment=0.06, requests={"successful": 151, "errored": 149, "incomplete": 0})
    assert summary.surge_rate_fit(results) is None


def test_the_surge_rate_note_needs_a_finished_surge_on_each_side():
    results = example.synthesize_scale_out()
    for x in _scored(results):
        x["levels"][0].update(slo_attainment=1.0, requests={"successful": 300, "errored": 0, "incomplete": 0})
        if x["of"] == "B":  # side B and its twin
            x["status"] = "failed"
    assert summary.surge_rate_fit(results) is None


def test_the_scale_out_cell_with_no_forced_step_and_no_gpu(monkeypatch):
    results = example.synthesize_scale_out()
    for x in results["launches"]:
        x.pop("scale", None)
    for side in results["plan"]["sides"]:
        side["gpu"] = False
    text = _shown(monkeypatch, results, show.scale_out)
    assert words.SCALE_NONE in text and words.SCALE_SHARE not in text
    assert words.SCALE_GPU_NONE in text and words.SCALE_GPU_SHARED not in text


def test_sides_with_different_compile_caches_are_named_next_to_the_primary_cases(monkeypatch):
    # A warm cache alone shortens a start, and the start-up times are primary cases: a run whose two sides differ
    # in compileCache says so in results.md and the notebook, next to the verdicts, and both show each side's setting.
    for results in (example.synthesize(), example.synthesize_scale_out()):
        results["plan"]["sides"][0]["compile_cache"] = "cold"
        results["plan"]["sides"][1]["compile_cache"] = "warm"
        markdown = report.markdown(results)
        assert "| Side | Name | Image | Digest | Server version | Compile cache |" in markdown
        assert re.search(r"^\| A \|.*\| cold \|$", markdown, re.MULTILINE)
        assert re.search(r"^\| B \|.*\| warm \|$", markdown, re.MULTILINE)
        primary, rest = markdown.split("Primary cases", 1)[1].split("Secondary cases", 1)
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
    assert "Both sides started their scored launches with warm compile caches" in report.markdown(results)
    assert words.CACHE_BOTH_WARM in _shown(monkeypatch, results, show.verdicts)
