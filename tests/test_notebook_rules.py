# SPDX-License-Identifier: Apache-2.0
"""The sample notebook's rules: generated and unchanged, no outputs, the layout, the stock server only, the wording
rules, the baseline label, and a headless REPLAY run that renders it.

Run with: python -m pytest -q tests/test_notebook_rules.py
Every guard here has a planted control that must trip it.
"""

import html
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
NOTEBOOK = NOTEBOOKS / "inference_timings.ipynb"
MAKE = NOTEBOOKS / "make_notebook.py"
SERVER = ROOT / "inference-server"
sys.path.insert(0, str(NOTEBOOKS))

from rhvp_bench import draw, showing, words  # noqa: E402

CAUSE_PHRASES = ("because", "due to", "thanks to", "as a result", "owing to", "which is why", "the reason", "so that")
CAUSE = re.compile(r"\b(?:" + "|".join(p.replace(" ", r"\s+") for p in CAUSE_PHRASES) + r")\b", re.IGNORECASE)
RATIO = re.compile(r"×|\b\d+(?:\.\d+)?\s?x\b|\bx\s+(?:faster|slower)\b|\btimes\s+(?:faster|slower|quicker|as\s+fast)\b",
                   re.IGNORECASE)
#: GPU and machine names that never appear in this repository's notebook, written as character classes so that this
#: file does not contain them either.
HARDWARE = re.compile(r"sm_?1[2]|G[B]\s?10|D[G]X", re.IGNORECASE)
#: The maintainer's company name: the notebook times the stock server only and never names it. Written as a character
#: class so that this file does not contain it either.
VENDOR = re.compile(r"e[u]go", re.IGNORECASE)
ADDRESS = re.compile(r"https?://|\.svc\b|\blocalhost\b|\b\d{1,3}(?:\.\d{1,3}){3}\b|:\d{4,5}\b", re.IGNORECASE)
TAGS = re.compile(r"<[^>]+>")


def _visible(markup: str) -> str:
    """The text a reader sees in some HTML, whitespace collapsed."""
    return " ".join(html.unescape(TAGS.sub(" ", markup)).split())


def _strings(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(key)
            yield from _strings(item)
    elif isinstance(value, list | tuple | set | frozenset):
        for item in value:
            yield from _strings(item)


def word_strings() -> list[str]:
    """Every string in words.py, its docstring included."""
    out = [words.__doc__ or ""]
    for name in dir(words):
        if not name.startswith("_"):
            out.extend(_strings(getattr(words, name)))
    return out


def _notebook() -> nbformat.NotebookNode:
    return nbformat.read(NOTEBOOK, as_version=4)


def _markdown_sources() -> list[str]:
    return [cell.source for cell in _notebook().cells if cell.cell_type == "markdown"]


def _sections(nb: nbformat.NotebookNode) -> list[tuple[int | None, nbformat.NotebookNode]]:
    """Each cell with the number of the section it is in (None before the first section)."""
    out, number = [], None
    for cell in nb.cells:
        match = re.match(r"## (\d+) · ", cell.source) if cell.cell_type == "markdown" else None
        if match:
            number = int(match.group(1))
        out.append((number, cell))
    return out


def _run_make(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(MAKE), *args], capture_output=True, text=True, check=False)


# ---- generated, unchanged, no outputs ----

def test_check_passes():
    done = _run_make("--check")
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.parametrize("which", ["--out", "--prompts", "--recording"])
def test_check_fails_on_a_changed_file(tmp_path, which):
    """Planted control: a copy with one change must make --check fail."""
    sources = {"--out": NOTEBOOK, "--prompts": NOTEBOOKS / "prompts.jsonl",
               "--recording": NOTEBOOKS / "recordings" / "synthetic.json"}
    paths = {}
    for flag, source in sources.items():
        paths[flag] = tmp_path / source.name
        shutil.copy(source, paths[flag])
    paths[which].write_text(paths[which].read_text(encoding="utf-8").replace("e", "E", 1), encoding="utf-8")
    flags = [item for flag, path in paths.items() for item in (flag, str(path))]
    done = _run_make("--check", *flags)
    assert done.returncode == 1
    assert "STALE" in done.stdout


def test_notebook_is_valid_nbformat_4_5_in_jupyters_own_layout():
    nb = _notebook()
    nbformat.validate(nb)
    assert (nb.nbformat, nb.nbformat_minor) == (4, 5)
    assert nb.metadata.kernelspec.name == "python3"
    assert nbformat.writes(nb) + "\n" == NOTEBOOK.read_text(encoding="utf-8")


def test_no_outputs_and_stable_ids():
    nb = _notebook()
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert cell.outputs == [] and cell.execution_count is None, cell.id
    assert [cell.id for cell in nb.cells] == [cell.id for cell in showing.cells()]
    assert len({cell.id for cell in nb.cells}) == len(nb.cells)
    assert "widgets" not in nb.metadata


# ---- layout ----

def test_title_and_readiness_cells_come_first():
    nb = _notebook()
    title, ready = nb.cells[0], nb.cells[1]
    assert title.cell_type == "markdown" and title.source.startswith(f"# {words.TITLE}")
    assert ready.cell_type == "code" and showing.READY_CALL in ready.source
    assert not re.search(r"^## ", ready.source, re.MULTILINE)


def test_sections_in_order_each_a_heading_then_one_short_code_cell():
    nb = _notebook()
    headings = [cell.source.splitlines()[0] for cell in nb.cells
                if cell.cell_type == "markdown" and cell.source.startswith("## ")]
    assert headings == [
        "## 1 · Server start to ready",
        "## 2 · Time to first token",
        "## 3 · Decode speed (one stream)",
        "## 4 · Throughput at 1, 8 and 32 concurrent streams",
    ]
    cells = nb.cells
    assert len(cells) == 2 + 2 * len(headings)
    for number, cell in _sections(nb):
        if number is not None and cell.cell_type == "markdown" and cell.source.startswith(f"## {number} "):
            following = cells[cells.index(cell) + 1]
            assert following.cell_type == "code" and following.source == showing.SECTION_CALL[number]
    sections = [number for number, cell in _sections(nb) if number is not None]
    assert sections == [1, 1, 2, 2, 3, 3, 4, 4]


def test_the_vendor_check_trips_on_planted_controls():
    for planted in ("WITHOUT E" + "UGO", "outside E" + "ugo.", "run.with_e" + "ugo()", "stock_vs_e" + "ugo"):
        assert VENDOR.search(planted), planted
    assert not VENDOR.search(words.INTRO)


def test_the_notebook_times_the_stock_server_only():
    """No vendor name in any text the notebook shows or in the files it is made from (the image's Containerfile
    names the maintainer, and is not part of the notebook), and no side banner to put one in."""
    for text in word_strings() + [cell.source for cell in _notebook().cells]:
        assert not VENDOR.search(text), text
    for path in sorted(NOTEBOOKS.rglob("*")):
        if path.is_file() and path.name != "Containerfile" and "__pycache__" not in path.parts:
            assert not VENDOR.search(path.read_text(encoding="utf-8")), path.relative_to(ROOT)
    assert not hasattr(draw, "side_banner")


def test_every_theme_variable_style_is_written_twice():
    for markup in (draw.notice("a message"), draw.strip("REPLAY"), draw.result_card("l", "1", "s", ["n"]),
                   draw.quiet_line("x")):
        for style in re.findall(r'style="([^"]*)"', markup):
            rules = [r for r in style.split(";") if r]
            for at, rule in enumerate(rules):
                if "var(" in rule:
                    prop = rule.split(":", 1)[0]
                    assert at > 0 and rules[at - 1].startswith(prop + ":") and "var(" not in rules[at - 1], style


# ---- wording ----

def test_the_required_sentences():
    assert words.TITLE == "vLLM + Ray Serve inference server: timings"
    assert words.SINGLE_RUN == "A single run can come out either way."
    assert words.ONE_MEASUREMENT == "One measurement, no p-value."
    assert words.STREAM_GROUPS_NOTE == "Ray Serve LLM sends streamed tokens in groups, every 50 ms by default."
    assert words.BASELINE == ("Stock vLLM 0.29.0 · Ray 2.60.0 · PyTorch 2.13.0 from PyPI · Python 3.12 · NVIDIA CUDA "
                              "13.0.3 UBI 9 image")
    assert words.IN_THIS_RUN == "In this run"


def test_the_cause_check_trips_on_planted_controls():
    for phrase in CAUSE_PHRASES:
        assert CAUSE.search(f"The first side took less time {phrase} its cache was warm."), phrase
    assert not CAUSE.search("Requests sent one at a time.")


def test_no_cause_words_in_any_notebook_text():
    texts = word_strings() + _markdown_sources()
    assert len(texts) > 50
    for text in texts:
        assert not CAUSE.search(text), text


def _readme_notebook_section(readme: str) -> str:
    """The README's "## The notebook" section, up to the next second-level heading."""
    match = re.search(r"^## The notebook\n(.*?)(?=^## )", readme, re.MULTILINE | re.DOTALL)
    assert match, "README.md has no '## The notebook' section"
    return match.group(1)


def test_the_readme_section_check_trips_on_planted_controls():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    planted = readme.replace("## The notebook\n", "## The notebook\nStock took 2.1 s because of 1.5x more work.\n", 1)
    section = _readme_notebook_section(planted)
    assert CAUSE.search(section) and RATIO.search(section)


def test_no_cause_words_or_ratios_in_the_readme_notebook_section():
    section = _readme_notebook_section((ROOT / "README.md").read_text(encoding="utf-8"))
    assert f"`{NOTEBOOK.name}`" in section
    assert not CAUSE.search(section), CAUSE.search(section)
    assert not RATIO.search(section), RATIO.search(section)


def test_the_ratio_check_trips_on_planted_controls():
    for planted in ("1.5x faster", "2× the speed", "3 x", "two times faster"):
        assert RATIO.search(planted), planted
    assert not RATIO.search("1, 8 and 32 concurrent streams · 0.061 s · 12 %")


def test_no_ratios_in_any_notebook_text():
    for text in word_strings() + _markdown_sources():
        assert not RATIO.search(text), text


def test_the_hardware_check_trips_on_planted_controls():
    for planted in ("sm" + "_120", "G" + "B10", "D" + "GX Spark"):
        assert HARDWARE.search(planted), planted


def test_no_hardware_names_in_the_notebook_folder():
    for path in sorted(NOTEBOOKS.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            text = path.read_text(encoding="utf-8")
            assert not HARDWARE.search(text), path.relative_to(ROOT)


def test_the_address_check_trips_on_planted_controls():
    for planted in ("http://x", "ray-serve-llm.rhvp-inference.svc", "localhost", "127.0.0.1", "port :8000"):
        assert ADDRESS.search(planted), planted


def test_no_addresses_in_words_markdown_or_code_cells():
    for text in word_strings() + [cell.source for cell in _notebook().cells]:
        assert not ADDRESS.search(text), text


# ---- the baseline label against the server's pins ----

def test_baseline_matches_the_server_pins():
    requirements = SERVER / "requirements.in"
    if not requirements.is_file():
        pytest.skip("inference-server/requirements.in is not in this checkout")
    version = {name: re.search(rf"{name} (\S+)", words.BASELINE).group(1)
               for name in ("vLLM", "Ray", "PyTorch", "Python", "CUDA")}
    pins = requirements.read_text(encoding="utf-8")
    assert re.search(rf"^ray\[llm\]=={re.escape(version['Ray'])}\s*$", pins, re.MULTILINE), "Ray pin"
    assert re.search(rf"^cuda-toolkit\[[^\]]*\]=={re.escape(version['CUDA'])}\s*$", pins, re.MULTILINE), "CUDA pin"
    # The two CPU-architecture locks only; requirements-pip.lock pins pip alone.
    locks = [SERVER / f"requirements-{arch}.lock" for arch in ("x86_64", "aarch64")]
    for lock in locks:
        assert lock.is_file(), lock.name
        text = lock.read_text(encoding="utf-8")
        assert re.search(rf"^vllm=={re.escape(version['vLLM'])}\b", text, re.MULTILINE), lock.name
        assert re.search(rf"^torch=={re.escape(version['PyTorch'])}\b", text, re.MULTILINE), lock.name
    containerfile = SERVER / "Containerfile"
    if containerfile.is_file():
        text = containerfile.read_text(encoding="utf-8")
        assert re.search(rf"^FROM \S*cuda:{re.escape(version['CUDA'])}-devel-ubi9\b", text, re.MULTILINE)
        assert f"python{version['Python']}" in text


def test_the_baseline_check_trips_on_a_planted_mismatch():
    pins = "ray[llm]==2.59.0\n"
    ray = re.search(r"Ray (\S+)", words.BASELINE).group(1)
    assert not re.search(rf"^ray\[llm\]=={re.escape(ray)}\s*$", pins, re.MULTILINE)


# ---- the INTERNAL strip ----

def test_internal_strip_is_off_by_default_and_on_with_the_setting(monkeypatch):
    from rhvp_bench import run

    monkeypatch.setenv("RHVP_REPLAY", str(NOTEBOOKS / "recordings" / "synthetic.json"))
    monkeypatch.delenv("RHVP_INTERNAL", raising=False)
    plain = run.Session()
    monkeypatch.setenv("RHVP_INTERNAL", "1")
    internal = run.Session()
    try:
        assert [_visible(s) for s in plain.strips()] == [words.REPLAY_STRIP]
        assert [_visible(s) for s in internal.strips()] == [words.REPLAY_STRIP, words.INTERNAL_STRIP]
        assert draw.AMBER in internal.strips()[1]
    finally:
        plain.server.stop()
        internal.server.stop()


# ---- a headless REPLAY run ----

def _cell_outputs(cell) -> str:
    parts = []
    for output in cell.get("outputs", []):
        data = output.get("data", {})
        parts.append(data.get("text/html") or data.get("text/plain") or output.get("text", ""))
    return "".join(parts)


@pytest.fixture(scope="module")
def replayed(tmp_path_factory):
    """The notebook executed in REPLAY mode against the stand-in server, in a copy of the folder: the executed
    notebook and its HTML, both in a temporary folder. The committed notebook is never written."""
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter

    before = NOTEBOOK.read_bytes()
    base = tmp_path_factory.mktemp("replay")
    work = base / "notebooks"
    shutil.copytree(NOTEBOOKS, work, ignore=shutil.ignore_patterns("__pycache__", ".ipynb_checkpoints"))
    kernels = base / "jupyter" / "kernels" / "rhvp-test"
    kernels.mkdir(parents=True)
    (kernels / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-Xfrozen_modules=off", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "rhvp test", "language": "python"}), encoding="utf-8")
    patch = pytest.MonkeyPatch()
    patch.setenv("JUPYTER_PATH", str(base / "jupyter"))
    patch.setenv("RHVP_REPLAY", str(work / "recordings" / "synthetic.json"))
    for name in ("RHVP_INTERNAL", "RHVP_ENDPOINT", "KUBERNETES_SERVICE_HOST"):
        patch.delenv(name, raising=False)
    try:
        nb = nbformat.read(work / NOTEBOOK.name, as_version=4)
        NotebookClient(nb, kernel_name="rhvp-test", timeout=300, resources={"metadata": {"path": str(work)}}).execute()
    finally:
        patch.undo()
    out = base / "out"
    out.mkdir()
    nbformat.write(nb, out / "executed.ipynb")
    body, _ = HTMLExporter().from_notebook_node(nb)
    (out / "executed.html").write_text(body, encoding="utf-8")
    assert NOTEBOOK.read_bytes() == before
    return nb, body


def test_replay_run_has_no_errors_or_printed_text(replayed):
    nb, _ = replayed
    for cell in nb.cells:
        for output in cell.get("outputs", []):
            assert output.output_type in ("display_data", "update_display_data"), (cell.id, output)


def test_replay_run_shows_one_result_per_section_and_no_side_banner(replayed):
    nb, body = replayed
    cards = {1: 1, 2: 1, 3: 1, 4: 3}  # section 4: one card per number of streams
    for number, cell in _sections(nb):
        if cell.cell_type != "code" or number is None:
            continue
        assert _cell_outputs(cell).count('data-rhvp="result-card"') == cards[number], cell.id
    assert body.count('data-rhvp="result-card"') == sum(cards.values())
    assert 'data-rhvp="side-banner"' not in body
    assert f"{draw.ACCENT};" in body


def test_replay_run_results_carry_the_required_lines(replayed):
    nb, body = replayed
    assert words.REPLAY_STRIP in body
    assert words.SINGLE_RUN in body
    for number, cell in _sections(nb):
        if cell.cell_type != "code" or number is None:
            continue
        shown = _visible(_cell_outputs(cell))
        assert words.REPLAY_STRIP in shown and words.SINGLE_RUN in shown, cell.id
        assert f"{words.IN_THIS_REPLAY} · " in shown, cell.id
        if number == 1:
            assert words.ONE_MEASUREMENT in shown and "Empty when the pod started:" in shown
        else:
            assert words.STREAM_GROUPS_NOTE in shown, cell.id
    ready = _visible(_cell_outputs(nb.cells[1]))
    assert words.BASELINE in ready and "Model: qwen2.5-0.5b-instruct" in ready


def test_replay_run_outputs_obey_the_wording_rules(replayed):
    nb, _ = replayed
    for cell in nb.cells:
        shown = _visible(_cell_outputs(cell))
        assert not ADDRESS.search(shown), (cell.id, ADDRESS.search(shown).group(0))
        assert not CAUSE.search(shown), cell.id
        assert not RATIO.search(shown), cell.id
        assert not HARDWARE.search(shown), cell.id
        assert not VENDOR.search(shown), cell.id


# ---- live mode, against the stand-in server given as the endpoint ----

class _Handle:
    def __init__(self, screen: list[str]) -> None:
        self.screen = screen
        self.at = len(screen)
        screen.append("")

    def set(self, markup: str) -> None:
        self.screen[self.at] = markup


def test_live_mode_labels_results_in_this_run_and_never_shows_the_endpoint(monkeypatch):
    from rhvp_bench import replay, run

    screen: list[str] = []

    def show(*parts):
        handle = _Handle(screen)
        handle.set("".join(parts))
        return handle

    monkeypatch.setattr(draw, "show", show)
    monkeypatch.setattr(draw, "update", lambda handle, *parts: handle.set("".join(parts)))
    with replay.FakeServer(replay.synthesize(max_tokens=16, short=-1)) as server:
        monkeypatch.delenv("RHVP_REPLAY", raising=False)
        monkeypatch.delenv("RHVP_INTERNAL", raising=False)
        monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
        monkeypatch.setenv("RHVP_ENDPOINT", server.url)
        monkeypatch.setenv("RHVP_MODEL_ID", "qwen2.5-0.5b-instruct")
        monkeypatch.setenv("RHVP_MODEL_REVISION", "0123abcd")
        monkeypatch.setattr(run, "_SESSION", None)
        monkeypatch.setattr(run.measure, "MAX_TOKENS", 16)
        run.ready()
        run.startup()
        run.first_token(requests=3, warmup=1)
        run.decode(streams=2, warmup=1)
        run.throughput(levels=(1, 2), warmup=1)
    shown = [_visible(markup) for markup in screen]
    everything = " ".join(shown)
    assert words.STARTUP_NOT_IN_CLUSTER in everything
    assert "Hugging Face revision: 0123abcd" in everything
    assert words.REPLAY_STRIP not in everything and words.INTERNAL_STRIP not in everything
    results = [text for text in shown if f"{words.IN_THIS_RUN} · " in text]
    assert len(results) == 3
    assert all(words.SINGLE_RUN in text and words.STREAM_GROUPS_NOTE in text for text in results)
    assert not VENDOR.search(everything)
    assert not ADDRESS.search(everything)
