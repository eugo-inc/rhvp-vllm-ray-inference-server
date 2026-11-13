# SPDX-License-Identifier: Apache-2.0
"""rhvp_compare end to end without a platform: the plan, the round loop with a fake backend and a fake server, the
results file against results.schema.json, the analysis, the markdown report and the command line.

The fake backend hands out one in-process server per round (httpx.MockTransport) and a GuideLLM report made from the
fixture, with side B's inter-token latency 20 % higher and its throughput lower, and a constant 100 W of GPU power.

Run with: python -m pytest -q tests/test_compare_runner.py
"""

import json
import os
import re
import signal
import sys
import time
from pathlib import Path

import httpx
import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))

from rhvp_compare import (  # noqa: E402
    __main__ as cli, analyze, energy, guidellm, plan as plans, report, runner, summary)

FIXTURES = ROOT / "tests" / "fixtures"
SCHEMA = json.loads((ROOT / "notebooks" / "rhvp_compare" / "results.schema.json").read_text())
REPORT = json.loads((FIXTURES / "guidellm-report-v0.8.1.json").read_text())


def config(**extra) -> dict:
    out = {"runName": "fake-run", "model": {"id": "m", "source": "org/m", "revision": ""},
           "sides": [{"name": "old", "image": "registry.example/server:1"},
                     {"name": "new", "image": "registry.example/server:2"}],
           "rounds": {"pairs": 10, "warmup": True, "aaPairs": 2},
           "load": {"concurrencies": [1, 4], "secondsPerLevel": 8}}
    out.update(extra)
    return out


def server(model: str = "m") -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": model}]})
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.0.1-test"})
        if request.url.path in ("/v1/completions", "/v1/chat/completions"):
            body = 'data: {"choices":[{"text":"Hi"}]}\n\ndata: [DONE]\n\n'
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        return httpx.Response(404)
    return httpx.MockTransport(handler)


class FakeSampler:
    source = "nvidia-smi"

    def start(self):
        pass

    def stop(self):
        pass

    def samples(self, start, end):
        return [energy.Sample(start + i * 0.5, 100.0) for i in range(int((end - start) * 2) + 1)], [], None


class FakeBackend:
    platform = "docker"

    def __init__(self, fail_round: int | None = None, hang_once: bool = False, fail_code: str = runner.SIDE_FAILED,
                 load_error: bool = False, term_in_round: int | None = None, errored: dict | None = None,
                 nodes: tuple[str, ...] = ("node-1",), zero: dict | None = None):
        self.fail_round, self.hang_once, self.fail_code = fail_round, hang_once, fail_code
        self.load_error, self.term_in_round = load_error, term_in_round
        self.errored = errored or {}  # round index -> the concurrency level where 40 requests fail
        self.zero = zero or {}  # round index -> the concurrency level whose requests finish with no output token
        self.nodes = nodes  # the node of round i is nodes[i % len(nodes)]
        self.started, self.stopped, self.prepared, self.cleaned = [], [], False, False
        self.events: list = []

    def prepare(self, plan):
        self.prepared = True

    def describe(self):
        return {"node": {"name": None, "arch": "amd64", "instance_type": None, "os_image": "test",
                         "kernel": "1", "container_runtime": "fake"}}

    def gpu_free(self, plan, side):
        pass

    def start(self, plan, round_, side, spec):
        if round_["index"] == self.fail_round:
            raise runner.RoundError("the side never got a node", self.fail_code)
        self.started.append((round_["index"], round_["side"]))
        return {"round": round_, "side": side["key"]}

    def endpoint(self, handle, deadline):
        return "http://side.test:8000"

    def alive(self, handle):
        return None

    def run_load(self, plan, round_, target, out_dir):
        if round_["index"] == self.term_in_round:  # as if the Job were deleted mid-load
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(30)
            raise AssertionError("SIGTERM did not cut the round short")
        if self.load_error:
            raise guidellm.LoadError("GuideLLM exited with code 2; see guidellm.log")
        if self.hang_once:
            self.hang_once = False
            raise guidellm.LoadError("GuideLLM did not finish within 1 s and was stopped", timed_out=True)
        data = json.loads(json.dumps(REPORT))
        slower = 1.2 if round_["side"] == "B" and round_["kind"] == "scored" else 1.0
        wobble = 1.0 + 0.001 * round_["index"]
        for bench in data["benchmarks"]:
            metrics = bench["metrics"]
            if bench["config"]["strategy"]["streams"] == self.errored.get(round_["index"]):
                metrics["request_totals"]["errored"] = 40
            metrics["inter_token_latency_ms"]["successful"]["median"] *= slower * wobble
            metrics["output_tokens_per_second"]["total"]["mean"] /= slower * wobble
            metrics["time_to_first_token_ms"]["successful"]["median"] *= wobble
            if bench["config"]["strategy"]["streams"] == self.zero.get(round_["index"]):
                metrics["output_tokens_per_second"]["total"]["mean"] = 0.0
        path = out_dir / "guidellm.json"
        path.write_text(json.dumps(data))
        return path

    def sampler(self, plan, handle):
        return FakeSampler()

    def logs(self, handle):
        return "2026-01-01T00:00:00Z vLLM API server version 0.0.1\n"

    def facts(self, handle):
        node = self.nodes[handle["round"]["index"] % len(self.nodes)]
        return {"node_name": node, "image_digest": f"registry.example/server@sha256:{handle['side'] * 64}",
                "gpu": {"product": "Test GPU", "driver": "1", "memory_mib": 1, "count": 1, "source": "fake"}}

    def stop(self, handle):
        self.stopped.append(handle["round"]["index"])
        self.events.append(("stop", handle["round"]["index"]))

    def cleanup(self):
        self.cleaned = True
        self.events.append(("cleanup",))


class ServerBackend(FakeBackend):
    """Takes a live server away in prepare and brings it back in restore_server, like kube.py; `restore_fails`
    makes every restore raise."""

    restore_before_teardown = True

    def __init__(self, restore_fails: bool = False, **kw):
        super().__init__(**kw)
        self.restore_fails, self.server_up = restore_fails, True

    def prepare(self, plan):
        super().prepare(plan)
        self.server_up = False

    def restore_server(self):
        self.events.append(("restore",))
        if self.restore_fails:
            raise runner.RoundError("could not scale ray-serve-llm back to 1 (HTTP 503). Bring it back by hand: oc "
                                    "scale deployment/ray-serve-llm --replicas=1 -n vllm-inference")
        self.server_up = True
        return True

    def on_stop(self):
        self.events.append(("on_stop",))


class PublishingBackend(FakeBackend):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.published = []

    def publish(self, run_id, status_text, result_text):
        self.published.append((run_id, json.loads(status_text), result_text))


def run(tmp_path, backend=None, model="m", clock=None, deadline_s=None, **extra) -> tuple[dict, FakeBackend,
                                                                                        runner.Runner]:
    backend = backend or FakeBackend()
    plan = plans.build(config(**extra), "docker")
    kwargs = {"clock": clock} if clock else {}
    the_runner = runner.Runner(plan, backend, tmp_path, sleep=lambda s: None, deadline_s=deadline_s,
                               client=lambda url: httpx.Client(base_url=url, transport=server(model)), **kwargs)
    return the_runner.run(), backend, the_runner


def test_round_order_is_abba_with_warmup_and_aa_last():
    order = plans.round_order(4, True, 1)
    assert [(r["side"], r["kind"]) for r in order] == [
        ("A", "warmup"), ("B", "warmup"), ("A", "scored"), ("B", "scored"), ("B", "scored"), ("A", "scored"),
        ("A", "scored"), ("B", "scored"), ("B", "scored"), ("A", "scored"), ("A", "aa"), ("A", "aa")]
    assert [r["index"] for r in order] == list(range(1, 13))


def test_plan_rejects_bad_configs():
    with pytest.raises(plans.PlanError, match="exactly two sides"):
        plans.build({**config(), "sides": [config()["sides"][0]]})
    with pytest.raises(plans.PlanError, match="runName"):
        plans.build({**config(), "runName": "Bad Name"})
    with pytest.raises(plans.PlanError, match="different names"):
        plans.build({**config(), "sides": [config()["sides"][0]] * 2})
    with pytest.raises(plans.PlanError, match="model.source"):
        plans.build({**config(), "model": {"id": "m"}})
    with pytest.raises(plans.PlanError, match="analysis.correction"):
        plans.build({**config(), "analysis": {"correction": "sidak"}})


def test_plan_declares_everything_before_the_run():
    plan = plans.build(config(), "openshift")
    assert plan["energy"]["source"] == "dcgm" and plans.build(config(), "docker")["energy"]["source"] == "nvidia-smi"
    assert [m["id"] for m in plan["metrics"]][:3] == ["startup.ready_s", "startup.first_answer_s",
                                                       "c1.ttft_ms_median"]
    # Four primary metrics share the correction: time to first answer, and TTFT, throughput and tokens per joule
    # at the highest level; the other six are secondary.
    assert plan["analysis"]["primary"] == ["startup.first_answer_s", "c4.ttft_ms_median", "c4.output_tokens_per_s",
                                           "c4.tokens_per_joule"]
    assert [m["id"] for m in plan["metrics"] if m["role"] == "primary"] == plan["analysis"]["primary"]
    assert sum(m["role"] == "secondary" for m in plan["metrics"]) == 6
    assert plan["floor"] == {"family_size": 4, "min_adjusted_p": pytest.approx(4 * 2 ** -9), "reachable": True,
                             "pairs_needed": 8}
    assert plans.build(config(), "docker")["analysis"]["primary"][-1] == "c4.tokens_per_joule"
    mock = plans.build(config(sides=[{"name": "a", "image": "x:1", "preset": "guidellm-mock", "gpu": False},
                                     {"name": "b", "image": "x:2", "preset": "guidellm-mock", "gpu": False}]))
    assert mock["analysis"]["primary"] == ["startup.first_answer_s", "c4.ttft_ms_median", "c4.output_tokens_per_s"]
    assert plan["run_id"] == f"fake-run-{plans.digest(config())}"
    assert plans.build(config(), "docker")["run_id"] != plans.build(config(runName="other"), "docker")["run_id"]
    few = plans.build(config(rounds={"pairs": 3}), "docker")
    assert not few["floor"]["reachable"]


def test_plan_takes_a_declared_primary_list():
    plan = plans.build(config(analysis={"primary": ["c1.itl_ms_median"]}), "docker")
    assert plan["floor"]["family_size"] == 1 and plan["analysis"]["primary"] == ["c1.itl_ms_median"]
    with pytest.raises(plans.PlanError, match="does not have: c64.itl_ms_median"):
        plans.build(config(analysis={"primary": ["c64.itl_ms_median"]}), "docker")
    for bad in ([], ["c1.itl_ms_median", "c1.itl_ms_median"], "c1.itl_ms_median", [{"id": "x"}]):
        with pytest.raises(plans.PlanError, match="analysis.primary"):
            plans.build(config(analysis={"primary": bad}), "docker")


def test_a_full_run(tmp_path):
    results, backend, the_runner = run(tmp_path)
    jsonschema.validate(results, SCHEMA)
    on_disk = json.loads(the_runner.path.read_text())
    assert on_disk == json.loads(json.dumps(results))
    assert results["run"]["status"] == "complete" and results["run"]["error_code"] is None
    assert backend.prepared and backend.cleaned
    assert all(r["error_code"] is None for r in results["rounds"])
    assert len(results["rounds"]) == 2 + 20 + 4 and all(r["status"] == "ok" for r in results["rounds"])
    assert sorted(backend.stopped) == list(range(1, 27))
    warm = results["rounds"][0]
    assert warm["kind"] == "warmup" and warm["levels"] == [] and set(warm["metrics"]) == {
        "startup.ready_s", "startup.first_answer_s"}
    scored = results["rounds"][2]
    assert scored["server_version"] == {"vllm": "0.0.1-test", "source": "endpoint"}
    assert scored["levels"][0]["energy"]["method"] == "power_mean"
    assert scored["levels"][0]["energy"]["mean_power_w"] == pytest.approx(100.0)
    assert scored["artifacts"] == {"server_log": "rounds/03-A-scored/server.log",
                                   "load_report": "rounds/03-A-scored/guidellm.json"}
    assert (the_runner.dir / scored["artifacts"]["server_log"]).is_file()
    prov = results["provenance"]
    assert prov["sides"]["B"]["image_digest"].endswith("B" * 64) and prov["gpu"]["product"] == "Test GPU"
    assert prov["load_tool"]["version"] == "0.8.1" and prov["seeds"] == {"load": 42, "server": 0}

    rows = {row["id"]: row for row in results["analysis"]["metrics"]}
    assert rows["c4.output_tokens_per_s"]["role"] == "primary" and rows["c4.output_tokens_per_s"]["verdict"] == "SLOWER"
    assert rows["c4.output_tokens_per_s"]["p_adjusted"] == pytest.approx(4 * 2 ** -9)  # Holm over the 4 primaries
    assert rows["c4.tokens_per_joule"]["verdict"] == "LESS_EFFICIENT"
    assert rows["c4.ttft_ms_median"]["verdict"] == "TIE"
    # Secondary: the same rule on the uncorrected p-value, which is kept; p_adjusted is null.
    assert rows["c1.itl_ms_median"]["role"] == "secondary" and rows["c1.itl_ms_median"]["verdict"] == "SLOWER"
    assert rows["c1.itl_ms_median"]["p_adjusted"] is None and rows["c1.itl_ms_median"]["p_value"] == 2 ** -9
    assert rows["c1.itl_ms_median"]["median_ratio"] == pytest.approx(1.2, rel=0.01)
    assert rows["c1.ttft_ms_median"]["verdict"] == "TIE"
    assert all(row["n_pairs"] == 10 for row in rows.values())
    outcomes = results["analysis"]["outcomes"]
    assert outcomes["primary"]["SLOWER"] == 1 and outcomes["primary"]["LESS_EFFICIENT"] == 1
    assert sum(outcomes["primary"].values()) == 4  # the time to first answer is real time here: its verdict varies
    assert sum(results["analysis"]["outcomes"]["secondary"].values()) == 6
    noise = {row["id"]: row for row in results["analysis"]["noise_floor"]}
    assert noise["c1.itl_ms_median"]["pairs"] == 2 and noise["c1.itl_ms_median"]["median_abs_change"] < 0.01


def test_a_failed_round_is_kept_and_the_run_goes_on(tmp_path):
    results, _, _ = run(tmp_path, FakeBackend(fail_round=5))
    jsonschema.validate(results, SCHEMA)
    failed = [r for r in results["rounds"] if r["status"] != "ok"]
    assert len(failed) == 1 and failed[0]["index"] == 5 and failed[0]["error"] == "the side never got a node"
    assert failed[0]["error_code"] == "SIDE_FAILED"
    assert results["run"]["status"] == "partial" and len(results["rounds"]) == 26
    assert results["run"]["error_code"] == "ROUNDS_FAILED"
    rows = {row["id"]: row for row in results["analysis"]["metrics"]}
    assert rows["c1.itl_ms_median"]["n_pairs"] == 9


@pytest.mark.parametrize("backend, model, code", [
    (FakeBackend(fail_round=1, fail_code=runner.GPU_NOT_FREE), "m", "GPU_NOT_FREE"),
    (FakeBackend(load_error=True), "m", "LOAD_FAILED"),
    (FakeBackend(), "another-model", "SIDE_NEVER_READY"),
])
def test_each_failure_has_its_code(tmp_path, backend, model, code):
    # A clock far in the past puts every round's deadline behind the probe's own clock: a side that never lists the
    # model is SIDE_NEVER_READY at once.
    results, _, _ = run(tmp_path, backend, model=model, clock=lambda: -1e9, rounds={"pairs": 1, "warmup": False})
    jsonschema.validate(results, SCHEMA)
    assert results["rounds"][0]["status"] == "failed" and results["rounds"][0]["error_code"] == code
    assert results["run"]["error_code"] == "ROUNDS_FAILED"


def test_a_side_that_dies_is_side_failed(tmp_path):
    backend = FakeBackend()
    backend.alive = lambda handle: "the server exited: Error, exit code 1"
    results, _, _ = run(tmp_path, backend, model="another-model", rounds={"pairs": 1, "warmup": False})
    assert results["rounds"][0]["error_code"] == "SIDE_FAILED"
    assert results["rounds"][0]["error"] == "the server exited: Error, exit code 1"


def test_sigterm_cuts_the_round_short_and_cleans_up(tmp_path):
    started = time.monotonic()
    results, backend, the_runner = run(tmp_path, FakeBackend(term_in_round=3))
    assert time.monotonic() - started < 20  # the load's 30 s sleep was interrupted
    jsonschema.validate(results, SCHEMA)
    assert [r["index"] for r in results["rounds"]] == [1, 2, 3]  # no round after the stop
    assert results["rounds"][2]["error_code"] == "STOPPED" and 3 in backend.stopped and backend.cleaned
    assert results["run"]["status"] == "partial" and results["run"]["error_code"] == "STOPPED"
    assert json.loads(the_runner.path.read_text())["run"]["error_code"] == "STOPPED"
    assert signal.getsignal(signal.SIGTERM) is not the_runner._on_term


def test_sigterm_near_the_jobs_time_limit_is_run_timeout(tmp_path):
    results, _, _ = run(tmp_path, FakeBackend(term_in_round=3), deadline_s=60)
    assert results["run"]["error_code"] == "RUN_TIMEOUT" and results["rounds"][2]["error_code"] == "STOPPED"


def test_sigterm_between_rounds_stops_before_the_next(tmp_path):
    backend = FakeBackend()
    real_stop = backend.stop

    def stop(handle):
        real_stop(handle)
        if handle["round"]["index"] == 2:
            os.kill(os.getpid(), signal.SIGTERM)  # arrives during teardown: not interruptible, the round stands
    backend.stop = stop
    results, _, _ = run(tmp_path, backend)
    assert [r["status"] for r in results["rounds"]] == ["ok", "ok"]
    assert results["run"]["error_code"] == "STOPPED"


def test_status_is_published_after_every_round(tmp_path):
    backend = PublishingBackend()
    results, _, the_runner = run(tmp_path, backend, rounds={"pairs": 2, "warmup": True})
    status_schema = {"$defs": SCHEMA["$defs"], "properties": {"run": SCHEMA["properties"]["run"]},
                     "$ref": "#/$defs/status"}
    assert len(backend.published) == 1 + 6 + 1  # the start, each round, the end
    for run_id, status, result_text in backend.published:
        jsonschema.validate(status, status_schema)
        assert run_id == results["run"]["id"] and json.loads(result_text)["run"]["id"] == run_id
    first, last = backend.published[1][1], backend.published[-1][1]
    assert first["phase"] == "running" and first["rounds_done"] == 1 and first["eta_seconds"] is not None
    assert last["phase"] == "complete" and last["completed_pairs"] == last["total_pairs"] == 2
    assert last["eta_seconds"] is None and [p["id"] for p in last["primary"]] == results["plan"]["analysis"]["primary"]


def test_a_failing_publish_is_a_note_not_a_failure(tmp_path):
    backend = PublishingBackend()
    backend.publish = lambda *a: (_ for _ in ()).throw(RuntimeError("forbidden"))
    results, _, _ = run(tmp_path, backend, rounds={"pairs": 1, "warmup": False})
    assert results["run"]["status"] == "complete"
    assert [n for n in results["run"]["notes"] if "not published" in n] == [
        "the run's status was not published: RuntimeError: forbidden"]


def test_main_run_prints_the_results_and_writes_the_termination_message(tmp_path, capsys, monkeypatch):
    path = tmp_path / "compare.json"
    path.write_text(json.dumps(config(rounds={"pairs": 1, "warmup": False})))
    message = tmp_path / "termination-log"
    original = runner.Runner.__init__

    def init(self, *a, **kw):  # main_run builds its own Runner: give it the fake server
        original(self, *a, client=lambda url: httpx.Client(base_url=url, transport=server()),
                 sleep=lambda s: None, **kw)
    monkeypatch.setattr(runner.Runner, "__init__", init)
    code = runner.main_run(str(path), str(tmp_path / "results"), FakeBackend(fail_round=2),
                           env={"RHVP_COMPARE_TERMINATION_LOG": str(message)})
    assert code == 1  # partial: the Job reads Failed
    out = capsys.readouterr().out.splitlines()
    begin = out.index(summary.BEGIN)
    assert out[begin + 2] == summary.END
    printed = json.loads(out[begin + 1])
    jsonschema.validate(printed, SCHEMA)
    short = json.loads(message.read_text())
    assert len(message.read_bytes()) <= 4096
    assert short["status"] == "partial" and short["error_code"] == "ROUNDS_FAILED"
    assert short["completed_pairs"] == 1 and short["total_pairs"] == 1 and short["run_id"] == printed["run"]["id"]


def test_a_config_that_cannot_run_still_says_so(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({**config(), "sides": []}))
    message = tmp_path / "termination-log"
    with pytest.raises(plans.PlanError):
        runner.main_run(str(path), str(tmp_path), FakeBackend(), env={"RHVP_COMPARE_TERMINATION_LOG": str(message)})
    assert json.loads(message.read_text())["error_code"] == "PLAN_INVALID"
    assert summary.BEGIN in capsys.readouterr().out


def test_the_termination_message_never_passes_4_kib():
    results = {"run": {"id": "r", "status": "complete", "error_code": None}, "plan": None,
               "analysis": {"method": {"pairs_usable_min": 10}, "outcomes": {"primary": {"FASTER": 300}},
                            "metrics": [{"id": f"c{i}.output_tokens_per_s", "verdict": "FASTER", "role": "primary"}
                                        for i in range(300)]}}
    text = summary.termination(results)
    assert len(text.encode()) <= 4096 and "primary" not in json.loads(text)


def test_a_hung_load_is_tried_once_more(tmp_path):
    results, _, _ = run(tmp_path, FakeBackend(hang_once=True), rounds={"pairs": 2, "warmup": False})
    first = results["rounds"][0]
    assert first["status"] == "ok" and "run once more" in first["notes"][0]


def test_a_complete_run_is_not_run_again(tmp_path):
    first, _, _ = run(tmp_path, rounds={"pairs": 1, "warmup": False})
    again, backend, _ = run(tmp_path, rounds={"pairs": 1, "warmup": False})
    assert backend.started == [] and again == json.loads(json.dumps(first))


def test_an_unfinished_run_is_kept_beside_the_new_one(tmp_path):
    plan = plans.build(config(rounds={"pairs": 1, "warmup": False}), "docker")
    folder = tmp_path / plan["run_id"]
    folder.mkdir()
    (folder / "results.json").write_text(json.dumps({"run": {"status": "running"}}))
    results, _, _ = run(tmp_path, rounds={"pairs": 1, "warmup": False})
    assert results["run"]["status"] == "complete"
    assert len(list(folder.glob("results.attempt-*.json"))) == 1


def test_analysis_is_a_pure_function_of_plan_and_rounds(tmp_path):
    results, _, _ = run(tmp_path)
    again = analyze.analyze({"plan": results["plan"], "rounds": results["rounds"]})
    assert json.loads(json.dumps(again)) == json.loads(json.dumps(results["analysis"]))
    assert again["tool"] == analyze.TOOL


def test_markdown_report(tmp_path):
    results, _, the_runner = run(tmp_path)
    text = report.markdown(results)
    assert (the_runner.dir / "results.md").read_text() == text
    assert "| inter-token latency, median, 1 concurrent |" in text and "SLOWER" in text
    assert "registry.example/server:1" in text and "A/A" in text
    assert not re.search(r"\d\s?[x×]\b", text)


def test_command_line_validate_analyze_report(tmp_path, capsys):
    path = tmp_path / "compare.json"
    path.write_text(json.dumps(config()))
    assert cli.main(["validate", "--config", str(path), "--platform", "docker"]) == 0
    out = capsys.readouterr().out
    assert "rounds (26; w = warm-up, * = A/A): Aw Bw A B B A" in out and "verdict is possible" in out
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**config(), "sides": []}))
    assert cli.main(["validate", "--config", str(bad)]) == 2
    results, _, the_runner = run(tmp_path)
    assert cli.main(["analyze", str(the_runner.path), "--write"]) == 0
    assert "SLOWER" in capsys.readouterr().out
    assert cli.main(["report", str(the_runner.path)]) == 0


def test_the_pinned_guidellm_image_is_the_same_everywhere():
    helpers = (ROOT / "charts/all/notebook/templates/_helpers.tpl").read_text()
    script = (ROOT / "scripts/compare-local.sh").read_text()
    assert plans.GUIDELLM_IMAGE in helpers and plans.GUIDELLM_IMAGE in script


def _main(tmp_path, backend, monkeypatch, **extra) -> tuple[int, Path, Path]:
    """main_run with the fake server; returns (exit code, the results root, the termination message file)."""
    path = tmp_path / "compare.json"
    path.write_text(json.dumps(config(**extra)))
    message = tmp_path / "termination-log"
    original = runner.Runner.__init__

    def init(self, *a, **kw):
        original(self, *a, client=lambda url: httpx.Client(base_url=url, transport=server()),
                 sleep=lambda s: None, **kw)
    monkeypatch.setattr(runner.Runner, "__init__", init)
    root = tmp_path / "results"
    code = runner.main_run(str(path), str(root), backend, env={"RHVP_COMPARE_TERMINATION_LOG": str(message)})
    return code, root, message


def test_requests_that_fail_on_one_side_make_its_metrics_inconclusive(tmp_path):
    """Side B loses 40 requests at 4 streams in two of its scored rounds. Those two pairs drop out of the 4-stream
    metrics; the 8 pairs left would make B look better than it was, so the metrics are INCONCLUSIVE, the run is
    partial (REQUESTS_FAILED) and the levels are listed everywhere a reader looks."""
    backend = PublishingBackend(errored={4: 4, 8: 4})  # rounds 4 and 8 are side B's scored rounds of pairs 1 and 3
    results, _, the_runner = run(tmp_path, backend, rounds={"pairs": 10, "warmup": True, "aaPairs": 0})
    jsonschema.validate(results, SCHEMA)
    assert [(r["index"], r["side"]) for r in results["rounds"] if r["index"] in (4, 8)] == [(4, "B"), (8, "B")]
    rows = {row["id"]: row for row in results["analysis"]["metrics"]}
    for metric in ("c4.ttft_ms_median", "c4.itl_ms_median", "c4.output_tokens_per_s", "c4.tokens_per_joule"):
        assert rows[metric]["n_pairs"] == 8
        assert rows[metric]["pairs_lost"] == {"A": 0, "B": 2, "both": 0, "other": 0}
        assert (rows[metric]["verdict"], rows[metric]["reason"]) == ("INCONCLUSIVE", "pairs_lost_to_one_side_failing")
    assert rows["c1.itl_ms_median"]["verdict"] == "SLOWER" and rows["c1.itl_ms_median"]["pairs_lost"]["B"] == 0
    assert results["run"]["status"] == "partial" and results["run"]["error_code"] == "REQUESTS_FAILED"
    status = backend.published[-1][1]
    assert [(u["index"], u["side"], u["concurrency"], u["errored"]) for u in status["levels_unusable"]] == [
        (4, "B", 4, 40), (8, "B", 4, 40)]
    text = (the_runner.dir / "results.md").read_text()
    assert "round 4 (side B, scored), 4 streams: 40 requests failed" in text
    assert "side A 0, side B 2" in text


def test_a_zero_from_one_side_is_a_pair_lost_to_it_not_a_pair_left_out(tmp_path):
    """Side B's requests at 4 streams finish but give no output token in two of its scored rounds: 0 tokens per
    second and 0 tokens per joule there. A 0 gives no ratio. Those two pairs used to drop out unseen, and the 8 left
    decided the verdict (SLOWER here); now they are pairs lost to side B, and both metrics are INCONCLUSIVE."""
    results, _, the_runner = run(tmp_path, FakeBackend(zero={4: 4, 8: 4}),
                                 rounds={"pairs": 10, "warmup": True, "aaPairs": 0})
    jsonschema.validate(results, SCHEMA)
    b4 = next(r for r in results["rounds"] if r["index"] == 4)
    assert b4["side"] == "B" and b4["metrics"]["c4.output_tokens_per_s"] == 0
    assert b4["metrics"]["c4.tokens_per_joule"] == 0
    rows = {row["id"]: row for row in results["analysis"]["metrics"]}
    for metric in ("c4.output_tokens_per_s", "c4.tokens_per_joule"):
        assert rows[metric]["n_pairs"] == 8
        assert rows[metric]["pairs_lost"] == {"A": 0, "B": 2, "both": 0, "other": 0}
        assert (rows[metric]["verdict"], rows[metric]["reason"]) == ("INCONCLUSIVE", "pairs_lost_to_one_side_failing")
    assert rows["c4.itl_ms_median"]["pairs_lost"]["B"] == 0 and rows["c4.itl_ms_median"]["n_pairs"] == 10
    assert "side A 0, side B 2" in (the_runner.dir / "results.md").read_text()


def test_a_zero_counts_against_the_side_that_measured_it():
    metric = {"id": "c4.output_tokens_per_s", "concurrency": 4}

    def scored(pair, side, value):
        return {"kind": "scored", "pair": pair, "side": side, "status": "ok", "index": 2 * pair + (side == "B"),
                "metrics": {} if value is None else {metric["id"]: value},
                "levels": [{"concurrency": 4, "usable": True}]}
    rounds = [scored(1, "A", 100.0), scored(1, "B", 0.0),   # B measured 0: lost to B
              scored(2, "A", 0), scored(2, "B", 0.0),       # both measured 0
              scored(3, "A", 0.0), scored(3, "B", None),    # A measured 0, B's figure is missing: lost to A
              scored(4, "A", 100.0), scored(4, "B", 120.0)]
    assert analyze.paired(rounds, "scored", metric["id"]) == [(100.0, 120.0)]
    assert analyze.pairs_lost(rounds, metric) == {"A": 1, "B": 1, "both": 1, "other": 0}
    assert analyze._side_median(rounds, "B", metric["id"]) == 0.0  # the zeros are in the side's median


def test_a_round_the_side_itself_failed_counts_against_it(tmp_path):
    results, _, _ = run(tmp_path, FakeBackend(fail_round=5), rounds={"pairs": 10, "warmup": True, "aaPairs": 0})
    row = {r["id"]: r for r in results["analysis"]["metrics"]}["c1.itl_ms_median"]
    assert row["pairs_lost"] == {"A": 0, "B": 1, "both": 0, "other": 0} and row["verdict"] == "INCONCLUSIVE"
    not_hers, _, _ = run(tmp_path / "gpu", FakeBackend(fail_round=5, fail_code=runner.GPU_NOT_FREE),
                         rounds={"pairs": 10, "warmup": True, "aaPairs": 0})
    row = {r["id"]: r for r in not_hers["analysis"]["metrics"]}["c1.itl_ms_median"]
    assert row["pairs_lost"] == {"A": 0, "B": 0, "both": 0, "other": 1} and row["verdict"] == "SLOWER"


def test_rounds_on_two_nodes_make_every_verdict_inconclusive(tmp_path):
    results, _, _ = run(tmp_path, FakeBackend(nodes=("node-1", "node-2")))
    jsonschema.validate(results, SCHEMA)
    assert {row["reason"] for row in results["analysis"]["metrics"]} <= {"rounds_ran_on_more_than_one_node",
                                                                         "median_inside_tie_band_but_interval_is_not",
                                                                         "not_significant_outside_tie_band"}
    assert {row["verdict"] for row in results["analysis"]["metrics"]} == {"INCONCLUSIVE"}
    assert [n["name"] for n in results["provenance"]["nodes"]] == ["node-2", "node-1"]


def test_the_live_server_comes_back_and_the_run_says_so(tmp_path, monkeypatch):
    backend = ServerBackend()
    code, root, message = _main(tmp_path, backend, monkeypatch, rounds={"pairs": 1, "warmup": False})
    assert code == 0 and backend.server_up
    assert backend.events[-2:] == [("restore",), ("cleanup",)]  # back first, then the cleanup
    [path] = root.glob("*/results.json")
    assert json.loads(path.read_text())["run"]["server_restored"] is True
    assert json.loads(message.read_text())["server_restored"] is True


def test_a_server_that_does_not_come_back_is_exit_4(tmp_path, monkeypatch, capsys):
    backend = ServerBackend(restore_fails=True)
    code, root, message = _main(tmp_path, backend, monkeypatch, rounds={"pairs": 1, "warmup": False})
    assert code == runner.EXIT_SERVER_NOT_RESTORED == 4
    [path] = root.glob("*/results.json")
    results = json.loads(path.read_text())
    jsonschema.validate(results, SCHEMA)
    assert results["run"]["status"] == "complete" and results["run"]["server_restored"] is False
    assert any("oc scale deployment/ray-serve-llm --replicas=1" in e for e in results["run"]["errors"])
    assert json.loads(message.read_text())["server_restored"] is False
    assert "error: the live server was not brought back" in capsys.readouterr().out
    assert "The live server was not brought back" in (path.parent / "results.md").read_text()


def test_a_stop_asks_the_server_back_before_the_teardown(tmp_path):
    backend = ServerBackend(term_in_round=3)
    results, _, _ = run(tmp_path, backend)
    assert results["run"]["error_code"] == "STOPPED" and results["run"]["server_restored"] is True
    events = backend.events
    assert events.index(("restore",)) < events.index(("stop", 3)) < events.index(("cleanup",))
    assert events.count(("restore",)) == 1 and ("on_stop",) in events


def test_a_second_run_refuses_to_start_and_changes_nothing(tmp_path, monkeypatch):
    backend = ServerBackend()

    def busy():
        raise runner.RunActive("another comparison is running: runner pod compare-x-1 (Running)")
    backend.check_alone = busy
    code, root, message = _main(tmp_path, backend, monkeypatch, rounds={"pairs": 1, "warmup": False})
    assert code == runner.EXIT_RUN_ACTIVE == 5
    assert not backend.prepared and backend.events == [] and backend.server_up  # nothing touched
    assert not list(root.glob("*/results.json")) if root.exists() else True
    short = json.loads(message.read_text())
    assert short["error_code"] == "RUN_ACTIVE" and short["status"] == "failed"


def test_a_run_that_fails_outside_a_round_is_exit_3(tmp_path, monkeypatch):
    backend = ServerBackend()
    backend.describe = lambda: (_ for _ in ()).throw(KeyError("boom"))  # recorded as a note, not a failure
    backend.gpu_free = lambda plan, side: None
    original = backend.prepare

    def prepare(plan):
        original(plan)
        raise runner.RoundError("could not list pods in vllm-inference: HTTP 403")
    backend.prepare = prepare
    code, root, message = _main(tmp_path, backend, monkeypatch, rounds={"pairs": 1, "warmup": False})
    assert code == runner.EXIT_FAILED == 3
    [path] = root.glob("*/results.json")
    results = json.loads(path.read_text())
    assert results["run"]["status"] == "failed" and results["run"]["error_code"] == "RUNNER_FAILED"
    assert results["run"]["server_restored"] is True and backend.server_up  # brought back all the same


def test_a_server_that_comes_back_on_the_last_try_is_back(tmp_path):
    backend = ServerBackend(term_in_round=3)
    tries = {"n": 0}

    def restore():
        tries["n"] += 1
        backend.events.append(("restore",))
        if tries["n"] == 1:  # the early try on the stop fails, the one at the end works
            raise runner.RoundError("could not scale ray-serve-llm back to 1 (HTTP 503)")
        return True
    backend.restore_server = restore
    results, _, _ = run(tmp_path, backend)
    assert tries["n"] == 2 and results["run"]["server_restored"] is True
    assert not [e for e in results["run"]["errors"] if e.startswith(runner.NOT_RESTORED)]
    assert [n for n in results["run"]["notes"] if "a later try brought it back" in n]
    assert runner.exit_code(results) == runner.EXIT_PARTIAL
