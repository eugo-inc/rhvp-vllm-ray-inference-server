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
                 load_error: bool = False, term_in_round: int | None = None):
        self.fail_round, self.hang_once, self.fail_code = fail_round, hang_once, fail_code
        self.load_error, self.term_in_round = load_error, term_in_round
        self.started, self.stopped, self.prepared, self.cleaned = [], [], False, False

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
            metrics["inter_token_latency_ms"]["successful"]["median"] *= slower * wobble
            metrics["output_tokens_per_second"]["total"]["mean"] /= slower * wobble
            metrics["time_to_first_token_ms"]["successful"]["median"] *= wobble
        path = out_dir / "guidellm.json"
        path.write_text(json.dumps(data))
        return path

    def sampler(self, plan, handle):
        return FakeSampler()

    def logs(self, handle):
        return "2026-01-01T00:00:00Z vLLM API server version 0.0.1\n"

    def facts(self, handle):
        return {"node_name": "node-1", "image_digest": f"registry.example/server@sha256:{handle['side'] * 64}",
                "gpu": {"product": "Test GPU", "driver": "1", "memory_mib": 1, "count": 1, "source": "fake"}}

    def stop(self, handle):
        self.stopped.append(handle["round"]["index"])

    def cleanup(self):
        self.cleaned = True


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
