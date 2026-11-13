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
    __main__ as cli, analyze, energy, guidellm, plan as plans, report, runner, show, summary)

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


#: The end of a vLLM log whose start stopped at the start-up memory check (as seen on a GPU whose memory is the
#: machine's own).
PROFILING_LOG = ("2026-01-01T00:00:00Z (EngineCore pid=1) ERROR [core.py:1510] AssertionError: Error in memory "
                 "profiling. Initial free memory 78.16 GiB, current free memory 78.49 GiB.\n")


def server(model: str = "m") -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "dead.test":  # a server that exited: nothing answers
            return httpx.Response(503)
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
                 nodes: tuple[str, ...] = ("node-1",), zero: dict | None = None, dies: dict | None = None,
                 dies_log: str = PROFILING_LOG):
        self.fail_round, self.hang_once, self.fail_code = fail_round, hang_once, fail_code
        self.load_error, self.term_in_round = load_error, term_in_round
        self.errored = errored or {}  # round index -> the concurrency level where 40 requests fail
        self.zero = zero or {}  # round index -> the concurrency level whose requests finish with no output token
        self.dies, self.dies_log = dies or {}, dies_log  # round index -> how many of its starts exit, with that log
        self.attempts: dict[int, int] = {}
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
        tries = self.attempts[round_["index"]] = self.attempts.get(round_["index"], 0) + 1
        return {"round": round_, "side": side["key"], "dies": tries <= self.dies.get(round_["index"], 0)}

    def endpoint(self, handle, deadline):
        return "http://dead.test:8000" if handle.get("dies") else "http://side.test:8000"

    def alive(self, handle):
        return "the server exited with code 1" if handle.get("dies") else None

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
        return "2026-01-01T00:00:00Z vLLM API server version 0.0.1\n" + (self.dies_log if handle.get("dies") else "")

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


class FoundDownBackend(FakeBackend):
    """A backend whose live server was running when the run began but had stopped by itself before prepare() came to
    take it away (dockerapi.py sets server_found_down then, and leaves the server as it is)."""

    server_found_down = None

    def prepare(self, plan):
        super().prepare(plan)
        self.server_found_down = "exited"

    def restore_server(self):
        return None  # nothing was taken away


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


def scale_config(**extra) -> dict:
    """A scaleout config; a `scaleOut` map in `extra` is merged into the one here."""
    scale = {"requestRate": 10, "surgeSeconds": 40, "stepAfterSeconds": 5, "ttftMs": 200, "tpotMs": 20}
    if isinstance(extra.get("scaleOut"), dict):
        extra["scaleOut"] = {**scale, **extra["scaleOut"]}
    out = config(mode="scaleout", rounds={"pairs": 7, "warmup": True, "aaPairs": 0}, scaleOut=scale)
    out.update(extra)
    return out


def test_plan_scale_out_mode():
    plan = plans.build(scale_config(), "docker")
    assert plan["mode"] == "scaleout" and "mode" not in plans.build(config(), "docker")
    assert plan["analysis"]["primary"] == ["scale.first_answer_s", "surge.slo_attainment"]
    assert [m["id"] for m in plan["metrics"] if m["role"] == "primary"] == plan["analysis"]["primary"]
    # Holm over the two primaries: 7 pairs are the fewest that can reach alpha 0.05 (2 * 2**-6 = 0.031).
    assert plan["floor"] == {"family_size": 2, "min_adjusted_p": pytest.approx(2 * 2 ** -6), "reachable": True,
                             "pairs_needed": 7}
    assert not plans.build(scale_config(rounds={"pairs": 6}), "docker")["floor"]["reachable"]
    assert plan["energy"]["source"] == "none" and plan["load"]["concurrencies"] == []
    assert plan["load"]["seconds_per_level"] == 40 and plan["load"]["timeout_s"] == 40 * 1.5 + 300
    assert plan["server"]["gpu_memory_utilization"] == plans.SHARED_GPU_MEMORY_UTILIZATION  # two replicas, one GPU
    scale = plan["scale_out"]
    assert (scale["request_rate"], scale["surge_s"], scale["step_after_s"], scale["slo"], scale["router"]) == (
        10, 40, 5, {"ttft_ms": 200, "tpot_ms": 20}, "least-busy")
    assert (scale["replicas_start"], scale["replicas_end"], scale["shared_gpu"]) == (1, 2, True)
    assert all(m["concurrency"] is None for m in plan["metrics"])
    assert plans.build(scale_config(server={"gpuMemoryUtilization": 0.3}), "docker")["server"][
        "gpu_memory_utilization"] == 0.3
    assert plans.build(scale_config(scaleOut={"sharedGpu": False}), "docker")["server"][
        "gpu_memory_utilization"] == 0.85
    assert plans.build(scale_config(scaleOut={"tpotMs": None}), "docker")["scale_out"]["slo"] == {
        "ttft_ms": 200, "tpot_ms": None}
    assert plans.build(config(), "docker")["sides"][0]["compile_cache"] == "cold"
    assert "launches" in SCHEMA["$defs"]["round"]["required"]  # every round of every mode writes it


@pytest.mark.parametrize("extra, platform, message", [
    ({}, "openshift", "mode scaleout runs on the docker backend only"),
    ({"scaleOut": {"stepAfterSeconds": 40}}, "docker", "must be less than scaleOut.surgeSeconds"),
    ({"scaleOut": {"ttftMs": None, "tpotMs": None}}, "docker", "at least one latency goal"),
    ({"scaleOut": "fast"}, "docker", "scaleOut must be a map"),
    ({"scaleOut": {"sharedGpu": "yes"}}, "docker", "sharedGpu must be true or false"),
    ({"server": {"gpuMemoryUtilization": 0.6}}, "docker", "two of them need it at most 0.475"),
    ({"energy": {"source": "nvidia-smi"}}, "docker", "is not measured in mode scaleout"),
    ({"analysis": {"primary": ["c4.ttft_ms_median"]}}, "docker", "does not have: c4.ttft_ms_median"),
])
def test_plan_scale_out_mode_refusals(extra, platform, message):
    with pytest.raises(plans.PlanError, match=re.escape(message)):
        plans.build(scale_config(**extra), platform)


def test_plan_warm_compile_cache_runs_on_docker_after_a_warm_up():
    warm = config(sides=[{"name": "old", "image": "registry.example/server:1", "compileCache": "warm"},
                         {"name": "new", "image": "registry.example/server:2"}])
    assert [s["compile_cache"] for s in plans.build(warm, "docker")["sides"]] == ["warm", "cold"]
    with pytest.raises(plans.PlanError, match="compileCache warm runs on the docker backend only"):
        plans.build(warm, "openshift")
    with pytest.raises(plans.PlanError, match="needs rounds.warmup: true"):
        plans.build({**warm, "rounds": {"pairs": 3, "warmup": False}}, "docker")
    with pytest.raises(plans.PlanError, match="compileCache must be one of cold, warm"):
        plans.build(config(sides=[{"name": "a", "image": "x:1", "compileCache": "hot"}, {"name": "b", "image": "x:2"}]),
                    "docker")


def test_the_scale_out_mock_config_builds_on_docker_only():
    """tests/fixtures/compare-scaleout-mock.json, the config of the docker mock run (scripts/compare-local.sh)."""
    mock = json.loads((FIXTURES / "compare-scaleout-mock.json").read_text())
    plan = plans.build(mock, "docker")
    assert plan["mode"] == "scaleout" and plan["pairs"] == 7 and plan["floor"]["reachable"]
    assert [m["id"] for m in plan["metrics"] if m.get("role", "primary") == "primary"] == list(plans.SCALE_PRIMARY)
    assert plan["scale_out"]["step_after_s"] < plan["scale_out"]["surge_s"] == plan["load"]["seconds_per_level"]
    spec = plans.side_spec(plan, plan["sides"][0])
    assert spec["command"][:2] == ["sh", "-c"] and spec["gpu"] is False
    assert spec["args"][-2:] == ["--max-concurrent-requests", "2"] and "--port" in spec["args"]
    assert plan["sides"][0]["image"] == plan["sides"][1]["image"] == plans.GUIDELLM_IMAGE
    with pytest.raises(plans.PlanError, match="docker backend only"):
        plans.build(mock, "openshift")


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
    results, _, the_runner = run(tmp_path, FakeBackend(hang_once=True), rounds={"pairs": 2, "warmup": False})
    jsonschema.validate(results, SCHEMA)
    first = results["rounds"][0]
    assert first["status"] == "ok" and "run once more" in first["notes"][0]
    assert [r["load_retries"] for r in results["rounds"]] == [1, 0, 0, 0]
    text = (the_runner.dir / "results.md").read_text()
    assert ("Rounds whose load ran again after GuideLLM timed out (GuideLLM ran once more against the same "
            "server") in text
    assert "- round 1 (side A, scored): ran again, then finished" in text
    head = "".join(show._head(results, [], []))
    assert "Rounds whose load ran again after GuideLLM timed out: 1 (side A)." in head
    assert "against the same server" in head


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


def test_compare_local_passes_on_settings_the_docker_backend_reads():
    script = (ROOT / "scripts/compare-local.sh").read_text()
    backend = (ROOT / "notebooks/rhvp_compare/dockerapi.py").read_text()
    passed = set(re.findall(r"-e (RHVP_COMPARE_[A-Z_]+)=", script))
    read = set(re.findall(r'env\.get\("(RHVP_COMPARE_[A-Z_]+)"', backend))
    assert {"RHVP_COMPARE_REPLICA_GPUS", "RHVP_COMPARE_CPUSET", "RHVP_COMPARE_CACHE_VOLUME"} <= passed
    assert passed <= read
    assert '"${cpuset_flag[@]}"' in script  # the runner container itself stays on the CPU set


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


def test_a_start_that_fails_vllms_memory_check_is_tried_once_more(tmp_path):
    backend = FakeBackend(dies={3: 1})
    results, _, the_runner = run(tmp_path, backend)
    jsonschema.validate(results, SCHEMA)
    r3 = next(r for r in results["rounds"] if r["index"] == 3)
    assert (r3["side"], r3["kind"], r3["status"], r3["start_retries"]) == ("A", "scored", "ok", 1)
    assert [(l["replica"], l["start_retries"]) for l in r3["launches"]] == [(1, 1)]  # one record, of the last start
    assert backend.started.count((3, "A")) == 2 and backend.stopped.count(3) == 2  # the first try, then the round
    first_try = the_runner.dir / r3["artifacts"]["server_log_attempt_1"]
    assert "Error in memory profiling" in first_try.read_text()
    assert "Error in memory profiling" not in (the_runner.dir / r3["artifacts"]["server_log"]).read_text()
    assert any("started once more" in note for note in r3["notes"])
    assert all(r["start_retries"] == 0 for r in results["rounds"] if r["index"] != 3)
    assert results["run"]["status"] == "complete" and runner.exit_code(results) == runner.EXIT_COMPLETE
    assert "- round 3 (side A, scored): started 2 times, then finished" in (the_runner.dir / "results.md").read_text()
    head = "".join(show._head(results, [], []))
    assert "started once more" in head and "3 (side A)" in head


def test_a_second_memory_check_failure_fails_the_round(tmp_path):
    backend = FakeBackend(dies={3: 2})
    results, _, the_runner = run(tmp_path, backend)
    jsonschema.validate(results, SCHEMA)
    r3 = next(r for r in results["rounds"] if r["index"] == 3)
    assert (r3["status"], r3["error_code"], r3["start_retries"]) == ("failed", "SIDE_FAILED", 1)
    assert r3["error"] == "the server exited with code 1"
    assert backend.started.count((3, "A")) == 2
    assert "started 2 times, then SIDE_FAILED" in (the_runner.dir / "results.md").read_text()


def test_a_server_that_exits_for_another_reason_is_not_started_again(tmp_path):
    backend = FakeBackend(dies={3: 1}, dies_log="2026-01-01T00:00:01Z torch.OutOfMemoryError: CUDA out of memory\n")
    results, _, _ = run(tmp_path, backend)
    r3 = next(r for r in results["rounds"] if r["index"] == 3)
    assert (r3["status"], r3["error_code"], r3["start_retries"]) == ("failed", "SIDE_FAILED", 0)
    assert backend.started.count((3, "A")) == 1 and "server_log_attempt_1" not in r3["artifacts"]


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
    assert json.loads(path.read_text())["run"]["server_found_down"] is None
    assert json.loads(message.read_text())["server_restored"] is True


def test_a_server_that_stopped_by_itself_before_the_run_took_it_is_recorded(tmp_path, monkeypatch, capsys):
    code, root, _ = _main(tmp_path, FoundDownBackend(), monkeypatch, rounds={"pairs": 1, "warmup": False})
    assert code == runner.EXIT_COMPLETE  # the run took nothing away, so this is not its failure
    [path] = root.glob("*/results.json")
    results = json.loads(path.read_text())
    jsonschema.validate(results, SCHEMA)
    run = results["run"]
    assert (run["server_found_down"], run["server_restored"], run["status"]) == ("exited", None, "complete")
    [note] = [n for n in run["notes"] if n.startswith(runner.FOUND_DOWN)]
    assert "its state then: exited" in note and "does not start it again" in note
    assert f"note: {note}" in capsys.readouterr().out  # the log says so too
    assert "**The live server had stopped by itself**" in (path.parent / "results.md").read_text()


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


# ---- the scale round (mode scaleout) ----

SLO_REPORT = json.loads((FIXTURES / "guidellm-report-v0.8.1-slo.json").read_text())


def surge_report(rate: float, attainment: float | None, goodput: float | None, successful: int = 40,
                 errored: int = 0) -> dict:
    """A surge's GuideLLM report. The fixture is a real GuideLLM v0.8.1 report with latency goals, of a concurrent
    run; a constant-rate run differs in its strategy block, which is written here by hand."""
    data = json.loads(json.dumps(SLO_REPORT))
    bench = data["benchmarks"][0]
    bench["config"]["strategy"] = {"type_": "constant", "rate": rate, "max_concurrency": None, "rampup_duration": 0.0}
    metrics = bench["metrics"]
    metrics["request_totals"].update(successful=successful, errored=errored, total=successful + errored + 1)
    metrics["slo_attainment"] = attainment
    metrics["slo_determined_requests"] = successful + errored if attainment is not None else 0
    if goodput is None:
        del metrics["request_goodput"]
    else:
        metrics["request_goodput"]["successful"]["mean"] = goodput
    return data


class FakeClock:
    """The runner's wall clock: moves only when the runner sleeps."""

    def __init__(self) -> None:
        self.now = 1_794_000_000.0

    def wall(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakeRunning:
    """GuideLLM started in the background: ends when the runner asks for its report."""

    def __init__(self, path: Path, data: dict, ended=lambda: False, stalls: bool = False) -> None:
        self.path, self.data, self.ended, self.code, self.stopped = path, data, ended, None, False
        self.stalls = stalls  # GuideLLM hangs at the end of the surge: finish() times out
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent / "guidellm.log").write_text(f"surge {'stalled' if stalls else 'finished'}\n")

    def poll(self):
        return 0 if self.ended() else self.code

    def finish(self):
        if self.stalls:
            self.stop()
            raise guidellm.LoadError("GuideLLM did not finish within 1 s and was stopped; see guidellm.log",
                                     timed_out=True)
        self.path.write_text(json.dumps(self.data))
        self.code = 0
        return self.path

    def stop(self):
        self.stopped, self.code = True, -15


class FakeRouter:
    """The router: the first stats call sees the surge's first request; the replica added later gets `share`."""

    def __init__(self, targets, clock: FakeClock, share: float) -> None:
        self.url, self.clock, self.share, self.stopped = "http://router.test:8787", clock, share, False
        self.targets, self.first_at, self.joined = list(targets), None, None

    def stats(self):
        self.first_at = self.first_at or self.clock.wall()
        sent2 = round(40 * self.share) if self.joined else 0
        backends = [{"url": self.targets[0], "sent": 40 - sent2, "finished": 40 - sent2, "failed": 0, "cancelled": 1,
                     "max_in_flight": 12, "first_at": self.first_at, "last_at": self.first_at + 39}]
        if self.joined:
            backends.append({"url": self.targets[1], "sent": sent2, "finished": sent2, "failed": 0, "cancelled": 0,
                             "max_in_flight": 3, "first_at": self.joined + 0.1, "last_at": self.first_at + 39})
        return {"policy": "least-busy", "first_request_at": self.first_at, "backends": backends}

    def add(self, target):
        self.targets.append(target)
        self.joined = self.clock.wall()
        return {"url": target, "joined_at": self.joined}

    def stop(self):
        self.stopped = True


class ScaleBackend(FakeBackend):
    """Two replicas per scored round, a router and a background surge. Side B meets the goals more often. `empty`:
    round index -> a surge in which no request finished; `late`: the surge has ended before replica 2 answers;
    `never_ready`: replica 2 never lists the model, and the runner's clock jumps back once the surge starts, so its
    time limit has passed at once. `dies`: (round index, replica) -> how many of that replica's starts exit at vLLM's
    start-up memory check. `stalls`: round index -> how many of its surges' GuideLLM runs time out at the end."""

    def __init__(self, clock: FakeClock, empty: tuple = (), late: bool = False, never_ready: int | None = None,
                 dies: dict | None = None, stalls: dict | None = None):
        super().__init__(dies=dies)
        self.clock, self.empty, self.late, self.never_ready = clock, empty, late, never_ready
        self.stalls, self.surges = stalls or {}, {}
        self.routers, self.loads, self.replica2_started = [], [], False
        self.rewind = False

    def start(self, plan, round_, side, spec, replica=1):
        self.started.append((round_["index"], round_["side"], replica))
        self.replica2_started = replica == 2
        key = (round_["index"], replica)
        tries = self.attempts[key] = self.attempts.get(key, 0) + 1
        return {"round": round_, "side": side["key"], "replica": replica, "dies": tries <= self.dies.get(key, 0)}

    def endpoint(self, handle, deadline):
        return "http://dead.test:8000" if handle["dies"] else f"http://side-{handle['replica']}.test:8000"

    def router(self, plan, targets, out_dir):
        self.routers.append(FakeRouter(targets, self.clock, 0.25 if not self.late else 0.0))
        return self.routers[-1]

    def start_load(self, plan, round_, target, out_dir):
        assert target == "http://router.test:8787"
        if round_["index"] == self.never_ready:
            self.rewind = True  # from here the runner's deadlines lie behind the probe's clock
        good = 0.9 if round_["side"] == "B" else 0.6
        good *= 1.0 + 0.001 * round_["index"]
        data = (surge_report(10, None, None, successful=0) if round_["index"] in self.empty
                else surge_report(10, good, 10 * good))
        surge = self.surges[round_["index"]] = self.surges.get(round_["index"], 0) + 1
        self.loads.append(FakeRunning(out_dir / "guidellm.json", data,
                                      ended=lambda: self.late and self.replica2_started,
                                      stalls=surge <= self.stalls.get(round_["index"], 0)))
        return self.loads[-1]

    def launch_facts(self, handle):
        return {"env": {"VLLM_X": "1", "HF_TOKEN": "hf_secret", "PATH": "/usr/bin"}, "env_source": "process",
                "image_entrypoint": ["/usr/bin/tini", "--"], "image_cmd": None}

    def logs(self, handle):
        if handle["dies"]:
            return "2026-01-01T00:00:00Z vLLM API server version 0.0.1\n" + PROFILING_LOG
        if handle["replica"] == 1 and handle["round"]["kind"] == "scored":
            ts = self.routers[-1].first_at + 12.5
            return ("2026-01-01T00:00:00Z vLLM API server version 0.0.1\n"
                    f'2026-01-01T00:00:01Z {{"event":"rhvp-scale","ts":{ts},"app":"llm","deployment":"d",'
                    '"from":1,"to":2,"reason":"target"}\n')
        return "2026-01-01T00:00:00Z vLLM API server version 0.0.1\n"

    def stop(self, handle):
        self.stopped.append((handle["round"]["index"], handle["replica"]))


def scale_run(tmp_path, backend: ScaleBackend, clock: FakeClock, **extra):
    plan = plans.build(scale_config(**extra), "docker")
    mono = time.monotonic
    the_runner = runner.Runner(plan, backend, tmp_path, wall=clock.wall, sleep=clock.sleep,
                               clock=lambda: -1e9 if backend.rewind else mono(),
                               client=lambda url: httpx.Client(base_url=url, transport=server(
                                   "another-model" if backend.rewind and "side-2" in url else "m")))
    return the_runner.run(), the_runner


def test_a_scale_out_run(tmp_path):
    clock = FakeClock()
    backend = ScaleBackend(clock)
    results, the_runner = scale_run(tmp_path, backend, clock)
    jsonschema.validate(results, SCHEMA)
    assert results["run"]["status"] == "complete" and results["plan"]["mode"] == "scaleout"
    warm, scored = results["rounds"][0], results["rounds"][2]
    assert warm["kind"] == "warmup" and "scale" not in warm and [l["replica"] for l in warm["launches"]] == [1]
    assert [(i, r) for i, _, r in backend.started[:4]] == [(1, 1), (2, 1), (3, 1), (3, 2)]
    assert sorted(backend.stopped) == sorted({(i, 1) for i in range(1, 17)} | {(i, 2) for i in range(3, 17)})
    assert all(r.stopped for r in backend.routers) and len(backend.routers) == 14

    scale = scored["scale"]
    assert scale["step_after_s"] == 5 and 5 <= scale["step_at_s"] < 5.5  # the forced step, on the runner's clock
    assert scale["new_replica"]["joined_during_surge"] is True and scale["new_replica"]["joined_at_s"] >= 5
    assert scale["new_replica_share"] == 0.25
    assert [b["replica"] for b in scale["router"]["backends"]] == [1, 2]
    assert scale["router"]["backends"][1]["sent"] == 10 and scale["router"]["backends"][0]["first_s"] == 0
    assert scale["load_triggered_s"] == 12.5 and scored["metrics"]["scale.load_triggered_s"] == 12.5
    assert [(e["from"], e["to"], e["reason"]) for e in scale["events"]] == [(1, 2, "target")]
    level = scored["levels"][0]
    assert level["concurrency"] is None and level["rate"] == 10 and level["energy"]["source"] == "none"
    assert level["slo_attainment"] == pytest.approx(0.6 * 1.003)
    assert set(scored["metrics"]) == {"startup.ready_s", "startup.first_answer_s", "scale.ready_s",
                                      "scale.first_answer_s", "scale.load_triggered_s", "surge.slo_attainment",
                                      "surge.goodput_rps", "surge.ttft_ms_median", "surge.itl_ms_median",
                                      "surge.output_tokens_per_s"}
    assert scored["artifacts"] == {"server_log": "rounds/03-A-scored/server.log",
                                   "server_log_2": "rounds/03-A-scored/server-2.log",
                                   "load_report": "rounds/03-A-scored/guidellm.json"}
    first, second = scored["launches"]
    assert (first["replica"], second["replica"]) == (1, 2) and first["entrypoint"] == "image"
    assert first["env"] == {"VLLM_X": "1"} and first["env_dropped"] == ["HF_TOKEN"] and first["env_source"] == "process"
    assert first["image_entrypoint"] == ["/usr/bin/tini", "--"] and "hf_secret" not in the_runner.path.read_text()

    rows = {row["id"]: row for row in results["analysis"]["metrics"]}
    assert rows["surge.slo_attainment"]["role"] == "primary" and rows["surge.slo_attainment"]["verdict"] == "FASTER"
    assert rows["surge.slo_attainment"]["p_adjusted"] == pytest.approx(2 * 2 ** -6)  # Holm over the 2 primaries
    assert rows["surge.goodput_rps"]["verdict"] == "FASTER" and rows["surge.goodput_rps"]["role"] == "secondary"
    assert rows["scale.first_answer_s"]["role"] == "primary" and rows["scale.first_answer_s"]["n_pairs"] == 7
    assert rows["surge.ttft_ms_median"]["verdict"] == "TIE"


def test_a_surge_in_which_nothing_finished_is_a_pair_lost_to_that_side(tmp_path):
    # Round 4 is side B of pair 1 (A B): no request finished in its surge, so its attainment and goodput are 0.
    clock = FakeClock()
    results, _ = scale_run(tmp_path, ScaleBackend(clock, empty=(4,)), clock)
    jsonschema.validate(results, SCHEMA)
    empty = results["rounds"][3]
    assert (empty["side"], empty["pair"]) == ("B", 1)
    assert empty["metrics"]["surge.slo_attainment"] == 0 and empty["metrics"]["surge.goodput_rps"] == 0
    assert "surge.ttft_ms_median" not in empty["metrics"] and empty["levels"][0]["usable"] is False
    assert "count as 0" in empty["levels"][0]["note"]
    rows = {row["id"]: row for row in results["analysis"]["metrics"]}
    for metric in ("surge.slo_attainment", "surge.goodput_rps", "surge.ttft_ms_median"):
        assert rows[metric]["pairs_lost"] == {"A": 0, "B": 1, "both": 0, "other": 0}, metric
        assert rows[metric]["verdict"] == "INCONCLUSIVE" and rows[metric]["n_pairs"] == 6
    assert rows["scale.first_answer_s"]["pairs_lost"]["B"] == 0
    assert results["run"]["status"] == "partial" and results["run"]["error_code"] == "REQUESTS_FAILED"


def test_a_new_replica_that_answers_after_the_surge_is_timed_with_a_share_of_0(tmp_path):
    clock = FakeClock()
    results, _ = scale_run(tmp_path, ScaleBackend(clock, late=True), clock, rounds={"pairs": 1, "warmup": False})
    jsonschema.validate(results, SCHEMA)
    scored = results["rounds"][0]
    assert scored["status"] == "ok" and scored["scale"]["new_replica"]["joined_during_surge"] is False
    assert scored["scale"]["new_replica_share"] == 0 and "scale.first_answer_s" in scored["metrics"]


def test_a_new_replica_that_never_answers_fails_the_round(tmp_path):
    clock = FakeClock()
    backend = ScaleBackend(clock, never_ready=1)
    results, _ = scale_run(tmp_path, backend, clock, rounds={"pairs": 1, "warmup": False})
    jsonschema.validate(results, SCHEMA)
    failed = results["rounds"][0]
    assert failed["status"] == "failed" and failed["error_code"] == "SIDE_NEVER_READY"
    assert backend.loads[0].stopped and backend.routers[0].stopped  # the surge and the router were stopped
    assert sorted(backend.stopped)[:2] == [(1, 1), (1, 2)]  # both replicas taken down
    assert [l["replica"] for l in failed["launches"]] == [1, 2]


def test_a_surge_whose_guidellm_timed_out_runs_the_round_once_more_and_the_round_counts(tmp_path):
    # Round 3 (side A, pair 1): GuideLLM hangs at the end of the first surge. As in mode pairs, one stall of the load
    # tool is tried once more, but from a new first replica: the old one has served that surge, and its prefix cache
    # would hold the same requests. The round starts both replicas again, and every figure comes from the second try.
    clock = FakeClock()
    backend = ScaleBackend(clock, stalls={3: 1})
    results, the_runner = scale_run(tmp_path, backend, clock)
    jsonschema.validate(results, SCHEMA)
    r3 = results["rounds"][2]
    assert (r3["index"], r3["status"], r3["error_code"], r3["load_retries"]) == (3, "ok", None, 1)
    assert all(r["load_retries"] == 0 for r in results["rounds"] if r["index"] != 3)
    assert backend.surges[3] == 2 and backend.loads[0].stopped  # the stalled GuideLLM was stopped
    assert [s for s in backend.started if s[0] == 3] == [(3, "A", 1), (3, "A", 2), (3, "A", 1), (3, "A", 2)]
    assert [s for s in backend.stopped if s[0] == 3][:2] == [(3, 2), (3, 1)]  # the first try, before the second
    assert backend.stopped.count((3, 1)) == 2 and backend.stopped.count((3, 2)) == 2
    assert all(r.stopped for r in backend.routers)
    assert {"startup.ready_s", "scale.first_answer_s", "surge.slo_attainment"} <= set(r3["metrics"])
    assert r3["startup"]["ready_s"] >= 0 and 5 <= r3["scale"]["step_at_s"] < 5.5
    assert r3["scale"]["new_replica_share"] == 0.25
    assert any("run once more from a new first replica, with the same requests" in note for note in r3["notes"])
    assert "stalled" in (the_runner.dir / r3["artifacts"]["guidellm_try_1"]).read_text()
    assert "finished" in (the_runner.dir / "rounds/03-A-scored/guidellm.log").read_text()
    for key in ("server_log_try_1", "server_log_2_try_1", "server_log", "server_log_2", "load_report"):
        assert (the_runner.dir / r3["artifacts"][key]).is_file(), key
    assert r3["artifacts"]["server_log_try_1"] == "rounds/03-A-scored/server.try-1.log"
    assert [(l["replica"], l["discarded"]) for l in r3["launches"]] == [(1, True), (2, True), (1, False), (2, False)]
    assert r3["start_retries"] == 0 and "started again after vLLM" not in (the_runner.dir / "results.md").read_text()
    rows = {row["id"]: row for row in results["analysis"]["metrics"]}
    assert rows["surge.slo_attainment"]["verdict"] == "FASTER" and rows["surge.slo_attainment"]["n_pairs"] == 7
    assert all(row["pairs_lost"] == {"A": 0, "B": 0, "both": 0, "other": 0} for row in rows.values())
    assert results["run"]["status"] == "complete" and results["run"]["error_code"] is None
    text = (the_runner.dir / "results.md").read_text()
    assert "(the whole round ran once more from a new first replica, with the same requests" in text
    assert "- round 3 (side A, scored): ran again, then finished" in text
    head = "".join(show._head(results, [], []))
    assert "Rounds whose load ran again after GuideLLM timed out: 3 (side A)." in head
    assert "from a new first replica" in head


def test_a_memory_check_retry_of_a_discarded_try_is_named_as_that_replica(tmp_path):
    # Round 3: replica 2's first start stops at vLLM's memory check, its second start works, then GuideLLM stalls and
    # the round runs again. Only the discarded replica 2 was started again, and the reports say so.
    clock = FakeClock()
    backend = ScaleBackend(clock, dies={(3, 2): 1}, stalls={3: 1})
    results, the_runner = scale_run(tmp_path, backend, clock)
    jsonschema.validate(results, SCHEMA)
    r3 = results["rounds"][2]
    assert (r3["status"], r3["start_retries"], r3["load_retries"]) == ("ok", 1, 1)
    assert [(l["replica"], l["discarded"], l["start_retries"]) for l in r3["launches"]] == [
        (1, True, 0), (2, True, 1), (1, False, 0), (2, False, 0)]
    assert all(l["gpus"] is None and l["env"] == {"VLLM_X": "1"} for l in r3["launches"])
    assert "Error in memory profiling" in (the_runner.dir / r3["artifacts"]["server_log_2_attempt_1_try_1"]).read_text()
    assert "server_log_2_attempt_1" not in r3["artifacts"]
    text = (the_runner.dir / "results.md").read_text()
    assert "- round 3 (side A, scored): replica 2 of the discarded try started 2 times, then finished" in text
    assert "3 (side A, replica 2 of the discarded try)" in "".join(show._head(results, [], []))


class DiesTwice(ScaleBackend):
    """Replica 2 of round 3 stops at vLLM's memory check on its 1st and 3rd starts (one in each try)."""

    def start(self, plan, round_, side, spec, replica=1):
        handle = super().start(plan, round_, side, spec, replica)
        handle["dies"] = (round_["index"], replica) == (3, 2) and self.attempts[(3, 2)] in (1, 3)
        return handle


def test_each_tries_memory_check_log_is_kept(tmp_path):
    clock = FakeClock()
    backend = DiesTwice(clock, stalls={3: 1})
    results, the_runner = scale_run(tmp_path, backend, clock)
    jsonschema.validate(results, SCHEMA)
    r3 = results["rounds"][2]
    assert (r3["status"], r3["start_retries"]) == ("ok", 2)
    assert [(l["replica"], l["discarded"], l["start_retries"]) for l in r3["launches"]] == [
        (1, True, 0), (2, True, 1), (1, False, 0), (2, False, 1)]
    first, second = r3["artifacts"]["server_log_2_attempt_1_try_1"], r3["artifacts"]["server_log_2_attempt_1"]
    assert first == "rounds/03-A-scored/server-2.attempt-1.try-1.log"
    assert second == "rounds/03-A-scored/server-2.attempt-1.log"
    assert all("Error in memory profiling" in (the_runner.dir / path).read_text() for path in (first, second))
    text = (the_runner.dir / "results.md").read_text()
    assert ("- round 3 (side A, scored): replica 2 of the discarded try started 2 times, replica 2 started 2 times, "
            "then finished") in text


def test_a_surge_whose_guidellm_timed_out_twice_is_load_failed(tmp_path):
    clock = FakeClock()
    backend = ScaleBackend(clock, stalls={1: 2})
    results, _ = scale_run(tmp_path, backend, clock, rounds={"pairs": 1, "warmup": False})
    jsonschema.validate(results, SCHEMA)
    failed = results["rounds"][0]
    assert (failed["status"], failed["error_code"]) == ("failed", "LOAD_FAILED")
    assert backend.surges[1] == 2 and backend.started.count((1, "A", 2)) == 2
    assert backend.started.count((1, "A", 1)) == 2 and failed["load_retries"] == 1
    assert sorted(s for s in backend.stopped if s[0] == 1) == [(1, 1), (1, 1), (1, 2), (1, 2)]


def test_a_scale_round_starts_its_new_replica_once_more_after_vllms_memory_check(tmp_path):
    clock = FakeClock()
    backend = ScaleBackend(clock, dies={(3, 2): 1})
    results, the_runner = scale_run(tmp_path, backend, clock)
    jsonschema.validate(results, SCHEMA)
    r3 = results["rounds"][2]
    assert (r3["index"], r3["status"], r3["start_retries"]) == (3, "ok", 1)
    assert [(l["replica"], l["start_retries"]) for l in r3["launches"]] == [(1, 0), (2, 1)]
    assert backend.started.count((3, "A", 2)) == 2 and backend.stopped.count((3, 2)) == 2
    assert backend.stopped.count((3, 1)) == 1  # the first replica kept serving the surge
    assert 5 <= r3["scale"]["step_at_s"] < 5.5  # the first ask, not the second start
    assert "Error in memory profiling" in (the_runner.dir / r3["artifacts"]["server_log_2_attempt_1"]).read_text()
    assert "Error in memory profiling" not in (the_runner.dir / r3["artifacts"]["server_log_2"]).read_text()
    assert "server_log_attempt_1" not in r3["artifacts"]
    assert any("replica 2 was started once more" in note and "first asked for" in note for note in r3["notes"])
    assert "scale.first_answer_s" in r3["metrics"]
    assert all(r["start_retries"] == 0 for r in results["rounds"] if r["index"] != 3)
    assert results["run"]["status"] == "complete"
    assert ("- round 3 (side A, scored): replica 2 started 2 times, then finished"
            in (the_runner.dir / "results.md").read_text())
    head = "".join(show._head(results, [], []))
    assert "started once more" in head and "3 (side A, replica 2)" in head


def test_a_scale_round_starts_its_first_replica_once_more_after_vllms_memory_check(tmp_path):
    clock = FakeClock()
    backend = ScaleBackend(clock, dies={(3, 1): 1})
    results, the_runner = scale_run(tmp_path, backend, clock)
    jsonschema.validate(results, SCHEMA)
    r3 = results["rounds"][2]
    assert (r3["status"], r3["start_retries"]) == ("ok", 1)
    assert [(l["replica"], l["start_retries"]) for l in r3["launches"]] == [(1, 1), (2, 0)]
    assert backend.stopped.count((3, 1)) == 2 and backend.stopped.count((3, 2)) == 1  # nothing stopped twice
    assert "Error in memory profiling" in (the_runner.dir / r3["artifacts"]["server_log_attempt_1"]).read_text()
    assert "- round 3 (side A, scored): started 2 times, then finished" in (the_runner.dir / "results.md").read_text()


def test_a_second_memory_check_failure_of_the_new_replica_fails_the_round(tmp_path):
    clock = FakeClock()
    backend = ScaleBackend(clock, dies={(1, 2): 2})
    results, _ = scale_run(tmp_path, backend, clock, rounds={"pairs": 1, "warmup": False})
    jsonschema.validate(results, SCHEMA)
    failed = results["rounds"][0]
    assert (failed["status"], failed["error_code"], failed["start_retries"]) == ("failed", "SIDE_FAILED", 1)
    assert backend.started.count((1, "A", 2)) == 2
    assert sorted(s for s in backend.stopped if s[0] == 1) == [(1, 1), (1, 2), (1, 2)]  # both tries, and replica 1
    assert backend.loads[0].stopped and backend.routers[0].stopped
    assert [(l["replica"], l["start_retries"]) for l in failed["launches"]] == [(1, 0), (2, 1)]
