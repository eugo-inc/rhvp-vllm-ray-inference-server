# SPDX-License-Identifier: Apache-2.0
"""rhvp_compare end to end without a platform: the plan, the launch loop with a fake backend and a fake server, the
results file against results.schema.json, the analysis, the markdown report and the command line.

The fake backend hands out one in-process server per launch (httpx.MockTransport) and a GuideLLM report made from
the fixture. Every launch draws its own noise of about 1 % (seeded by its index), so a side and its twin land a
little apart, as two launches of one image do; side B's inter-token latency is 20 % higher and its throughput 20 %
lower. GPU power is a constant 100 W.

Run with: python -m pytest -q tests/test_compare_runner.py
"""

import json
import os
import random
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
    __main__ as cli, analyze, energy, guidellm, plan as plans, report, runner, show, summary, words)

FIXTURES = ROOT / "tests" / "fixtures"
SCHEMA = json.loads((ROOT / "notebooks" / "rhvp_compare" / "results.schema.json").read_text())
REPORT = json.loads((FIXTURES / "guidellm-report-v0.8.1.json").read_text())
#: The launches of side B in the default plan (8 rounds, warm-up first), by round: rounds 1 and 2 are launches 5
#: and 8.
B_LAUNCHES = (5, 8, 11, 18, 20, 25, 30, 31)


def config(**extra) -> dict:
    out = {"runName": "fake-run", "model": {"id": "m", "source": "org/m", "revision": ""},
           "sides": [{"name": "old", "image": "registry.example/server:1"},
                     {"name": "new", "image": "registry.example/server:2"}],
           "rounds": {"count": 8, "warmup": True, "twins": "both"},
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

    def __init__(self, fail_launch: int | None = None, hang_once: bool = False, fail_code: str = runner.SIDE_FAILED,
                 load_error: bool = False, term_in_launch: int | None = None, errored: dict | None = None,
                 nodes: tuple[str, ...] = ("node-1",), zero: dict | None = None, dies: dict | None = None,
                 dies_log: str = PROFILING_LOG, hangs: dict | None = None, hang_report: bool = False):
        self.fail_launch, self.hang_once, self.fail_code = fail_launch, hang_once, fail_code
        self.hangs = dict(hangs or {})  # launch index -> how many of its loads time out
        self.hang_report = hang_report  # a load that times out leaves a partial guidellm.json
        self.loads: list = []  # (launch index, the start of the launch's side that each load ran against)
        self.gpu_checks = []  # the number of starts so far, at each GPU check
        self.load_error, self.term_in_launch = load_error, term_in_launch
        self.errored = errored or {}  # launch index -> the concurrency level where 40 requests fail
        self.zero = zero or {}  # launch index -> the concurrency level whose requests finish with no output token
        self.dies, self.dies_log = dies or {}, dies_log  # launch index -> how many of its starts exit, with that log
        self.attempts: dict[int, int] = {}
        self.nodes = nodes  # the node of launch i is nodes[i % len(nodes)]
        self.started, self.stopped, self.prepared, self.cleaned = [], [], False, False
        self.events: list = []

    def prepare(self, plan):
        self.prepared = True

    def describe(self):
        return {"node": {"name": None, "arch": "amd64", "instance_type": None, "os_image": "test",
                         "kernel": "1", "container_runtime": "fake"}}

    def gpu_free(self, plan, side):
        self.gpu_checks.append(len(self.started))

    def start(self, plan, launch, side, spec):
        if launch["index"] == self.fail_launch:
            raise runner.RoundError("the side never got a node", self.fail_code)
        self.started.append((launch["index"], launch["side"]))
        tries = self.attempts[launch["index"]] = self.attempts.get(launch["index"], 0) + 1
        return {"launch": launch, "side": side["key"], "dies": tries <= self.dies.get(launch["index"], 0)}

    def endpoint(self, handle, deadline):
        return "http://dead.test:8000" if handle.get("dies") else "http://side.test:8000"

    def alive(self, handle):
        return "the server exited with code 1" if handle.get("dies") else None

    def run_load(self, plan, launch, target, out_dir):
        if launch["index"] == self.term_in_launch:  # as if the Job were deleted mid-load
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(30)
            raise AssertionError("SIGTERM did not cut the launch short")
        if self.load_error:
            raise guidellm.LoadError("GuideLLM exited with code 2; see guidellm.log")
        self.loads.append((launch["index"], self.attempts.get(launch["index"])))
        if self.hang_once or self.hangs.get(launch["index"]):
            if self.hang_once:
                self.hang_once = False
            else:
                self.hangs[launch["index"]] -= 1
            (out_dir / "guidellm.log").write_text("stalled\n")
            if self.hang_report:
                (out_dir / "guidellm.json").write_text('{"partial": true}')
            raise guidellm.LoadError("GuideLLM did not finish within 1 s and was stopped; see guidellm.log",
                                     timed_out=True)
        (out_dir / "guidellm.log").write_text("finished\n")
        data = json.loads(json.dumps(REPORT))
        slower = 1.2 if launch["of"] == "B" and launch["kind"] == "scored" else 1.0
        rng = random.Random(launch["index"])  # every launch its own noise: a twin lands near its side, not on it
        itl, tps, ttft = (1.0 + rng.gauss(0.0, 0.01) for _ in range(3))
        for bench in data["benchmarks"]:
            metrics = bench["metrics"]
            if bench["config"]["strategy"]["streams"] == self.errored.get(launch["index"]):
                metrics["request_totals"]["errored"] = 40
            metrics["inter_token_latency_ms"]["successful"]["median"] *= slower * itl
            metrics["output_tokens_per_second"]["total"]["mean"] /= slower * tps
            metrics["time_to_first_token_ms"]["successful"]["median"] *= ttft
            if bench["config"]["strategy"]["streams"] == self.zero.get(launch["index"]):
                metrics["output_tokens_per_second"]["total"]["mean"] = 0.0
        path = out_dir / "guidellm.json"
        path.write_text(json.dumps(data))
        return path

    def sampler(self, plan, handle):
        return FakeSampler()

    def logs(self, handle):
        return "2026-01-01T00:00:00Z vLLM API server version 0.0.1\n" + (self.dies_log if handle.get("dies") else "")

    def facts(self, handle):
        node = self.nodes[handle["launch"]["index"] % len(self.nodes)]
        return {"node_name": node, "image_digest": f"registry.example/server@sha256:{handle['side'] * 64}",
                "gpu": {"product": "Test GPU", "driver": "1", "memory_mib": 1, "count": 1, "source": "fake"}}

    def stop(self, handle):
        self.stopped.append(handle["launch"]["index"])
        self.events.append(("stop", handle["launch"]["index"]))

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


ONE_ROUND = {"count": 1, "warmup": False}


def run(tmp_path, backend=None, model="m", clock=None, deadline_s=None, **extra) -> tuple[dict, FakeBackend,
                                                                                        runner.Runner]:
    backend = backend or FakeBackend()
    plan = plans.build(config(**extra), "docker")
    kwargs = {"clock": clock} if clock else {}
    the_runner = runner.Runner(plan, backend, tmp_path, sleep=lambda s: None, deadline_s=deadline_s,
                               client=lambda url: httpx.Client(base_url=url, transport=server(model)), **kwargs)
    return the_runner.run(), backend, the_runner


def rows_of(results) -> dict:
    return {row["case"]: row for row in results["analysis"]["comparisons"]}


def test_launch_order_rotates_and_reverses_with_the_warmup_first():
    plan = plans.build(config(), "docker")
    launches = plan["launches"]
    assert len(launches) == 2 + 8 * 4 == 34 and [x["index"] for x in launches] == list(range(1, 35))
    assert [(x["side"], x["kind"], x["round"]) for x in launches[:2]] == [("A", "warmup", None), ("B", "warmup", None)]
    by_round = {r: [x["side"] for x in launches if x["round"] == r] for r in range(1, 9)}
    assert by_round[1] == ["A", "A~same", "B", "B~same"]
    assert by_round[2] == ["A~same", "B", "B~same", "A"]
    assert by_round[5] == ["B~same", "B", "A~same", "A"]  # the second cycle runs reversed
    assert by_round[8] == ["B", "A~same", "A", "B~same"]
    assert tuple(x["index"] for x in launches if x["side"] == "B" and x["kind"] == "scored") == B_LAUNCHES
    assert all(x["of"] == x["side"][0] for x in launches)  # a twin launches its own side
    assert plan["balance"]["balanced"] and plan["balance"]["step"] == 8
    assert plan["balance"]["pairs"]["B vs A"] == {"first_ran_first": 4, "second_ran_first": 4,
                                                  "position_difference_sum": 0}
    assert not plans.build(config(rounds={"count": 12}), "docker")["balance"]["balanced"]


def test_plan_rejects_bad_configs():
    with pytest.raises(plans.PlanError, match="exactly two sides"):
        plans.build({**config(), "sides": [config()["sides"][0]]})
    with pytest.raises(plans.PlanError, match="runName"):
        plans.build({**config(), "runName": "Bad Name"})
    with pytest.raises(plans.PlanError, match="different names"):
        plans.build({**config(), "sides": [config()["sides"][0]] * 2})
    with pytest.raises(plans.PlanError, match="model.source"):
        plans.build({**config(), "model": {"id": "m"}})
    with pytest.raises(plans.PlanError, match="rounds.twins"):
        plans.build(config(rounds={"count": 8, "twins": "C"}))
    with pytest.raises(plans.PlanError, match="rounds.count"):
        plans.build(config(rounds={"count": 0}))
    # The keys of the earlier rule say what replaced them.
    for section, key in (("rounds", "pairs"), ("rounds", "aaPairs"), ("analysis", "tieBand"),
                         ("analysis", "correction"), ("analysis", "test"), ("analysis", "minPairs"),
                         ("analysis", "ciLevel")):
        with pytest.raises(plans.PlanError, match=rf"{section}\.{key} is no longer read"):
            plans.build(config(**{section: {key: 1}}))


def test_plan_declares_everything_before_the_run():
    plan = plans.build(config(), "openshift")
    assert plan["energy"]["source"] == "dcgm" and plans.build(config(), "docker")["energy"]["source"] == "nvidia-smi"
    assert plan["energy"]["label"] == "GPU-only power" and plan["energy"]["scope"] == "gpu"
    assert [c["id"] for c in plan["cases"]][:3] == ["startup/ready_s", "startup/first_answer_s", "c1/ttft_ms_median"]
    # Four primary cases share the correction: time to first answer, and TTFT, throughput and tokens per joule at
    # the highest level; the other six are secondary.
    assert plan["analysis"]["primary"] == ["startup/first_answer_s", "c4/ttft_ms_median", "c4/output_tokens_per_s",
                                           "c4/tokens_per_joule[gpu]"]
    assert [c["id"] for c in plan["cases"] if c["role"] == "primary"] == plan["analysis"]["primary"]
    assert sum(c["role"] == "secondary" for c in plan["cases"]) == 6
    kinds = {c["id"]: c["kind"] for c in plan["cases"]}
    assert (kinds["c4/ttft_ms_median"], kinds["c4/output_tokens_per_s"], kinds["c4/tokens_per_joule[gpu]"]) == (
        "time", "rate", "efficiency")
    groups = {c["id"]: c["group"] for c in plan["cases"]}
    assert groups["c4/ttft_ms_median"] == "primary:ttft_ms_median"
    assert groups["c1/ttft_ms_median"] == "secondary:ttft_ms_median"
    assert plan["power"] == {"family_size": 4, "min_rounds": 8, "reachable": True, "both_twins": True,
                             "must_agree": 8, "max_primary_cases": 6,
                             "plain": "corrected for testing 4 cases at once"}
    assert plan["analysis"]["rule_fields"]["bar"] == "sum_of_both" and plan["analysis"]["ratio"] == "B/A"
    mock = plans.build(config(sides=[{"name": "a", "image": "x:1", "preset": "guidellm-mock", "gpu": False},
                                     {"name": "b", "image": "x:2", "preset": "guidellm-mock", "gpu": False}]))
    assert mock["analysis"]["primary"] == ["startup/first_answer_s", "c4/ttft_ms_median", "c4/output_tokens_per_s"]
    assert plan["run_id"] == f"fake-run-{plans.digest(config())}"
    assert plans.build(config(), "docker")["run_id"] != plans.build(config(runName="other"), "docker")["run_id"]
    few = plans.build(config(rounds={"count": 7}), "docker")
    assert not few["power"]["reachable"] and few["power"]["must_agree"] is None
    one_twin = plans.build(config(rounds={"count": 8, "twins": "A"}), "docker")
    assert [e["key"] for e in one_twin["lineup"]] == ["A", "A~same", "B"] and len(one_twin["launches"]) == 26
    assert not one_twin["power"]["reachable"] and not one_twin["power"]["both_twins"]
    assert any("without a twin for both sides" in line for line in runner.plan_lines(one_twin))
    assert any("12 rounds do not balance" in line
               for line in runner.plan_lines(plans.build(config(rounds={"count": 12}), "docker")))


def test_plan_takes_a_declared_primary_list():
    plan = plans.build(config(analysis={"primary": ["c1/itl_ms_median"]}), "docker")
    assert plan["power"]["family_size"] == 1 and plan["analysis"]["primary"] == ["c1/itl_ms_median"]
    assert plan["power"]["min_rounds"] == 6
    with pytest.raises(plans.PlanError, match="does not have: c64/itl_ms_median"):
        plans.build(config(analysis={"primary": ["c64/itl_ms_median"]}), "docker")
    for bad in ([], ["c1/itl_ms_median", "c1/itl_ms_median"], "c1/itl_ms_median", [{"id": "x"}]):
        with pytest.raises(plans.PlanError, match="analysis.primary"):
            plans.build(config(analysis={"primary": bad}), "docker")


def test_a_full_run(tmp_path):
    results, backend, the_runner = run(tmp_path)
    jsonschema.validate(results, SCHEMA)
    on_disk = json.loads(the_runner.path.read_text())
    assert on_disk == json.loads(json.dumps(results))
    assert results["run"]["status"] == "complete" and results["run"]["error_code"] is None
    assert backend.prepared and backend.cleaned
    launches = results["launches"]
    assert len(launches) == 34 and all(x["status"] == "ok" and x["error_code"] is None for x in launches)
    assert sorted(backend.stopped) == list(range(1, 35))
    warm = launches[0]
    assert warm["kind"] == "warmup" and warm["round"] is None and warm["levels"] == []
    assert set(warm["values"]) == {"startup/ready_s", "startup/first_answer_s"}
    scored, twin = launches[2], launches[3]
    assert (scored["index"], scored["round"], scored["pos"], scored["side"], scored["of"]) == (3, 1, 1, "A", "A")
    assert (twin["side"], twin["of"], twin["pos"]) == ("A~same", "A", 2)
    assert scored["server_version"] == {"vllm": "0.0.1-test", "source": "endpoint"}
    assert scored["levels"][0]["energy"]["method"] == "power_mean"
    assert scored["levels"][0]["energy"]["mean_power_w"] == pytest.approx(100.0)
    assert scored["artifacts"] == {"server_log": "launches/03-A-scored/server.log",
                                   "load_report": "launches/03-A-scored/guidellm.json"}
    assert twin["artifacts"]["server_log"] == "launches/04-A-same-scored/server.log"
    assert (the_runner.dir / scored["artifacts"]["server_log"]).is_file()
    assert set(scored["values"]) == {c["id"] for c in results["plan"]["cases"]}
    assert twin["image_digest"].endswith("A" * 64)  # the twin runs its side's image
    prov = results["provenance"]
    assert set(prov["sides"]) == {"A", "B"} and prov["sides"]["B"]["image_digest"].endswith("B" * 64)
    assert prov["gpu"]["product"] == "Test GPU"
    assert prov["load_tool"]["version"] == "0.8.1" and prov["seeds"] == {"load": 42, "server": 0}

    analysis = results["analysis"]
    assert analysis["rule"]["family_size"] == 4 and analysis["rule"]["plain"] == "corrected for testing 4 cases at once"
    assert analysis["rule"]["min_rounds"] == {"primary": 8, "secondary": 6}
    rows = rows_of(results)
    tps = rows["c4/output_tokens_per_s"]
    assert (tps["role"], tps["outcome"], tps["verdict"], tps["reason"]) == ("primary", "worse", "WORSE", None)
    # All 8 rounds lean the same way: 2 * 2^-8 uncorrected, and Holm over the 4 primary cases makes it 4 times that.
    assert tps["p_raw"] == 2 ** -7 and tps["p_raw_fraction"] == [2, 256]
    assert tps["p_corrected"] == pytest.approx(4 * 2 ** -7)
    assert tps["ratio"] == pytest.approx(1 / 1.2, rel=0.03) and tps["gap"] > tps["bar"] > 0
    assert tps["floor"]["source"] == "twins" and tps["floor"]["subject"]["n"] == 8
    assert rows["c4/tokens_per_joule[gpu]"]["verdict"] == "WORSE"
    # Secondary: the same rule on the uncorrected p-value; p_corrected is null.
    itl = rows["c1/itl_ms_median"]
    assert (itl["role"], itl["verdict"], itl["p_corrected"], itl["p_raw"]) == ("secondary", "SLOWER", None, 2 ** -7)
    assert itl["ratio"] == pytest.approx(1.2, rel=0.03) and itl["bar"] < 0.05
    assert rows["c4/ttft_ms_median"]["verdict"] not in ("FASTER", "SLOWER")  # no difference was put in
    assert all(row["n_rounds"] == 8 and row["lost"] == {"A": 0, "B": 0} for row in rows.values())
    assert all(sum(row["order_split"]) == row["n_rounds"] for row in rows.values())
    outcomes = analysis["outcomes"]
    assert outcomes["primary"]["WORSE"] == 2
    assert sum(outcomes["primary"].values()) == 4  # the time to first answer is real time here: its verdict varies
    assert sum(outcomes["secondary"].values()) == 6
    twins = {(t["side"], t["case"]): t for t in analysis["same_vs_same"]}
    assert len(twins) == 2 * 10
    assert twins[("B", "c1/itl_ms_median")]["n"] == 8 and twins[("B", "c1/itl_ms_median")]["floor"] < 0.02
    assert twins[("A", "c1/itl_ms_median")]["twin"] == "A~same"
    assert analysis["refusals"] == [] and analysis["nodes"] == ["node-1"]


def test_a_failed_launch_is_kept_and_the_run_goes_on(tmp_path):
    results, _, _ = run(tmp_path, FakeBackend(fail_launch=5))  # side B in round 1
    jsonschema.validate(results, SCHEMA)
    failed = [x for x in results["launches"] if x["status"] != "ok"]
    assert len(failed) == 1 and failed[0]["index"] == 5 and failed[0]["error"] == "the side never got a node"
    assert failed[0]["error_code"] == "SIDE_FAILED"
    assert results["run"]["status"] == "partial" and len(results["launches"]) == 34
    assert results["run"]["error_code"] == "LAUNCHES_FAILED"
    # The 7 rounds left would flatter side B: no verdict, and the lost round is counted against B.
    itl = rows_of(results)["c1/itl_ms_median"]
    assert (itl["n_rounds"], itl["lost"], itl["outcome"], itl["reason"]) == (7, {"B": 1, "A": 0}, "no_verdict",
                                                                             "side_failed")
    # A launch lost to something that is not the side's doing (the GPU was not free) is only one round fewer.
    not_hers, _, _ = run(tmp_path / "gpu", FakeBackend(fail_launch=5, fail_code=runner.GPU_NOT_FREE))
    itl = rows_of(not_hers)["c1/itl_ms_median"]
    assert (itl["n_rounds"], itl["lost"], itl["verdict"], itl["p_raw"]) == (7, {"B": 0, "A": 0}, "SLOWER", 2 ** -6)


@pytest.mark.parametrize("backend, model, code", [
    (FakeBackend(fail_launch=1, fail_code=runner.GPU_NOT_FREE), "m", "GPU_NOT_FREE"),
    (FakeBackend(load_error=True), "m", "LOAD_FAILED"),
    (FakeBackend(), "another-model", "SIDE_NEVER_READY"),
])
def test_each_failure_has_its_code(tmp_path, backend, model, code):
    # A clock far in the past puts every launch's deadline behind the probe's own clock: a side that never lists the
    # model is SIDE_NEVER_READY at once.
    results, _, _ = run(tmp_path, backend, model=model, clock=lambda: -1e9, rounds=ONE_ROUND)
    jsonschema.validate(results, SCHEMA)
    assert results["launches"][0]["status"] == "failed" and results["launches"][0]["error_code"] == code
    assert results["run"]["error_code"] == "LAUNCHES_FAILED"


def test_a_side_that_dies_is_side_failed(tmp_path):
    backend = FakeBackend()
    backend.alive = lambda handle: "the server exited: Error, exit code 1"
    results, _, _ = run(tmp_path, backend, model="another-model", rounds=ONE_ROUND)
    assert results["launches"][0]["error_code"] == "SIDE_FAILED"
    assert results["launches"][0]["error"] == "the server exited: Error, exit code 1"


def test_sigterm_cuts_the_launch_short_and_cleans_up(tmp_path):
    started = time.monotonic()
    results, backend, the_runner = run(tmp_path, FakeBackend(term_in_launch=3))
    assert time.monotonic() - started < 20  # the load's 30 s sleep was interrupted
    jsonschema.validate(results, SCHEMA)
    assert [x["index"] for x in results["launches"]] == [1, 2, 3]  # no launch after the stop
    assert results["launches"][2]["error_code"] == "STOPPED" and 3 in backend.stopped and backend.cleaned
    assert results["run"]["status"] == "partial" and results["run"]["error_code"] == "STOPPED"
    assert json.loads(the_runner.path.read_text())["run"]["error_code"] == "STOPPED"
    assert signal.getsignal(signal.SIGTERM) is not the_runner._on_term


def test_sigterm_near_the_jobs_time_limit_is_run_timeout(tmp_path):
    results, _, _ = run(tmp_path, FakeBackend(term_in_launch=3), deadline_s=60)
    assert results["run"]["error_code"] == "RUN_TIMEOUT" and results["launches"][2]["error_code"] == "STOPPED"


def test_sigterm_between_launches_stops_before_the_next(tmp_path):
    backend = FakeBackend()
    real_stop = backend.stop

    def stop(handle):
        real_stop(handle)
        if handle["launch"]["index"] == 2:
            os.kill(os.getpid(), signal.SIGTERM)  # arrives during teardown: not interruptible, the launch stands
    backend.stop = stop
    results, _, _ = run(tmp_path, backend)
    assert [x["status"] for x in results["launches"]] == ["ok", "ok"]
    assert results["run"]["error_code"] == "STOPPED"


def test_status_is_published_after_every_launch(tmp_path):
    backend = PublishingBackend()
    results, _, the_runner = run(tmp_path, backend, rounds={"count": 2, "warmup": True})
    status_schema = {"$defs": SCHEMA["$defs"], "properties": {"run": SCHEMA["properties"]["run"]},
                     "$ref": "#/$defs/status"}
    assert len(backend.published) == 1 + 10 + 1  # the start, each of the 2 + 2 * 4 launches, the end
    for run_id, status, result_text in backend.published:
        jsonschema.validate(status, status_schema)
        assert run_id == results["run"]["id"] and json.loads(result_text)["run"]["id"] == run_id
    first, last = backend.published[1][1], backend.published[-1][1]
    assert first["phase"] == "running" and first["launches_done"] == 1 and first["eta_seconds"] is not None
    assert first["launches_total"] == 10 and first["completed_rounds"] == 0
    assert last["phase"] == "complete" and last["completed_rounds"] == last["total_rounds"] == 2
    assert last["eta_seconds"] is None
    assert [p["case"] for p in last["primary"]] == results["plan"]["analysis"]["primary"]
    # Two rounds are too few for any verdict, and the status says why.
    assert {(p["verdict"], p["reason"]) for p in last["primary"]} == {("NO VERDICT", "too_few_rounds")}


def test_a_failing_publish_is_a_note_not_a_failure(tmp_path):
    backend = PublishingBackend()
    backend.publish = lambda *a: (_ for _ in ()).throw(RuntimeError("forbidden"))
    results, _, _ = run(tmp_path, backend, rounds=ONE_ROUND)
    assert results["run"]["status"] == "complete"
    assert [n for n in results["run"]["notes"] if "not published" in n] == [
        "the run's status was not published: RuntimeError: forbidden"]


def test_main_run_prints_the_results_and_writes_the_termination_message(tmp_path, capsys, monkeypatch):
    path = tmp_path / "compare.json"
    path.write_text(json.dumps(config(rounds=ONE_ROUND)))
    message = tmp_path / "termination-log"
    original = runner.Runner.__init__

    def init(self, *a, **kw):  # main_run builds its own Runner: give it the fake server
        original(self, *a, client=lambda url: httpx.Client(base_url=url, transport=server()),
                 sleep=lambda s: None, **kw)
    monkeypatch.setattr(runner.Runner, "__init__", init)
    code = runner.main_run(str(path), str(tmp_path / "results"), FakeBackend(fail_launch=2),
                           env={"RHVP_COMPARE_TERMINATION_LOG": str(message)})
    assert code == 1  # partial: the Job reads Failed
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("run fake-run-") and "4 server starts" in out[0]
    begin = out.index(summary.BEGIN)
    assert out[begin + 2] == summary.END
    printed = json.loads(out[begin + 1])
    jsonschema.validate(printed, SCHEMA)
    # Launch 2 is side A's twin: its failure costs the floor a round, not the comparison of B with A.
    assert "  c1/itl_ms_median (secondary): NO VERDICT, single_measurement (rounds 1)" in out
    short = json.loads(message.read_text())
    assert len(message.read_bytes()) <= 4096
    assert short["status"] == "partial" and short["error_code"] == "LAUNCHES_FAILED"
    assert short["completed_rounds"] == 1 and short["total_rounds"] == 1 and short["run_id"] == printed["run"]["id"]


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
               "analysis": {"outcomes": {"primary": {"WORSE": 300}},
                            "comparisons": [{"case": f"c{i}/output_tokens_per_s", "verdict": "WORSE",
                                             "outcome": "worse", "role": "primary"} for i in range(300)]}}
    text = summary.termination(results)
    assert len(text.encode()) <= 4096 and "primary" not in json.loads(text)


def _clock(backend, index):
    """A runner clock that each start of launch `index` moves 100 s back, so a start-up figure says which start it
    came from: about 0 s from the first, about 100 s from the second."""
    return lambda: time.monotonic() - 100 * backend.attempts.get(index, 0)


def test_a_hung_load_runs_the_launch_once_more_from_a_new_start(tmp_path, monkeypatch):
    # Launch 1: GuideLLM stalls and leaves a partial report. Its retry sends the same requests (the same seed), so it
    # runs against a new start of the side, not against the server that has just served them, and the launch's
    # figures come from that try.
    backend = FakeBackend(hangs={1: 1}, hang_report=True)
    results, _, the_runner = run(tmp_path, backend, clock=_clock(backend, 1), rounds=ONE_ROUND)
    jsonschema.validate(results, SCHEMA)
    first = results["launches"][0]
    assert (first["status"], first["error_code"], first["load_timeouts"], first["load_retries"]) == ("ok", None, 1, 1)
    assert (first["starts"], first["start_retries"]) == (2, 0)
    assert [x["load_retries"] for x in results["launches"]] == [1, 0, 0, 0]
    assert backend.started.count((1, "A")) == 2 and backend.stopped.count(1) == 2
    assert [load for load in backend.loads if load[0] == 1] == [(1, 1), (1, 2)]  # the second load: the second start
    assert backend.gpu_checks == [0, 1, 2, 3, 4]  # every start, the second try's too, waits for the GPU first
    assert 99 < first["startup"]["ready_s"] < 150  # the second start's figure, not the first's (about 0 s)
    assert any("ran once more from a new start of the side, with the same requests" in n for n in first["notes"])
    folder = the_runner.dir / "launches/01-A-scored"
    assert first["artifacts"]["guidellm_try_1"] == "launches/01-A-scored/guidellm.try-1.log"
    assert first["artifacts"]["server_log_try_1"] == "launches/01-A-scored/server.try-1.log"
    assert first["artifacts"]["load_report_try_1"] == "launches/01-A-scored/guidellm.try-1.json"
    assert (folder / "guidellm.try-1.log").read_text() == "stalled\n"
    assert (folder / "guidellm.try-1.json").read_text() == '{"partial": true}'  # kept, never read
    assert (folder / "guidellm.log").read_text() == "finished\n"
    assert json.loads((folder / "guidellm.json").read_text())["benchmarks"]  # the second try's report
    assert {"startup/ready_s", "c1/ttft_ms_median"} <= set(first["values"])
    assert results["run"]["status"] == "complete"
    # Every time-out counts against its side, though the launch finished.
    assert results["analysis"]["load_timeouts"] == {"A": 1, "B": 0}
    assert {row["load_timeouts"]["A"] for row in results["analysis"]["comparisons"]} == {1}
    text = (the_runner.dir / "results.md").read_text()
    assert "- launch 1 (A): timed out 1 time, ran again, then finished" in text
    assert "**GuideLLM timed out** 1 times in the launches of side A and 0 times" in text
    assert "Only side A timed out" in text and "started again after vLLM" not in text
    head = "".join(show._head(results, [], []))
    assert "Launches whose GuideLLM timed out: 1 (A): ran again." in head and words.LOAD_RETRIED_HOW in head
    shown = []
    monkeypatch.setenv("RHVP_COMPARE_REPLAY", str(the_runner.path))
    monkeypatch.setattr(show.draw, "show", lambda *parts: shown.extend(parts))
    show.verdicts()
    assert words.LOAD_TIMEOUTS_ONE_SIDED.format(side="A") in "".join(shown)


def test_a_report_left_by_an_earlier_attempt_is_not_kept_as_the_tries(tmp_path):
    plan = plans.build(config(rounds=ONE_ROUND), "docker")
    folder = tmp_path / plan["run_id"] / "launches/01-A-scored"
    folder.mkdir(parents=True)
    (folder / "guidellm.json").write_text('{"stale": true}')  # an earlier, unfinished attempt of the same run
    results, _, _ = run(tmp_path, FakeBackend(hangs={1: 1}), rounds=ONE_ROUND)
    first = results["launches"][0]
    assert first["status"] == "ok" and "load_report_try_1" not in first["artifacts"]
    assert not (folder / "guidellm.try-1.json").exists()


def test_a_load_that_times_out_in_both_tries_is_load_failed(tmp_path):
    backend = FakeBackend(hangs={1: 2})
    results, _, the_runner = run(tmp_path, backend, clock=_clock(backend, 1), rounds=ONE_ROUND)
    jsonschema.validate(results, SCHEMA)
    failed = results["launches"][0]
    assert (failed["status"], failed["error_code"], failed["load_timeouts"], failed["load_retries"]) == (
        "failed", "LOAD_FAILED", 2, 1)
    assert backend.started.count((1, "A")) == 2 and backend.stopped.count(1) == 2
    assert backend.loads.count((1, 1)) == 1 and backend.loads.count((1, 2)) == 1  # one load per start, no third
    assert 99 < failed["startup"]["ready_s"] < 150  # the second start's own figure; nothing of the first try
    assert failed["levels"] == [] and set(failed["values"]) == {"startup/ready_s", "startup/first_answer_s"}
    assert results["analysis"]["load_timeouts"] == {"A": 2, "B": 0}
    text = (the_runner.dir / "results.md").read_text()
    assert "- launch 1 (A): timed out 2 times, ran again, then LOAD_FAILED" in text


class DiesInEachTry(FakeBackend):
    """Launch 3's side stops at vLLM's memory check on its 1st and 3rd starts: once in each try of the launch."""

    def start(self, plan, launch, side, spec):
        handle = super().start(plan, launch, side, spec)
        handle["dies"] = launch["index"] == 3 and self.attempts[3] in (1, 3)
        return handle


def test_each_try_of_a_retried_launch_gets_its_own_memory_check_retry(tmp_path):
    backend = DiesInEachTry(hangs={3: 1})
    results, _, the_runner = run(tmp_path, backend)
    jsonschema.validate(results, SCHEMA)
    l3 = next(x for x in results["launches"] if x["index"] == 3)
    assert (l3["status"], l3["starts"], l3["start_retries"], l3["load_retries"]) == ("ok", 4, 2, 1)
    assert backend.started.count((3, "A")) == 4 and backend.stopped.count(3) == 4
    assert [load for load in backend.loads if load[0] == 3] == [(3, 2), (3, 4)]
    first, second = l3["artifacts"]["server_log_attempt_1_try_1"], l3["artifacts"]["server_log_attempt_1"]
    assert first == "launches/03-A-scored/server.attempt-1.try-1.log"
    assert second == "launches/03-A-scored/server.attempt-1.log"
    assert all("Error in memory profiling" in (the_runner.dir / path).read_text() for path in (first, second))
    assert "Error in memory profiling" not in (the_runner.dir / l3["artifacts"]["server_log_try_1"]).read_text()
    # The first try's note is marked as that try's, and names its log as kept.
    old = [n for n in l3["notes"] if n.startswith("try 1:")]
    assert any("server.attempt-1.try-1.log is the log of the start that stopped" in n for n in old)
    assert any("started once more after vLLM's start-up memory check" in n for n in old)
    text = (the_runner.dir / "results.md").read_text()
    assert "- launch 3 (A): started 4 times in all, then finished" in text
    assert "- launch 3 (A): timed out 1 time, ran again, then finished" in text
    assert "3 (A, 4)" in "".join(show._head(results, [], []))


class StopFailsOnce(FakeBackend):
    """The first teardown of launch 1's side fails, as an API error would."""

    def stop(self, handle):
        super().stop(handle)
        if self.stopped.count(1) == 1 and handle["launch"]["index"] == 1:
            raise runner.RoundError("could not delete the side's pod (HTTP 500)")


class TermInTeardown(FakeBackend):
    """SIGTERM lands while launch 1's side is taken down between its two tries."""

    def stop(self, handle):
        super().stop(handle)
        if self.stopped.count(1) == 1 and handle["launch"]["index"] == 1:
            os.kill(os.getpid(), signal.SIGTERM)


@pytest.mark.parametrize("backend, code", [(StopFailsOnce(hang_once=True), "RUNNER_ERROR"),
                                           (TermInTeardown(hang_once=True), "STOPPED")])
def test_a_launch_cut_short_between_its_tries_keeps_none_of_the_first_try(tmp_path, backend, code):
    results, _, the_runner = run(tmp_path, backend, rounds=ONE_ROUND)
    jsonschema.validate(results, SCHEMA)
    failed = results["launches"][0]
    # The time-out is on record and counts against the side; the load never ran again, so load_retries says 0.
    assert (failed["status"], failed["error_code"], failed["load_timeouts"], failed["load_retries"]) == (
        "failed", code, 1, 0)
    assert failed["startup"] is None and failed["levels"] == [] and failed["values"] == {}
    assert any("did not finish within 1 s" in note for note in failed["notes"])
    assert backend.started.count((1, "A")) == 1 and backend.stopped.count(1) == 2  # the launch's end tore it down
    assert "server_log" not in failed["artifacts"] and "server_log_try_1" in failed["artifacts"]
    assert results["analysis"]["load_timeouts"]["A"] == 1
    text = (the_runner.dir / "results.md").read_text()
    assert f"- launch 1 (A): timed out 1 time, did not run again, then {code}" in text
    assert "ran again" not in text.split("- launch 1 (A): timed out", 1)[1].splitlines()[0]
    if code == "STOPPED":
        assert results["run"]["error_code"] == "STOPPED" and [x["index"] for x in results["launches"]] == [1]


class SamplerFailsFirst(FakeBackend):
    """The power sampler of the first load does not start."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.samplers = 0

    def sampler(self, plan, handle):
        self.samplers += 1
        if self.samplers == 1:
            raise RuntimeError("no power")
        return FakeSampler()


def test_the_first_tries_notes_are_marked_as_that_tries(tmp_path):
    results, _, _ = run(tmp_path, SamplerFailsFirst(hang_once=True), rounds=ONE_ROUND)
    first = results["launches"][0]
    assert first["status"] == "ok" and first["levels"][0]["energy"]["mean_power_w"] == pytest.approx(100.0)
    assert "try 1: power sampling did not start: RuntimeError" in first["notes"]
    assert not any(n.startswith("power sampling") for n in first["notes"])  # the second try's sampler started
    [why] = [n for n in first["notes"] if "the side is to be taken down" in n]
    assert "see guidellm.try-1.log" in why  # the time-out's own text names the kept log, not the second try's


class StartFailsInSecondTry(FakeBackend):
    """Launch 1's side starts once; its second start, after the time-out, fails."""

    def start(self, plan, launch, side, spec):
        if launch["index"] == 1 and self.attempts.get(1):
            raise runner.RoundError("the side's image could not be pulled", runner.SIDE_FAILED)
        return super().start(plan, launch, side, spec)


class GpuBusyInSecondTry(FakeBackend):
    """The second try's GPU check finds the GPU held, after a memory-check start in the first try."""

    def gpu_free(self, plan, side):
        super().gpu_free(plan, side)
        if len(self.gpu_checks) == 3:
            raise runner.RoundError("another program held the GPU", runner.GPU_NOT_FREE)


class TermInSecondGpuCheck(FakeBackend):
    def gpu_free(self, plan, side):
        super().gpu_free(plan, side)
        if len(self.gpu_checks) == 2:
            os.kill(os.getpid(), signal.SIGTERM)


class TermInFirstTriesLog(FakeBackend):
    """SIGTERM lands while the runner reads the first try's server log, before its teardown."""

    def logs(self, handle):
        if handle["launch"]["index"] == 1 and not self.stopped:
            os.kill(os.getpid(), signal.SIGTERM)
        return super().logs(handle)


@pytest.mark.parametrize("backend, code, starts", [
    (StartFailsInSecondTry(hang_once=True), "SIDE_FAILED", 1),
    (GpuBusyInSecondTry(hang_once=True, dies={1: 1}), "GPU_NOT_FREE", 2),
    (TermInSecondGpuCheck(hang_once=True), "STOPPED", 1),
    (TermInFirstTriesLog(hang_once=True), "STOPPED", 1),
])
def test_a_second_try_that_never_loads_says_so(tmp_path, backend, code, starts):
    results, _, the_runner = run(tmp_path, backend, rounds=ONE_ROUND)
    jsonschema.validate(results, SCHEMA)
    failed = results["launches"][0]
    assert (failed["status"], failed["error_code"], failed["load_timeouts"], failed["load_retries"]) == (
        "failed", code, 1, 0)
    assert failed["starts"] == starts == backend.started.count((1, "A"))  # only the starts that happened
    assert failed["startup"] is None and failed["levels"] == [] and failed["values"] == {}
    assert [load for load in backend.loads if load[0] == 1] == [(1, starts)]
    text = (the_runner.dir / "results.md").read_text()
    assert f"- launch 1 (A): timed out 1 time, did not run again, then {code}" in text
    if starts > 1:
        assert f"- launch 1 (A): started {starts} times in all, then {code}" in text
    if code == "STOPPED":  # the teardown kept the first try's log under its own name, and nothing as the launch's
        assert "server_log_try_1" in failed["artifacts"] and "server_log" not in failed["artifacts"]
        assert "server.log" not in {p.name for p in (the_runner.dir / "launches/01-A-scored").iterdir()}


class GpuBusyAfterTheMemoryCheck(FakeBackend):
    """Launch 1's first start stops at vLLM's memory check; the GPU check before its start again fails."""

    def gpu_free(self, plan, side):
        super().gpu_free(plan, side)
        if len(self.gpu_checks) == 2:
            raise runner.RoundError("another program held the GPU", runner.GPU_NOT_FREE)


def test_a_memory_check_retry_counts_only_once_the_side_starts_again(tmp_path):
    results, backend, the_runner = run(tmp_path, GpuBusyAfterTheMemoryCheck(dies={1: 1}), rounds=ONE_ROUND)
    failed = results["launches"][0]
    assert (failed["error_code"], failed["starts"], failed["start_retries"]) == ("GPU_NOT_FREE", 1, 0)
    assert backend.started.count((1, "A")) == 1
    assert not any("started once more after" in note for note in failed["notes"])
    assert any("to be taken down and started once more" in note for note in failed["notes"])
    assert "server_log_attempt_1" in failed["artifacts"]  # the log of the start that stopped is kept
    assert "started again after vLLM" not in (the_runner.dir / "results.md").read_text()


class TermInSecondSampler(FakeBackend):
    """SIGTERM lands while the second try's power sampler starts, before its GuideLLM run."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.samplers = 0

    def sampler(self, plan, handle):
        self.samplers += 1
        if self.samplers == 2:
            os.kill(os.getpid(), signal.SIGTERM)
        return FakeSampler()


def test_load_retries_counts_only_once_the_second_load_begins(tmp_path):
    results, backend, the_runner = run(tmp_path, TermInSecondSampler(hang_once=True), rounds=ONE_ROUND)
    failed = results["launches"][0]
    assert (failed["error_code"], failed["starts"], failed["load_timeouts"], failed["load_retries"]) == (
        "STOPPED", 2, 1, 0)
    assert backend.loads == [(1, 1)]
    assert not any("ran once more" in note for note in failed["notes"])
    assert "- launch 1 (A): timed out 1 time, did not run again, then STOPPED" in (
        the_runner.dir / "results.md").read_text()


def test_a_finished_results_file_of_the_earlier_schema_is_kept_and_the_run_runs(tmp_path):
    plan = plans.build(config(rounds=ONE_ROUND), "docker")
    folder = tmp_path / plan["run_id"]
    folder.mkdir(parents=True)
    (folder / "results.json").write_text(json.dumps({"schema": "rhvp-image-compare/1", "run": {"status": "complete"},
                                                     "rounds": []}))
    results, backend, the_runner = run(tmp_path, rounds=ONE_ROUND)
    assert results["schema"] == plans.SCHEMA and results["run"]["status"] == "complete" and backend.started
    [kept] = list(folder.glob("results.attempt-*.json"))
    assert json.loads(kept.read_text())["schema"] == "rhvp-image-compare/1"
    assert cli.main(["report", str(kept)]) == 2 and cli.main(["analyze", str(kept)]) == 2


class DiesThenStopFails(FakeBackend):
    """Launch 1: the first start stops at the memory check, the second loads and times out, and the teardown
    between the tries fails."""

    def stop(self, handle):
        super().stop(handle)
        if handle["launch"]["index"] == 1 and self.stopped.count(1) == 2:
            raise runner.RoundError("could not delete the side's pod (HTTP 500)")


def test_the_start_count_is_the_starts_that_happened(tmp_path):
    backend = DiesThenStopFails(hang_once=True, dies={1: 1})
    results, _, the_runner = run(tmp_path, backend, rounds=ONE_ROUND)
    failed = results["launches"][0]
    assert (failed["error_code"], failed["starts"], failed["start_retries"], failed["load_retries"]) == (
        "RUNNER_ERROR", 2, 1, 0)
    assert backend.started.count((1, "A")) == 2
    assert "- launch 1 (A): started 2 times in all, then RUNNER_ERROR" in (the_runner.dir / "results.md").read_text()


def test_a_twins_time_out_counts_against_its_side(tmp_path):
    results, _, the_runner = run(tmp_path, FakeBackend(hangs={2: 1}), rounds=ONE_ROUND)
    twin = results["launches"][1]
    assert (twin["side"], twin["status"], twin["load_timeouts"], twin["load_retries"]) == ("A~same", "ok", 1, 1)
    assert results["analysis"]["load_timeouts"] == {"A": 1, "B": 0}
    # A comparison of B with A counts only the launches of A and B themselves; the twin's is in the totals.
    assert {row["load_timeouts"]["A"] for row in results["analysis"]["comparisons"]} == {0}
    text = (the_runner.dir / "results.md").read_text()
    assert "- launch 2 (A~same): timed out 1 time, ran again, then finished" in text
    assert "Only side A timed out" in text


def test_a_complete_run_is_not_run_again(tmp_path):
    first, _, _ = run(tmp_path, rounds=ONE_ROUND)
    again, backend, _ = run(tmp_path, rounds=ONE_ROUND)
    assert backend.started == [] and again == json.loads(json.dumps(first))


def test_an_unfinished_run_is_kept_beside_the_new_one(tmp_path):
    plan = plans.build(config(rounds=ONE_ROUND), "docker")
    folder = tmp_path / plan["run_id"]
    folder.mkdir()
    (folder / "results.json").write_text(json.dumps({"run": {"status": "running"}}))
    results, _, _ = run(tmp_path, rounds=ONE_ROUND)
    assert results["run"]["status"] == "complete"
    assert len(list(folder.glob("results.attempt-*.json"))) == 1


def test_analysis_is_a_pure_function_of_plan_and_launches(tmp_path):
    results, _, _ = run(tmp_path)
    again = analyze.analyze({"plan": results["plan"], "launches": results["launches"]})
    assert json.loads(json.dumps(again)) == json.loads(json.dumps(results["analysis"]))
    assert again["tool"] == analyze.TOOL
    # The boundary a replacement plugs into: plain rows in, the same comparisons out.
    given = analyze.inputs(results)
    assert set(given["rows"][0]) == {"round", "pos", "side", "case", "value"}
    assert given["twins"] == {"A~same": "A", "B~same": "B"}
    out = analyze.analyse(given["rows"], given["positions"], given["twins"], given["cases"], given["alpha"])
    assert json.loads(json.dumps(out["comparisons"])) == json.loads(json.dumps(results["analysis"]["comparisons"]))


def test_markdown_report(tmp_path):
    results, _, the_runner = run(tmp_path)
    text = report.markdown(results)
    assert (the_runner.dir / "results.md").read_text() == text
    assert "| inter-token latency, median, 1 concurrent |" in text and "SLOWER" in text and "WORSE" in text
    assert "registry.example/server:1" in text and "A~same" in text
    assert "corrected for testing 4 cases at once" in text
    assert not re.search(r"\d\s?[x×]\b", text)


def test_command_line_validate_analyze_report(tmp_path, capsys):
    path = tmp_path / "compare.json"
    path.write_text(json.dumps(config()))
    assert cli.main(["validate", "--config", str(path), "--platform", "docker"]) == 0
    out = capsys.readouterr().out
    assert "warm-up: A B" in out and "round 1: A A~same B B~same" in out and "round 5: B~same B A~same A" in out
    assert "server starts: 34" in out and "twins: A, B" in out
    assert "a verdict on a primary case needs 8 of the 8 rounds to lean the same way" in out
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**config(), "sides": []}))
    assert cli.main(["validate", "--config", str(bad)]) == 2
    results, _, the_runner = run(tmp_path)
    assert cli.main(["analyze", str(the_runner.path), "--write"]) == 0
    assert "SLOWER" in capsys.readouterr().out
    assert cli.main(["report", str(the_runner.path)]) == 0
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert cli.main(["analyze", str(broken)]) == runner.EXIT_FAILED == 3  # never 1, which means a partial run


def test_the_shipped_configs_build():
    """The example in values-global.yaml (as the notebook chart passes it: global.compare without enabled, plus the
    model) and the mock config of compare-local.sh, without a chart renderer."""
    import yaml

    values = yaml.safe_load((ROOT / "values-global.yaml").read_text())["global"]
    example = {k: v for k, v in values["compare"].items() if k != "enabled"}
    example["model"] = {k: values["model"][k] for k in ("id", "source", "revision")}
    plan = plans.build(example, "openshift")
    assert (plan["rounds"], len(plan["launches"]), plan["twins"], plan["warmup"]) == (8, 34, ["A", "B"], True)
    assert plan["power"]["reachable"] and plan["balance"]["balanced"] and plan["energy"]["source"] == "dcgm"
    assert plan["analysis"]["primary"] == ["startup/first_answer_s", "c32/ttft_ms_median",
                                           "c32/output_tokens_per_s", "c32/tokens_per_joule[gpu]"]
    mock = plans.build(json.loads((FIXTURES / "compare-local-mock.json").read_text()), "docker")
    assert len(mock["launches"]) == 8 and mock["energy"]["source"] == "none"


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


def test_requests_that_fail_on_one_side_give_no_verdict(tmp_path):
    """Side B loses 40 requests at 4 streams in rounds 1 and 2. The 6 rounds left would make B look better than it
    was, so the 4-stream cases get NO VERDICT (side_failed), the run is partial (REQUESTS_FAILED) and the levels are
    listed everywhere a reader looks."""
    backend = PublishingBackend(errored={5: 4, 8: 4})
    results, _, the_runner = run(tmp_path, backend)
    jsonschema.validate(results, SCHEMA)
    assert [(x["index"], x["side"], x["round"]) for x in results["launches"] if x["index"] in (5, 8)] == [
        (5, "B", 1), (8, "B", 2)]
    rows = rows_of(results)
    for case in ("c4/ttft_ms_median", "c4/itl_ms_median", "c4/output_tokens_per_s", "c4/tokens_per_joule[gpu]"):
        assert rows[case]["n_rounds"] == 6 and rows[case]["lost"] == {"B": 2, "A": 0}
        assert (rows[case]["verdict"], rows[case]["reason"]) == ("NO VERDICT", "side_failed")
    assert rows["c1/itl_ms_median"]["verdict"] == "SLOWER" and rows["c1/itl_ms_median"]["lost"]["B"] == 0
    assert results["run"]["status"] == "partial" and results["run"]["error_code"] == "REQUESTS_FAILED"
    status = backend.published[-1][1]
    assert [(u["index"], u["round"], u["side"], u["concurrency"], u["errored"]) for u in status["levels_unusable"]] == [
        (5, 1, "B", 4, 40), (8, 2, "B", 4, 40)]
    text = (the_runner.dir / "results.md").read_text()
    assert "- launch 5 (B), 4 streams: 40 requests failed" in text
    assert "side A 0, side B 2" in text


def test_a_zero_from_one_side_gives_no_verdict(tmp_path):
    """Side B's requests at 4 streams finish but give no output token in rounds 1 and 2: 0 tokens per second and 0
    tokens per joule there. A 0 gives no ratio, and leaving those rounds out would decide the verdict on the rest, so
    both cases get NO VERDICT (nonpositive_value)."""
    results, _, the_runner = run(tmp_path, FakeBackend(zero={5: 4, 8: 4}))
    jsonschema.validate(results, SCHEMA)
    b5 = next(x for x in results["launches"] if x["index"] == 5)
    assert b5["side"] == "B" and b5["values"]["c4/output_tokens_per_s"] == 0
    assert b5["values"]["c4/tokens_per_joule[gpu]"] == 0
    rows = rows_of(results)
    for case in ("c4/output_tokens_per_s", "c4/tokens_per_joule[gpu]"):
        assert rows[case]["n_rounds"] == 8 and rows[case]["lost"] == {"B": 0, "A": 0}
        assert (rows[case]["verdict"], rows[case]["reason"]) == ("NO VERDICT", "nonpositive_value")
    assert rows["c4/itl_ms_median"]["verdict"] == "SLOWER" and rows["c4/itl_ms_median"]["n_rounds"] == 8
    assert "NO VERDICT (nonpositive_value)" in (the_runner.dir / "results.md").read_text()


def test_a_start_that_fails_vllms_memory_check_is_tried_once_more(tmp_path):
    backend = FakeBackend(dies={3: 1})
    results, _, the_runner = run(tmp_path, backend)
    jsonschema.validate(results, SCHEMA)
    l3 = next(x for x in results["launches"] if x["index"] == 3)
    assert (l3["side"], l3["kind"], l3["status"], l3["start_retries"]) == ("A", "scored", "ok", 1)
    assert backend.started.count((3, "A")) == 2 and backend.stopped.count(3) == 2  # the first try, then the launch
    first_try = the_runner.dir / l3["artifacts"]["server_log_attempt_1"]
    assert "Error in memory profiling" in first_try.read_text()
    assert "Error in memory profiling" not in (the_runner.dir / l3["artifacts"]["server_log"]).read_text()
    assert any("started once more" in note for note in l3["notes"])
    assert all(x["start_retries"] == 0 for x in results["launches"] if x["index"] != 3)
    assert results["run"]["status"] == "complete" and runner.exit_code(results) == runner.EXIT_COMPLETE
    assert "- launch 3 (A): started 2 times in all, then finished" in (the_runner.dir / "results.md").read_text()
    head = "".join(show._head(results, [], []))
    assert "started again after vLLM" in head and "3 (A, 2)" in head


def test_a_second_memory_check_failure_fails_the_launch(tmp_path):
    backend = FakeBackend(dies={3: 2})
    results, _, the_runner = run(tmp_path, backend)
    jsonschema.validate(results, SCHEMA)
    l3 = next(x for x in results["launches"] if x["index"] == 3)
    assert (l3["status"], l3["error_code"], l3["start_retries"]) == ("failed", "SIDE_FAILED", 1)
    assert l3["error"] == "the server exited with code 1"
    assert backend.started.count((3, "A")) == 2
    assert "started 2 times in all, then SIDE_FAILED" in (the_runner.dir / "results.md").read_text()


def test_a_server_that_exits_for_another_reason_is_not_started_again(tmp_path):
    backend = FakeBackend(dies={3: 1}, dies_log="2026-01-01T00:00:01Z torch.OutOfMemoryError: CUDA out of memory\n")
    results, _, _ = run(tmp_path, backend)
    l3 = next(x for x in results["launches"] if x["index"] == 3)
    assert (l3["status"], l3["error_code"], l3["start_retries"]) == ("failed", "SIDE_FAILED", 0)
    assert backend.started.count((3, "A")) == 1 and "server_log_attempt_1" not in l3["artifacts"]


def test_launches_on_two_nodes_give_no_verdict(tmp_path):
    results, _, _ = run(tmp_path, FakeBackend(nodes=("node-1", "node-2")))
    jsonschema.validate(results, SCHEMA)
    rows = results["analysis"]["comparisons"]
    assert {(row["outcome"], row["reason"]) for row in rows} == {("no_verdict", "more_than_one_node")}
    assert [r["code"] for r in results["analysis"]["refusals"]] == ["MORE_THAN_ONE_NODE"]
    assert [n["name"] for n in results["provenance"]["nodes"]] == ["node-2", "node-1"]


@pytest.mark.parametrize("twins, launches, reason", [("A", 24, "one_sided_floor"), ("none", 16, "no_floor")])
def test_without_both_twins_no_case_gets_a_verdict(tmp_path, twins, launches, reason):
    results, _, _ = run(tmp_path, rounds={"count": 8, "warmup": False, "twins": twins})
    jsonschema.validate(results, SCHEMA)
    assert len(results["launches"]) == launches
    assert {(row["verdict"], row["reason"]) for row in results["analysis"]["comparisons"]} == {("NO VERDICT", reason)}


def test_rounds_too_few_for_the_primary_cases_still_grade_the_secondary_ones(tmp_path):
    # 6 rounds: one case alone can reach alpha (2 * 2^-6 < 0.05), four corrected together cannot (they need 8).
    results, _, _ = run(tmp_path, rounds={"count": 6, "warmup": False})
    rows = rows_of(results)
    assert (rows["c4/output_tokens_per_s"]["verdict"], rows["c4/output_tokens_per_s"]["reason"]) == (
        "NO VERDICT", "too_few_rounds")
    assert rows["c1/itl_ms_median"]["verdict"] == "SLOWER" and rows["c1/itl_ms_median"]["p_raw"] == 2 ** -5
    [note] = [line for line in runner.plan_lines(results["plan"]) if "need at least 8 rounds" in line]
    assert "every primary case gets NO VERDICT" in note and "secondary cases are graded" in note
    assert "at most 4 primary cases can get one" in runner.plan_lines(
        plans.build(config(rounds={"count": 16, "warmup": True, "twins": "both"}), "docker"))[1]


def test_a_side_that_differs_from_its_twin_is_shown_with_no_cause(tmp_path):
    results, _, _ = run(tmp_path, rounds=ONE_ROUND)
    twins = results["analysis"]["same_vs_same"]
    twins[0]["p_raw"] = 2 ** -7
    head = "".join(show._head(results, [], []))
    shown = words.TWIN_DIFFERS.format(cases=words.TWIN_DIFFERS_CASE.format(
        side=twins[0]["side"], case=twins[0]["case"], p="0.00781"),
        tested=sum(1 for t in twins if t["p_raw"] is not None))  # one round: only the case set here has a p-value
    assert shown in head and "bias" not in head


def test_the_live_server_comes_back_and_the_run_says_so(tmp_path, monkeypatch):
    backend = ServerBackend()
    code, root, message = _main(tmp_path, backend, monkeypatch, rounds=ONE_ROUND)
    assert code == 0 and backend.server_up
    assert backend.events[-2:] == [("restore",), ("cleanup",)]  # back first, then the cleanup
    [path] = root.glob("*/results.json")
    assert json.loads(path.read_text())["run"]["server_restored"] is True
    assert json.loads(path.read_text())["run"]["server_found_down"] is None
    assert json.loads(message.read_text())["server_restored"] is True


def test_a_server_that_stopped_by_itself_before_the_run_took_it_is_recorded(tmp_path, monkeypatch, capsys):
    code, root, _ = _main(tmp_path, FoundDownBackend(), monkeypatch, rounds=ONE_ROUND)
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
    code, root, message = _main(tmp_path, backend, monkeypatch, rounds=ONE_ROUND)
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
    backend = ServerBackend(term_in_launch=3)
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
    code, root, message = _main(tmp_path, backend, monkeypatch, rounds=ONE_ROUND)
    assert code == runner.EXIT_RUN_ACTIVE == 5
    assert not backend.prepared and backend.events == [] and backend.server_up  # nothing touched
    assert not list(root.glob("*/results.json")) if root.exists() else True
    short = json.loads(message.read_text())
    assert short["error_code"] == "RUN_ACTIVE" and short["status"] == "failed"


def test_a_run_that_fails_outside_a_launch_is_exit_3(tmp_path, monkeypatch):
    backend = ServerBackend()
    backend.describe = lambda: (_ for _ in ()).throw(KeyError("boom"))  # recorded as a note, not a failure
    backend.gpu_free = lambda plan, side: None
    original = backend.prepare

    def prepare(plan):
        original(plan)
        raise runner.RoundError("could not list pods in vllm-inference: HTTP 403")
    backend.prepare = prepare
    code, root, message = _main(tmp_path, backend, monkeypatch, rounds=ONE_ROUND)
    assert code == runner.EXIT_FAILED == 3
    [path] = root.glob("*/results.json")
    results = json.loads(path.read_text())
    assert results["run"]["status"] == "failed" and results["run"]["error_code"] == "RUNNER_FAILED"
    assert results["run"]["server_restored"] is True and backend.server_up  # brought back all the same


def test_a_server_that_comes_back_on_the_last_try_is_back(tmp_path):
    backend = ServerBackend(term_in_launch=3)
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
