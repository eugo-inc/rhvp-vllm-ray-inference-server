# SPDX-License-Identifier: Apache-2.0
"""Command line.

    python -m rhvp_compare validate --config compare.json [--platform openshift|docker]
    python -m rhvp_compare run --backend kube|docker --config compare.json --results /results
    python -m rhvp_compare analyze <results.json> [--write]
    python -m rhvp_compare report <results.json>

validate prints the plan: every round's order, the cases, and whether the planned rounds can give a verdict.
analyze recomputes the verdicts of a results file (after a change to the analysis, or with another tool behind the
same boundary) and prints them; --write puts them back into the file and rewrites results.md next to it.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from . import analyze, plan as plans, report, runner


def _validate(args) -> int:
    the_plan = plans.load(args.config, args.platform)
    print(f"run id: {the_plan['run_id']}")
    print(f"sides: A = {the_plan['sides'][0]['name']} ({the_plan['sides'][0]['image']})")
    print(f"       B = {the_plan['sides'][1]['name']} ({the_plan['sides'][1]['image']})")
    print(f"twins: {', '.join(the_plan['twins']) or 'none'}")
    warmups = [launch for launch in the_plan["launches"] if launch["kind"] == plans.WARMUP]
    if warmups:
        print(f"warm-up: {' '.join(launch['side'] for launch in warmups)}")
    for r in range(1, the_plan["rounds"] + 1):
        order = [launch["side"] for launch in the_plan["launches"] if launch["round"] == r]
        print(f"round {r}: {' '.join(order)}")
    print(f"server starts: {len(the_plan['launches'])}")
    print(f"energy: {the_plan['energy']['source']}")
    print(f"cases: {len(the_plan['cases'])}, primary {the_plan['power']['family_size']}; rule "
          f"{the_plan['analysis']['rule']}, alpha {the_plan['analysis']['alpha']}")
    for line in runner.plan_lines(the_plan)[1:]:
        print(line)
    return 0


def _backend(name: str, results: Path):
    if name == "docker":
        from .dockerapi import DockerBackend

        return DockerBackend(results_root=results)
    from .kube import KubeBackend

    return KubeBackend(results_root=results)


def _run(args) -> int:
    results = Path(args.results)
    return runner.main_run(args.config, str(results), _backend(args.backend, results))


def _analyze(args) -> int:
    path = Path(args.results_file)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["analysis"] = analyze.analyze(data)
    if args.write:
        runner.write_json(path, data)
        (path.parent / "results.md").write_text(report.markdown(data), encoding="utf-8")
    print(report.markdown(data))
    return 0


def _report(args) -> int:
    print(report.markdown(json.loads(Path(args.results_file).read_text(encoding="utf-8"))))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rhvp_compare", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("validate", help="check a config and print its plan")
    p.add_argument("--config", required=True)
    p.add_argument("--platform", choices=("openshift", "docker"), default="openshift")
    p.set_defaults(func=_validate)
    p = sub.add_parser("run", help="run a comparison")
    p.add_argument("--backend", choices=("kube", "docker"), required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--results", required=True, help="the folder results go into (one subfolder per run)")
    p.set_defaults(func=_run)
    p = sub.add_parser("analyze", help="recompute the verdicts of a results file")
    p.add_argument("results_file")
    p.add_argument("--write", action="store_true")
    p.set_defaults(func=_analyze)
    p = sub.add_parser("report", help="print the markdown table of a results file")
    p.add_argument("results_file")
    p.set_defaults(func=_report)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except plans.PlanError as error:
        print(f"config error: {error}", file=sys.stderr)
        return 2
    except runner.RoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return runner.EXIT_FAILED
    except Exception:  # anything else is a failure too (3), never the 1 that means a partial run
        traceback.print_exc()
        return runner.EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
