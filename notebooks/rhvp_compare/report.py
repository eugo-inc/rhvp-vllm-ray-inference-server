# SPDX-License-Identifier: Apache-2.0
"""The short markdown table written next to each results file (results.md). Standard library only."""

from __future__ import annotations

from collections.abc import Mapping

from . import summary


def _num(value, unit: str = "") -> str:
    if value is None:
        return "-"
    if unit in ("s",):
        text = f"{value:.2f}"
    elif unit == "ms":
        text = f"{value:.1f}"
    elif unit == "tokens/J":
        text = f"{value:.3g}"
    else:
        text = f"{value:.1f}" if abs(value) < 1000 else f"{value:.0f}"
    return f"{text} {unit}".strip()


def _ratio(row: Mapping) -> str:
    if row.get("median_ratio") is None:
        return "-"
    ci = row.get("ci") or {}
    if ci.get("low") is None:
        return f"{row['median_ratio']:.3f}"
    return f"{row['median_ratio']:.3f} [{ci['low']:.3f}, {ci['high']:.3f}]"


def _p(value) -> str:
    return "-" if value is None else (f"{value:.3g}" if value >= 0.001 else f"{value:.1e}")


def markdown(results: Mapping) -> str:
    """The run's summary: what was compared, on what, how, and one row per metric."""
    plan = results["plan"]
    run = results.get("run") or {}
    prov = results.get("provenance") or {}
    analysis = results.get("analysis") or {}
    method = plan["analysis"]
    model = plan["model"]
    lines = [f"# Image comparison: {plan['run_id']}", ""]
    if run.get("synthetic"):
        lines += ["**Synthetic: made up by a script, never measured. Not a result.**", ""]
    code = f" ({run['error_code']})" if run.get("error_code") else ""
    lines.append(f"Status: {run.get('status', 'unknown')}{code}. Started {run.get('started_at') or '-'}, finished "
                 f"{run.get('finished_at') or '-'}. Platform: {plan.get('platform', '-')}.")
    revision = f" at revision {model['revision']}" if model.get("revision") else ""
    lines += ["", f"Model: `{model['source']}`{revision}, served as `{model['id']}`.", ""]
    lines += ["| Side | Name | Image | Digest | Server version |", "|---|---|---|---|---|"]
    sides_prov = prov.get("sides") or {}
    for side in plan["sides"]:
        p = sides_prov.get(side["key"]) or {}
        version = p.get("server_version") or {}
        shown = " ".join(f"{k} {v}" for k, v in version.items() if k != "source" and v) or "-"
        lines.append(f"| {side['key']} | {side['name']} | `{side['image']}` | `{p.get('image_digest') or '-'}` | "
                     f"{shown} |")
    gpu = prov.get("gpu") or {}
    node = prov.get("node") or {}
    lines += ["", f"GPU: {gpu.get('product') or '-'} (count {gpu.get('count') or '-'}, driver "
              f"{gpu.get('driver') or '-'}). Node type: {node.get('instance_type') or '-'}, "
              f"{node.get('arch') or '-'}."]
    load = plan["load"]
    warmup = ", after one warm-up round per side" if plan.get("warmup") else ""
    aa = (f", then {plan['aa_pairs']} A/A pair{'s' if plan['aa_pairs'] != 1 else ''}" if plan.get("aa_pairs")
          else "")
    lines.append(f"Plan: {plan['pairs']} pairs in ABBA order{warmup}{aa}. "
                 f"Load: GuideLLM, {load['prompt_tokens']} prompt and {load['output_tokens']} output tokens per "
                 f"request, {load['seconds_per_level']:.0f} s at each of "
                 f"{', '.join(str(c) for c in load['concurrencies'])} concurrent streams, seed {load['seed']}.")
    primary_count = sum(1 for m in plan["metrics"] if m.get("role", "primary") == "primary")
    lines.append(f"Method: B/A per pair; exact {method['test']} test, {method['correction']} correction over the "
                 f"{primary_count} primary metrics, alpha {method['alpha']}; tie band ±{method['tie_band']:.0%} "
                 f"(declared before the run). Energy: {plan['energy']['label']}, source "
                 f"{plan['energy']['source']}.")
    rows = analysis.get("metrics") or []
    for role, title, p_head in (("primary", "Primary metrics (corrected for multiple comparisons)", "p adj."),
                                ("secondary", "Secondary metrics (descriptive: p not corrected, so a verdict here "
                                              "is not a finding)", "p")):
        chosen = [row for row in rows if row.get("role", "primary") == role]
        if not chosen:
            continue
        lines += ["", f"{title}:", "",
                  f"| Metric | A median | B median | B/A median [CI] | pairs | {p_head} | Verdict |",
                  "|---|---|---|---|---|---|---|"]
        for row in chosen:
            p_value = row.get("p_adjusted") if role == "primary" else row.get("p_value")
            lines.append(f"| {row['label']} | {_num(row['a_median'], row['unit'])} | "
                         f"{_num(row['b_median'], row['unit'])} | {_ratio(row)} | {row['n_pairs']} | "
                         f"{_p(p_value)} | {row['verdict']} |")
    floor = plan.get("floor") or {}
    if floor and not floor.get("reachable", True):
        lines += ["", f"Note: with {plan['pairs']} pairs and {floor['family_size']} primary metrics the smallest "
                  f"possible adjusted p-value is {floor['min_adjusted_p']:.3g}, above alpha {method['alpha']}, so no "
                  f"primary metric can be FASTER or SLOWER in this run; {floor['pairs_needed']} pairs are needed."]
    noise = analysis.get("noise_floor") or []
    if noise:
        lines += ["", "A/A noise floor (side A against itself): typical and largest change between two rounds.", "",
                  "| Metric | A/A pairs | typical | largest |", "|---|---|---|---|"]
        labels = {m["id"]: m["label"] for m in plan["metrics"]}
        for row in noise:
            typical = "-" if row["median_abs_change"] is None else f"{row['median_abs_change']:.1%}"
            largest = "-" if row["max_abs_change"] is None else f"{row['max_abs_change']:.1%}"
            lines.append(f"| {labels.get(row['id'], row['id'])} | {row['pairs']} | {typical} | {largest} |")
    failed = [r for r in results.get("rounds") or [] if r.get("status") != "ok"]
    if failed:
        lines += ["", "Rounds that did not finish:"]
        lines += [f"- round {r['index']} (side {r['side']}, {r['kind']}): {r.get('error_code') or 'failed'}: "
                  f"{r.get('error') or r.get('status')}" for r in failed]
    retried = [r for r in results.get("rounds") or [] if r.get("start_retries")]
    if retried:
        lines += ["", "Rounds whose side was started again after vLLM's start-up memory check failed "
                  "(timed from the last start):"]
        lines += [f"- round {r['index']} (side {r['side']}, {r['kind']}): started {r['start_retries'] + 1} times, "
                  f"then {'finished' if r.get('status') == 'ok' else r.get('error_code') or 'failed'}" for r in retried]
    unusable = summary.levels_unusable(results)
    if unusable:
        lines += ["", "Load levels left out of the comparison:"]
        lines += [f"- round {u['index']} (side {u['side']}, {u['kind']}), {u['concurrency']} streams: "
                  f"{u['errored']} requests failed, {u['successful']} succeeded" for u in unusable]
    lost = [row for row in rows if (row.get("pairs_lost") or {}).get("A") or (row.get("pairs_lost") or {}).get("B")]
    if lost:
        lines += ["", "Metrics that lost pairs because one side failed (so INCONCLUSIVE):"]
        lines += [f"- {row['label']}: side A {row['pairs_lost']['A']}, side B {row['pairs_lost']['B']}" for row in lost]
    if run.get("server_restored") is False:
        lines += ["", "**The live server was not brought back.** " + "; ".join(
            e for e in run.get("errors") or [] if "live server" in e)]
    return "\n".join(lines) + "\n"
