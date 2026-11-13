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
    return "-" if row.get("ratio") is None else f"{row['ratio']:.3f}"


def _bar(row: Mapping) -> str:
    return "-" if row.get("bar") is None else f"±{row['bar']:.1%}"


def _p(value) -> str:
    return "-" if value is None else (f"{value:.3g}" if value >= 0.001 else f"{value:.1e}")


def _verdict(row: Mapping) -> str:
    return f"{row['verdict']} ({row['reason']})" if row.get("reason") else row["verdict"]


def markdown(results: Mapping) -> str:
    """The run's summary: what was compared, on what, how, and one row per case."""
    plan = results["plan"]
    run = results.get("run") or {}
    prov = results.get("provenance") or {}
    analysis = results.get("analysis") or {}
    model = plan["model"]
    power = plan["power"]
    rule = analysis.get("rule") or {}
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
    warmup = ", after one warm-up launch per side" if plan.get("warmup") else ""
    lineup = ", ".join(e["key"] for e in plan["lineup"])
    lines.append(f"Plan: {plan['rounds']} rounds of {lineup} in rotating order ({plan['schedule']}){warmup}: "
                 f"{len(plan['launches'])} launches. Load: GuideLLM, {load['prompt_tokens']} prompt and "
                 f"{load['output_tokens']} output tokens per request, {load['seconds_per_level']:.0f} s at each of "
                 f"{', '.join(str(c) for c in load['concurrencies'])} concurrent streams, seed {load['seed']}.")
    lines.append(f"Method: B/A per round; exact two-sided sign test, Holm's correction over the "
                 f"{power['family_size']} primary cases ({rule.get('plain') or power['plain']}), alpha "
                 f"{plan['analysis']['alpha']}; the bar is the noise each side's twin measured (rule "
                 f"{plan['analysis']['rule']}). Energy: {plan['energy']['label']}, source {plan['energy']['source']}, "
                 f"at least {plan['energy']['min_samples']} power samples per level.")
    labels = {c["id"]: c for c in plan["cases"]}
    rows = analysis.get("comparisons") or []
    for role, title, p_key in (("primary", "Primary cases", "p_corrected"),
                               ("secondary", "Secondary cases (descriptive: p not corrected, so a verdict here is "
                                             "not a finding)", "p_raw")):
        chosen = [row for row in rows if row.get("role", "primary") == role]
        if not chosen:
            continue
        lines += ["", f"{title}:", "",
                  "| Case | A median | B median | B/A median | Bar | Rounds (untied) | p | Verdict |",
                  "|---|---|---|---|---|---|---|---|"]
        for row in chosen:
            unit = labels.get(row["case"], {}).get("unit", "")
            lines.append(f"| {labels.get(row['case'], {}).get('label', row['case'])} | "
                         f"{_num(row['baseline_median'], unit)} | {_num(row['subject_median'], unit)} | "
                         f"{_ratio(row)} | {_bar(row)} | {row['n_rounds']} ({row['n_untied']}) | "
                         f"{_p(row.get(p_key))} | {_verdict(row)} |")
    timeouts = analysis.get("load_timeouts") or {}
    if timeouts.get("A") or timeouts.get("B"):
        lines += ["", f"**GuideLLM timed out** {timeouts.get('A', 0)} times in the launches of side A and "
                      f"{timeouts.get('B', 0)} times in those of side B, twins included. Each such launch ran again "
                      f"from a new start, or failed."]
        if not (timeouts.get("A") and timeouts.get("B")):
            lines[-1] += (f" Only side {'A' if timeouts.get('A') else 'B'} timed out: a load that stalls on one side "
                          f"only is a difference the verdicts above do not show.")
    if not power.get("both_twins"):
        lines += ["", "Note: without a twin for both sides no case can get a verdict."]
    elif not power.get("reachable"):
        lines += ["", f"Note: {power['family_size']} primary cases need at least {power['min_rounds']} rounds for any "
                  f"verdict; this run planned {plan['rounds']}."]
    twins = analysis.get("same_vs_same") or []
    if twins:
        lines += ["", "Each side against its twin (the noise floor):", "",
                  "| Side | Case | side / twin, median | Floor | Rounds | p, not corrected |",
                  "|---|---|---|---|---|---|"]
        for t in twins:
            lines.append(f"| {t['side']} | {labels.get(t['case'], {}).get('label', t['case'])} | {t['ratio']:.3f} | "
                         f"{t['floor']:.1%} | {t['n']} | {_p(t.get('p_raw'))} |")
    for note in analysis.get("notes") or []:
        lines += ["", f"**Note:** {note}."]
    launches = results.get("launches") or []
    failed = [r for r in launches if r.get("status") != "ok"]
    if failed:
        lines += ["", "Launches that did not finish:"]
        lines += [f"- launch {r['index']} (round {r.get('round') or 'warm-up'}, {r['side']}): "
                  f"{r.get('error_code') or 'failed'}: {r.get('error') or r.get('status')}" for r in failed]
    retried = [r for r in launches if r.get("start_retries")]
    if retried:
        lines += ["", "Launches whose side was started again after vLLM's start-up memory check failed "
                  "(timed from the last start):"]
        lines += [f"- launch {r['index']} ({r['side']}): started {r.get('starts')} times in all, "
                  f"then {'finished' if r.get('status') == 'ok' else r.get('error_code') or 'failed'}" for r in retried]
    timed_out = [r for r in launches if r.get("load_timeouts")]
    if timed_out:
        lines += ["", "Launches whose GuideLLM timed out (after a time-out the side was taken down and the launch ran "
                  "again from a new start, timed from that start, with the same requests; every figure of such a "
                  "launch comes from that second try, and the first try's files are its *.try-1.* files):"]
        lines += [f"- launch {r['index']} ({r['side']}): timed out {r['load_timeouts']} "
                  f"time{'s' if r['load_timeouts'] > 1 else ''}, "
                  + ("ran again, then " if r.get("load_retries") else "did not run again, then ")
                  + ("finished" if r.get("status") == "ok" else (r.get("error_code") or "failed"))
                  for r in timed_out]
    unusable = summary.levels_unusable(results)
    if unusable:
        lines += ["", "Load levels left out of the comparison:"]
        lines += [f"- launch {u['index']} ({u['side']}), {u['concurrency']} streams: "
                  f"{u['errored']} requests failed, {u['successful']} succeeded" for u in unusable]
    lost = [row for row in rows if any((row.get("lost") or {}).values())]
    if lost:
        lines += ["", "Cases that lost rounds because one side failed on its own account (so NO VERDICT, "
                  "side_failed):"]
        lines += [f"- {row['case']}: side A {row['lost']['A']}, side B {row['lost']['B']}" for row in lost]
    if run.get("server_restored") is False:
        lines += ["", "**The live server was not brought back.** " + "; ".join(
            e for e in run.get("errors") or [] if "live server" in e)]
    if run.get("server_found_down"):
        lines += ["", f"**The live server had stopped by itself** during the image pulls, before the run would have "
                      f"stopped it (its state then: {run['server_found_down']}). The run left it as it was and did not "
                      f"start it again."]
    return "\n".join(lines) + "\n"
