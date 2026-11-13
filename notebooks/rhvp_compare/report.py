# SPDX-License-Identifier: Apache-2.0
"""The short markdown table written next to each results file (results.md). Standard library only."""

from __future__ import annotations

from collections.abc import Mapping

from . import stats, summary, words


def _num(value, unit: str = "") -> str:
    if value is None:
        return "-"
    if unit == "share":
        return f"{value:.1%}"
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


def _starts(round_: Mapping) -> str:
    """How often a retried round started its servers: per replica (and try) unless only the first replica of the
    round's kept try was started again; from the round's count for a file without launch records."""
    again = [item for item in round_.get("launches") or [] if item.get("start_retries")]
    if not again:  # a file without launch records: a round whose load ran again also started its side once more
        return f"started {round_['start_retries'] + (round_.get('load_retries') or 0) + 1} times"
    if len(again) == 1 and again[0].get("replica") == 1 and not again[0].get("discarded"):
        return f"started {again[0]['start_retries'] + 1} times"
    return ", ".join(f"replica {item.get('replica')}{' of the discarded try' if item.get('discarded') else ''} "
                     f"started {item['start_retries'] + 1} times" for item in again)


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
    lines += ["| Side | Name | Image | Digest | Server version | Compile cache |", "|---|---|---|---|---|---|"]
    sides_prov = prov.get("sides") or {}
    caches = [side.get("compile_cache") or "cold" for side in plan["sides"]]
    for side, cache in zip(plan["sides"], caches):
        p = sides_prov.get(side["key"]) or {}
        version = p.get("server_version") or {}
        shown = " ".join(f"{k} {v}" for k, v in version.items() if k != "source" and v) or "-"
        lines.append(f"| {side['key']} | {side['name']} | `{side['image']}` | `{p.get('image_digest') or '-'}` | "
                     f"{shown} | {cache} |")
    gpu = prov.get("gpu") or {}
    node = prov.get("node") or {}
    lines += ["", f"GPU: {gpu.get('product') or '-'} (count {gpu.get('count') or '-'}, driver "
              f"{gpu.get('driver') or '-'}). Node type: {node.get('instance_type') or '-'}, "
              f"{node.get('arch') or '-'}."]
    load = plan["load"]
    warmup = ", after one warm-up round per side" if plan.get("warmup") else ""
    aa = (f", then {plan['aa_pairs']} A/A pair{'s' if plan['aa_pairs'] != 1 else ''}" if plan.get("aa_pairs")
          else "")
    scale = plan.get("scale_out") if plan.get("mode") == "scaleout" else None
    if scale:
        shape = (f"{scale['request_rate']:g} requests per second for {load['seconds_per_level']:.0f} s through a "
                 f"least-busy router, a second replica asked for {scale['step_after_s']:g} s after the first "
                 f"request")
    else:
        shape = (f"{load['seconds_per_level']:.0f} s at each of {', '.join(str(c) for c in load['concurrencies'])} "
                 f"concurrent streams")
    lines.append(f"Plan: {plan['pairs']} pairs in ABBA order{warmup}{aa}. "
                 f"Load: GuideLLM, {load['prompt_tokens']} prompt and {load['output_tokens']} output tokens per "
                 f"request, {shape}, seed {load['seed']}.")
    if scale:
        goals = "; ".join(f"{name} at most {scale['slo'][key]:g} ms" for key, name in
                          (("ttft_ms", "time to first token"), ("tpot_ms", "inter-token latency"))
                          if scale["slo"].get(key) is not None)
        lines.append(f"Latency goals: {goals}. Mode scaleout: {scale['surge_rule']}.")
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
        if role == "primary" and caches[0] != caches[1]:
            lines += ["", f"**The two sides started with different compile caches: side A {caches[0]}, side B "
                          f"{caches[1]}.** The start-up figures and their verdicts compare the two cache settings "
                          "together with the two images."]
        elif role == "primary" and caches == ["warm", "warm"]:
            lines += ["", "Both sides started their scored rounds with warm compile caches, kept from their warm-up "
                          "rounds."]
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
    if scale:
        lines += _scale_lines(results)
        fit = summary.surge_rate_fit(results)
        if fit:
            lines += ["", "**" + words.SCALE_RATE_NOTES[fit] + "**"]
    failed = [r for r in results.get("rounds") or [] if r.get("status") != "ok"]
    if failed:
        lines += ["", "Rounds that did not finish:"]
        lines += [f"- round {r['index']} (side {r['side']}, {r['kind']}): {r.get('error_code') or 'failed'}: "
                  f"{r.get('error') or r.get('status')}" for r in failed]
    retried = [r for r in results.get("rounds") or [] if r.get("start_retries")]
    if retried:
        lines += ["", "Rounds whose side was started again after vLLM's start-up memory check failed (a round's "
                  "first replica timed from its last start, a replica a scale-out round adds from when it was first "
                  "asked for):"]
        lines += [f"- round {r['index']} (side {r['side']}, {r['kind']}): {_starts(r)}, "
                  f"then {'finished' if r.get('status') == 'ok' else r.get('error_code') or 'failed'}" for r in retried]
    reloaded = [r for r in results.get("rounds") or [] if r.get("load_retries")]
    if reloaded:
        how = ("the whole round ran once more from a new first replica, with the same requests, and every figure of "
               "the round comes from that second try; the first try's files are the round's *.try-1.* files"
               if plan.get("mode") == "scaleout" else
               "the side was taken down and started again, timed from that new start, and GuideLLM ran once more "
               "with the same requests; every figure of the round comes from that second try, and the first try's "
               "files are the round's *.try-1.* files")
        lines += ["", f"Rounds whose load ran again after GuideLLM timed out ({how}):"]
        lines += [f"- round {r['index']} (side {r['side']}, {r['kind']}): ran again, then "
                  f"{'finished' if r.get('status') == 'ok' else r.get('error_code') or 'failed'}" for r in reloaded]
    troubled = summary.levels_unusable(results)
    unusable = [u for u in troubled if u["concurrency"] is not None]
    if unusable:
        lines += ["", "Load levels left out of the comparison:"]
        lines += [f"- round {u['index']} (side {u['side']}, {u['kind']}), {u['concurrency']} streams: "
                  f"{u['errored']} requests failed, {u['successful']} succeeded" for u in unusable]
    surges = [u for u in troubled if u["concurrency"] is None]
    if surges:
        lines += ["", "Surges in which requests failed or none finished. Only their latency medians are left out of "
                      "the comparison: a failed request counts as one that missed the goals in the share within the "
                      "goals and in goodput, and a surge in which no request finished counts as 0 for both:"]
        lines += [f"- round {u['index']} (side {u['side']}, {u['kind']}): {u['errored']} requests failed, "
                  f"{u['successful']} succeeded" for u in surges]
    lost = [row for row in rows if (row.get("pairs_lost") or {}).get("A") or (row.get("pairs_lost") or {}).get("B")]
    if lost:
        lines += ["", "Metrics that lost pairs because one side failed (so INCONCLUSIVE):"]
        lines += [f"- {row['label']}: side A {row['pairs_lost']['A']}, side B {row['pairs_lost']['B']}" for row in lost]
    if run.get("server_restored") is False:
        lines += ["", "**The live server was not brought back.** " + "; ".join(
            e for e in run.get("errors") or [] if "live server" in e)]
    if run.get("server_found_down"):
        lines += ["", f"**The live server had stopped by itself** before the run came to take it away (its state "
                      f"then: {run['server_found_down']}). The run left it as it was and did not start it again."]
    return "\n".join(lines) + "\n"


def _scale_lines(results: Mapping) -> list[str]:
    """Mode scaleout: per side, the medians over the scored rounds of when the second replica was asked for and
    first sent requests, and its share of the surge's requests through the router."""
    out = ["", "The second replica (medians over the scored rounds that reached the forced step; seconds after "
               "the surge's first request):", "",
           "| Side | Rounds | asked for | first sent requests | its share of the requests | answered after the "
           "surge |", "|---|---|---|---|---|---|"]
    for side in ("A", "B"):
        scales = [r["scale"] for r in results.get("rounds") or [] if r.get("kind") == "scored"
                  and r.get("side") == side and r.get("status") == "ok" and (r.get("scale") or {}).get("new_replica")]

        def median(values: list) -> float | None:
            values = [v for v in values if isinstance(v, (int, float))]
            return stats.median(values) if values else None

        late = sum(1 for s in scales if s["new_replica"].get("joined_during_surge") is False)
        joined = [s["new_replica"].get("joined_at_s") for s in scales if s["new_replica"].get("joined_during_surge")]
        out.append(f"| {side} | {len(scales)} | {_num(median([s.get('step_at_s') for s in scales]), 's')} | "
                   f"{_num(median(joined), 's')} | "
                   f"{_num(median([s.get('new_replica_share') for s in scales]), 'share')} | {late} |")
    scale = results["plan"].get("scale_out") or {}
    if any(side.get("gpu", True) for side in results["plan"]["sides"]):
        declared = ("both replicas of a side share one GPU" if scale.get("shared_gpu")
                    else "each replica has its own GPU")
        given = summary.replica_gpus(results)
        shown = "; ".join(f"replica {n}: " + " or ".join(
            f"`{g['gpus']}` ({', '.join(f'`{u}`' for u in g['uuids']) or 'UUID not read'})" for g in entries)
            for n, entries in given.items()) or "not recorded"
        out += ["", f"GPUs: declared, {declared} (scaleOut.sharedGpu {str(bool(scale.get('shared_gpu'))).lower()}). "
                    f"Given to each replica's container, from the launch records, with the UUIDs nvidia-smi read "
                    f"through each request before the first round: {shown}."]
    return out
