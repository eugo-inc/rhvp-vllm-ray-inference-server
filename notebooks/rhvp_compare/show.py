# SPDX-License-Identifier: Apache-2.0
"""The comparison notebook's cells: read a results file, show it as tables. Nothing is measured here.

Settings (environment variables, the notebook chart sets the first while global.compare.enabled is true):

- RHVP_COMPARE_RESULTS: the results volume (default /results); every run is a folder with a results.json.
- RHVP_COMPARE_REPLAY=<results.json>: REPLAY mode, that file only, with the amber REPLAY strip above every result.

Every visible sentence comes from words.py or from the results file, escaped. The styles follow rhvp_bench.draw: each
theme variable written twice, the plain colour first.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path

from rhvp_bench import draw

from . import stats, summary, words

GOOD, BAD, NEUTRAL, OPEN = "#1E7B45", "#B3261E", draw.MUTED, draw.AMBER
VERDICT_COLOUR = {"FASTER": GOOD, "MORE_EFFICIENT": GOOD, "SLOWER": BAD, "LESS_EFFICIENT": BAD, "TIE": NEUTRAL,
                  "INCONCLUSIVE": OPEN}


# ---- reading ----

def _replay_path() -> Path | None:
    value = os.environ.get("RHVP_COMPARE_REPLAY", "").strip()
    return Path(value) if value else None


def _results_root() -> Path:
    return Path(os.environ.get("RHVP_COMPARE_RESULTS", "").strip() or "/results")


def load_all() -> tuple[list[dict], list[str]]:
    """Every readable results file, newest first, and a note per file that could not be read."""
    replay = _replay_path()
    paths = [replay] if replay else sorted(_results_root().glob("*/results.json"))
    found, problems = [], []
    for path in paths:
        try:
            found.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as error:
            problems.append((words.REPLAY_UNUSABLE if replay else words.UNREADABLE).format(
                error=type(error).__name__))
    found.sort(key=lambda r: (r.get("run") or {}).get("started_at") or "", reverse=True)
    return found, problems


def pick(run: str | None = None) -> tuple[dict | None, list[dict], list[str]]:
    """(the chosen run or None, every run, notices). Without `run`, the newest."""
    found, problems = load_all()
    if not found:
        return None, found, problems
    if run is None:
        return found[0], found, problems
    chosen = next((r for r in found if (r.get("run") or {}).get("id") == run), None)
    if chosen is None:
        ids = ", ".join((r.get("run") or {}).get("id", "?") for r in found)
        problems.append(words.NOT_FOUND.format(run=run, runs=ids))
    return chosen, found, problems


# ---- markup ----

def _t(value: object) -> str:
    return html.escape(str(value))


def strip(text: str) -> str:
    return draw.doubled(
        f'<div data-rhvp="strip" style="font-family:{draw.FONT};max-width:{draw.WIDTH};margin:8px 0;'
        f'padding:6px 12px;font-size:12px;font-weight:600;letter-spacing:.04em;color:{draw.AMBER};'
        f'border:1px solid {draw.AMBER};border-radius:6px">{_t(text)}</div>')


def line(text: str) -> str:
    return draw.doubled(f'<div style="font-family:{draw.FONT};font-size:13px;line-height:1.7;color:{draw.SOFT};'
                        f'max-width:{draw.WIDTH}">{_t(text)}</div>')


def chip(verdict: str) -> str:
    colour = VERDICT_COLOUR.get(verdict, NEUTRAL)
    return (f'<span data-rhvp="verdict" style="font-weight:700;letter-spacing:.03em;color:{colour};'
            f'border-left:3px solid {colour};padding-left:6px">{_t(verdict)}</span>')


def verdict_cell(row: dict) -> str:
    """The verdict chip, marked "descriptive" for a secondary metric (its p-value is not corrected)."""
    cell = chip(row.get("verdict", "INCONCLUSIVE"))
    if row.get("role") == "secondary":
        cell += (f' <span data-rhvp="role" style="font-size:11px;color:{draw.MUTED}">'
                 f'{_t(words.SECONDARY_MARK)}</span>')
    return cell


def table(headers: tuple[str, ...], rows: list[list[str]], raw_last: bool = False) -> str:
    """A plain table. Cells are escaped, except the last column's when `raw_last` (verdict chips)."""
    head = "".join(f'<th style="text-align:left;padding:6px 10px;border-bottom:2px solid {draw.BORDER};'
                   f'color:{draw.MUTED};font-weight:600">{_t(h)}</th>' for h in headers)
    body = []
    for row in rows:
        cells = []
        for at, cell in enumerate(row):
            content = cell if (raw_last and at == len(row) - 1) else _t(cell)
            cells.append(f'<td style="padding:6px 10px;border-bottom:1px solid {draw.BORDER};'
                         f'font-variant-numeric:tabular-nums">{content}</td>')
        body.append(f"<tr>{''.join(cells)}</tr>")
    return draw.doubled(
        f'<table data-rhvp="table" style="font-family:{draw.FONT};font-size:13px;color:{draw.INK};'
        f'border-collapse:collapse;margin:8px 0;max-width:{draw.WIDTH}"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table>')


def _num(value, unit: str = "") -> str:
    if not isinstance(value, (int, float)):
        return words.NO_FIGURE
    if unit == "share":
        return f"{value:.1%}"
    if unit == "s":
        text = f"{value:.2f}"
    elif unit == "ms":
        text = f"{value:.1f}"
    elif unit in ("tokens/J", "W"):
        text = f"{value:.3g}"
    else:
        text = f"{value:.1f}" if abs(value) < 1000 else f"{value:.0f}"
    return f"{text} {unit}".strip()


def _ratio(row: dict) -> str:
    if row.get("median_ratio") is None:
        return words.NO_FIGURE
    ci = row.get("ci") or {}
    if ci.get("low") is None:
        return f"{row['median_ratio']:.3f}"
    return f"{row['median_ratio']:.3f} [{ci['low']:.3f}, {ci['high']:.3f}]"


def _p(value) -> str:
    return words.NO_FIGURE if value is None else (f"{value:.3g}" if value >= 0.001 else f"{value:.1e}")


# ---- the frame around every result ----

def _retried(round_: dict) -> str:
    """A round that started a replica again: "3 (side A)", or with the replicas when a scale-out round added one."""
    again = [str(item.get("replica")) for item in round_.get("launches") or [] if item.get("start_retries")]
    if any(n != "1" for n in again):
        return words.RETRIED_REPLICAS.format(index=round_["index"], side=round_["side"], replicas=", ".join(again))
    return words.RETRIED_ROUND.format(index=round_["index"], side=round_["side"])


def _head(results: dict | None, found: list[dict], problems: list[str]) -> list[str]:
    parts = []
    if _replay_path():
        parts.append(strip(words.REPLAY_STRIP))
    if results and (results.get("run") or {}).get("synthetic"):
        parts.append(strip(words.SYNTHETIC_STRIP))
    parts += [draw.notice(p) for p in problems]
    if not results:
        return parts
    run, plan = results.get("run") or {}, results["plan"]
    done = sum(1 for r in results.get("rounds") or [] if r.get("status") == "ok")
    parts.append(line(words.RUN_LINE.format(run=run.get("id"), status=run.get("status"), done=done,
                                            planned=len(plan["rounds"]), platform=run.get("platform"))))
    a, b = plan["sides"]
    parts.append(line(words.SIDE_LINE.format(a=f"{a['name']} ({a['image']})", b=f"{b['name']} ({b['image']})")))
    failed = [f"{r['index']} ({r['error_code']})" if r.get("error_code") else str(r["index"])
              for r in results.get("rounds") or [] if r.get("status") != "ok"]
    if failed:
        parts.append(line(words.RUN_FAILED_ROUNDS.format(rounds=", ".join(failed))))
    retried = [_retried(r) for r in results.get("rounds") or [] if r.get("start_retries")]
    if retried:
        parts.append(line(words.RUN_RETRIED.format(rounds=", ".join(retried))))
    unusable = [(words.LEVEL_SURGE if u["concurrency"] is None else words.LEVEL_AT).format(**u)
                for u in summary.levels_unusable(results)]
    if unusable:
        parts.append(line(words.RUN_LEVELS.format(levels=", ".join(unusable))))
    if run.get("error_code"):
        parts.append(line(words.RUN_CODE.format(code=run["error_code"])))
    lost = [f"{row['label']} (A {row['pairs_lost']['A']}, B {row['pairs_lost']['B']})"
            for row in (results.get("analysis") or {}).get("metrics") or []
            if (row.get("pairs_lost") or {}).get("A") or (row.get("pairs_lost") or {}).get("B")]
    if lost:
        parts.append(line(words.RUN_LOST.format(lost="; ".join(lost))))
    if run.get("server_restored") is False:
        parts.append(draw.notice(words.RUN_SERVER))
    return parts


def _frame(run: str | None):
    """(results, parts so far); results None means the parts already say why there is nothing to show."""
    results, found, problems = pick(run)
    parts = _head(results, found, problems)
    if results is None and not problems:
        parts.append(draw.notice(words.NO_RESULTS, words.NO_RESULTS_LINES))
    return results, parts


def _caches(plan: dict) -> tuple[str, str]:
    """Each side's compileCache (cold when the results file predates the setting)."""
    a, b = plan["sides"]
    return a.get("compile_cache") or "cold", b.get("compile_cache") or "cold"


def _cache_note(plan: dict) -> str | None:
    """A notice when the two sides started with different compile caches, a line when both were warm, else None."""
    a, b = _caches(plan)
    if a != b:
        return draw.notice(words.CACHE_DIFFER.format(a=words.CACHE_NAMES.get(a, a), b=words.CACHE_NAMES.get(b, b)))
    if a == "warm":
        return line(words.CACHE_BOTH_WARM)
    return None


def _rows(results: dict) -> dict[str, dict]:
    return {row["id"]: row for row in (results.get("analysis") or {}).get("metrics") or []}


# ---- the cells ----

def runs() -> None:
    """Which runs the results volume holds (the first code cell)."""
    results, parts = _frame(None)
    if results:
        found, _ = load_all()
        others = [(r.get("run") or {}).get("id", "?") for r in found[1:]]
        if others:
            parts.append(line(words.RUN_OTHERS.format(runs=", ".join(others))))
    draw.show(*parts)


def verdicts(run: str | None = None) -> None:
    results, parts = _frame(run)
    if results:
        method = results["plan"]["analysis"]
        floor = results["plan"]["floor"]
        band = method["tie_band"]
        everything = list(_rows(results).values())
        for role, title, headers, p_key in (("primary", words.PRIMARY_TITLE, words.VERDICT_HEADERS, "p_adjusted"),
                                            ("secondary", words.SECONDARY_TITLE, words.SECONDARY_HEADERS,
                                             "p_value")):
            rows = [[row["label"], _num(row.get("a_median"), row["unit"]), _num(row.get("b_median"), row["unit"]),
                     _ratio(row), str(row["n_pairs"]), _p(row.get(p_key)), chip(row["verdict"])]
                    for row in everything if row.get("role", "primary") == role]
            if rows:
                parts.append(line(title))
                parts.append(table(headers, rows, raw_last=True))
                if role == "primary" and _cache_note(results["plan"]):
                    parts.append(_cache_note(results["plan"]))
        parts.append(line(words.METHOD_LINE.format(
            test=words.TEST_NAMES.get(method["test"], method["test"]),
            correction=words.CORRECTION_NAMES.get(method["correction"], method["correction"]),
            family=floor["family_size"], alpha=method["alpha"], band=f"±{band:.0%}", low=1 / (1 + band),
            high=1 + band)))
        if not floor["reachable"]:
            parts.append(draw.notice(words.FLOOR_NOTE.format(
                pairs=results["plan"]["pairs"], family=floor["family_size"], floor=floor["min_adjusted_p"],
                alpha=method["alpha"], needed=floor["pairs_needed"])))
    draw.show(*parts)


def _scored(results: dict, side: str) -> list[dict]:
    return [r for r in results.get("rounds") or [] if r.get("kind") == "scored" and r.get("side") == side
            and r.get("status") == "ok"]


def _median(values: list[float]) -> float | None:
    return stats.median(values) if values else None


def startup(run: str | None = None) -> None:
    results, parts = _frame(run)
    if results:
        rows = _rows(results)
        table_rows = []
        for metric, label in (("startup.ready_s", words.STARTUP_READY), ("startup.first_answer_s",
                                                                         words.STARTUP_FIRST)):
            row = rows.get(metric) or {}
            table_rows.append([label, _num(row.get("a_median"), "s"), _num(row.get("b_median"), "s"),
                               _ratio(row) if row else words.NO_FIGURE, verdict_cell(row)])
        parts.append(table(words.STARTUP_HEADERS, table_rows, raw_last=True))
        if _cache_note(results["plan"]):
            parts.append(_cache_note(results["plan"]))
        if any(rows.get(m, {}).get("role") == "secondary" for m in ("startup.ready_s", "startup.first_answer_s")):
            parts.append(line(words.SECONDARY_NOTE))
        phases = _phase_table(results, words.PHASE_HEADERS, lambda r: (r.get("startup") or {}).get("breakdown"))
        parts += [phases, line(words.PHASE_NOTE)] if phases else [line(words.PHASE_NONE)]
    draw.show(*parts)


def _phase_table(results: dict, headers: tuple[str, ...], breakdown) -> str | None:
    """The median of each start phase per side, over the scored rounds; `breakdown(round)` gives a round's phase
    list. None when no round has a timed phase."""
    phases: dict[str, dict] = {}
    for side in ("A", "B"):
        for r in _scored(results, side):
            for entry in breakdown(r) or []:
                if isinstance(entry.get("seconds"), (int, float)):
                    slot = phases.setdefault(entry["phase"], {"label": entry["label"], "A": [], "B": []})
                    slot[side].append(entry["seconds"])
    if not phases:
        return None
    return table(headers, [
        [p["label"], _num(_median(p["A"]), "s"), _num(_median(p["B"]), "s"), f"{len(p['A'])}, {len(p['B'])}"]
        for p in phases.values()])


def energy(run: str | None = None) -> None:
    results, parts = _frame(run)
    if results:
        plan = results["plan"]
        if plan["energy"]["source"] == "none":
            parts.append(draw.notice(words.ENERGY_OFF))
        else:
            rows = _rows(results)
            table_rows, missing = [], []
            for c in plan["load"]["concurrencies"]:
                row = rows.get(f"c{c}.tokens_per_joule") or {}
                power = {}
                for side in ("A", "B"):
                    values = [lv["energy"]["mean_power_w"] for r in _scored(results, side) for lv in r["levels"]
                              if lv.get("concurrency") == c and isinstance((lv.get("energy") or {}).get(
                                  "mean_power_w"), (int, float))]
                    power[side] = _median(values)
                if not row.get("n_pairs"):
                    missing.append(str(c))
                table_rows.append([str(c), _num(row.get("a_median"), "tokens/J"), _num(row.get("b_median"),
                                                                                       "tokens/J"),
                                   _ratio(row) if row else words.NO_FIGURE, _num(power["A"], "W"),
                                   _num(power["B"], "W"), verdict_cell(row)])
            parts.append(table(words.ENERGY_HEADERS, table_rows, raw_last=True))
            if any((rows.get(f"c{c}.tokens_per_joule") or {}).get("role") == "secondary"
                   for c in plan["load"]["concurrencies"]):
                parts.append(line(words.SECONDARY_NOTE))
            parts.append(line(words.ENERGY_LINE.format(
                source=words.ENERGY_SOURCE.get(plan["energy"]["source"], plan["energy"]["source"]),
                label=plan["energy"]["label"])))
            if missing:
                parts.append(line(words.ENERGY_MISSING.format(what=", ".join(missing))))
    draw.show(*parts)


def _goals(slo: dict) -> str:
    goals = []
    if slo.get("ttft_ms") is not None:
        goals.append(words.SCALE_GOAL_TTFT.format(value=slo["ttft_ms"]))
    if slo.get("tpot_ms") is not None:
        goals.append(words.SCALE_GOAL_TPOT.format(value=slo["tpot_ms"]))
    return words.SCALE_GOALS_JOIN.join(goals) or words.NO_FIGURE


def scale_out(run: str | None = None) -> None:
    """Mode scaleout: the forced step, the second replica's start, and how the router shared the surge out."""
    results, parts = _frame(run)
    if results:
        plan = results["plan"]
        scale = plan.get("scale_out")
        if plan.get("mode") != "scaleout" or not scale:
            parts.append(draw.notice(words.SCALE_OFF))
            draw.show(*parts)
            return
        parts.append(line(words.SCALE_LINE.format(rate=scale["request_rate"], surge=scale["surge_s"],
                                                  step=scale["step_after_s"], goals=_goals(scale.get("slo") or {}))))
        if not any(side.get("gpu", True) for side in plan["sides"]):
            parts.append(line(words.SCALE_GPU_NONE))
        else:
            parts.append(line(words.SCALE_GPU_SHARED if scale.get("shared_gpu") else words.SCALE_GPU_OWN))
            given = summary.replica_gpus(results)
            if given:
                parts.append(line(words.SCALE_GPU_GIVEN.format(given="; ".join(
                    words.SCALE_GPU_REPLICA.format(replica=n, gpus=" or ".join(g)) for n, g in given.items()))))
        rows = _rows(results)
        declared = [m for m in plan["metrics"] if m["id"].startswith("scale.")]
        ids = [m["id"] for m in declared]
        parts.append(table(words.SCALE_HEADERS, [
            [m["label"], _num((rows.get(m["id"]) or {}).get("a_median"), m["unit"]),
             _num((rows.get(m["id"]) or {}).get("b_median"), m["unit"]),
             _ratio(rows[m["id"]]) if m["id"] in rows else words.NO_FIGURE, verdict_cell(rows.get(m["id"]) or {})]
            for m in declared], raw_last=True))
        if _cache_note(plan):
            parts.append(_cache_note(plan))
        if any((rows.get(m) or {}).get("role") == "secondary" for m in ids):
            parts.append(line(words.SECONDARY_NOTE))
        stepped = {side: [r for r in _scored(results, side) if (r.get("scale") or {}).get("new_replica")]
                   for side in ("A", "B")}
        if not stepped["A"] and not stepped["B"]:
            parts.append(line(words.SCALE_NONE))
            draw.show(*parts)
            return
        facts = []
        for label, unit, get in (
                (words.SCALE_ASKED, "s", lambda s: s.get("step_at_s")),
                (words.SCALE_JOINED, "s", lambda s: s["new_replica"].get("joined_at_s")
                 if s["new_replica"].get("joined_during_surge") else None),
                (words.SCALE_SHARE, "share", lambda s: s.get("new_replica_share"))):
            values = {side: [v for r in stepped[side] if isinstance(v := get(r["scale"]), (int, float))]
                      for side in ("A", "B")}
            facts.append([label, _num(_median(values["A"]), unit), _num(_median(values["B"]), unit),
                          f"{len(values['A'])}, {len(values['B'])}"])
        parts.append(table(words.SCALE_FACT_HEADERS, facts))
        late = [str(r["index"]) for side in ("A", "B") for r in stepped[side]
                if r["scale"]["new_replica"].get("joined_during_surge") is False]
        if late:
            parts.append(line(words.SCALE_LATE.format(rounds=", ".join(sorted(late, key=int)))))
        counts = []
        for side in ("A", "B"):
            sums: dict[int, dict] = {}
            for r in stepped[side]:
                for b in (r["scale"].get("router") or {}).get("backends") or []:
                    slot = sums.setdefault(b["replica"], {"sent": 0, "finished": 0, "failed": 0, "cancelled": 0,
                                                          "max_in_flight": 0})
                    for key in ("sent", "finished", "failed", "cancelled"):
                        slot[key] += b.get(key) or 0
                    slot["max_in_flight"] = max(slot["max_in_flight"], b.get("max_in_flight") or 0)
            counts += [[side, str(n), str(c["sent"]), str(c["finished"]), str(c["failed"]), str(c["cancelled"]),
                        str(c["max_in_flight"])] for n, c in sorted(sums.items())]
        if counts:
            parts.append(table(words.SCALE_REQUESTS_HEADERS, counts))
            parts.append(line(words.SCALE_REQUESTS_NOTE))
        parts.append(line(words.SCALE_SURGE_RULE))
        logged = {side: sum(1 for r in stepped[side] if r["scale"].get("load_triggered_s") is not None)
                  for side in ("A", "B")}
        parts.append(line(words.SCALE_EVENTS.format(a=logged["A"], na=len(stepped["A"]), b=logged["B"],
                                                    nb=len(stepped["B"]))))
        phases = _phase_table(results, words.SCALE_PHASE_HEADERS,
                              lambda r: ((r.get("scale") or {}).get("new_replica") or {}).get("breakdown"))
        if phases:
            parts += [phases, line(words.PHASE_NOTE)]
    draw.show(*parts)


def provenance(run: str | None = None) -> None:
    results, parts = _frame(run)
    if results:
        plan, prov = results["plan"], results.get("provenance") or {}
        sides = prov.get("sides") or {}
        rows = []
        for side, cache in zip(plan["sides"], _caches(plan)):
            p = sides.get(side["key"]) or {}
            version = (p.get("server_version") or {}).get("vllm")
            rows.append([side["key"], side["name"], side["image"], p.get("image_digest") or words.NO_FIGURE,
                         f"vLLM {version}" if version else words.NO_FIGURE, words.CACHE_NAMES.get(cache, cache)])
        parts.append(table(words.SIDES_HEADERS, rows))
        gpu, node, cluster = prov.get("gpu") or {}, prov.get("node") or {}, prov.get("cluster") or {}
        parts.append(line(words.MACHINE_LINE.format(
            gpu=gpu.get("product") or words.NO_FIGURE, count=gpu.get("count") or words.NO_FIGURE,
            driver=gpu.get("driver") or words.NO_FIGURE, node=node.get("instance_type") or words.NO_FIGURE,
            arch=node.get("arch") or words.NO_FIGURE, openshift=cluster.get("openshift_version") or words.NO_FIGURE)))
        load = plan["load"]
        common = {"version": (prov.get("load_tool") or {}).get("version") or words.NO_FIGURE,
                  "prompt": load["prompt_tokens"], "output": load["output_tokens"],
                  "seconds": load["seconds_per_level"], "seed": load["seed"], "server_seed": plan["server"]["seed"]}
        if plan.get("mode") == "scaleout" and plan.get("scale_out"):
            parts.append(line(words.SURGE_LOAD_LINE.format(rate=plan["scale_out"]["request_rate"], **common)))
        else:
            parts.append(line(words.LOAD_LINE.format(levels=", ".join(str(c) for c in load["concurrencies"]),
                                                     **common)))
        parts.append(line(words.PLAN_LINE.format(
            pairs=plan["pairs"], warmup=words.PLAN_WARMUP if plan.get("warmup") else "",
            aa=words.PLAN_AA.format(n=plan["aa_pairs"]) if plan.get("aa_pairs") else "")))
        noise = (results.get("analysis") or {}).get("noise_floor")
        if noise:
            labels = {m["id"]: m["label"] for m in plan["metrics"]}
            parts.append(line(words.NOISE_NOTE))
            parts.append(table(words.NOISE_HEADERS, [
                [labels.get(n["id"], n["id"]), str(n["pairs"]),
                 words.NO_FIGURE if n["median_abs_change"] is None else f"{n['median_abs_change']:.1%}",
                 words.NO_FIGURE if n["max_abs_change"] is None else f"{n['max_abs_change']:.1%}"] for n in noise]))
    draw.show(*parts)
