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

from . import plan as plans, stats, summary, words

GOOD, BAD, NEUTRAL, OPEN = "#1E7B45", "#B3261E", draw.MUTED, draw.AMBER
VERDICT_COLOUR = {"FASTER": GOOD, "BETTER": GOOD, "SLOWER": BAD, "WORSE": BAD, "TIE": NEUTRAL, "UNRESOLVED": OPEN,
                  "NO VERDICT": NEUTRAL}


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
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            problems.append((words.REPLAY_UNUSABLE if replay else words.UNREADABLE).format(
                error=type(error).__name__))
            continue
        if isinstance(data, dict) and data.get("schema") == plans.SCHEMA:
            found.append(data)
        else:
            schema = data.get("schema") if isinstance(data, dict) else None
            problems.append(words.OLD_SCHEMA.format(schema=schema or "no schema"))
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
    """The verdict chip with its reason, marked "descriptive" for a secondary case (its p-value is not corrected)."""
    cell = chip(row.get("verdict", "NO VERDICT"))
    if row.get("reason"):
        cell += (f' <span data-rhvp="reason" style="font-size:11px;color:{draw.MUTED}">'
                 f'{_t(words.REASON_TEXT.get(row["reason"], row["reason"]))}</span>')
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
    return words.NO_FIGURE if row.get("ratio") is None else f"{row['ratio']:.3f}"


def _bar(row: dict) -> str:
    return words.NO_FIGURE if row.get("bar") is None else f"±{row['bar']:.1%}"


def _p(value) -> str:
    return words.NO_FIGURE if value is None else (f"{value:.3g}" if value >= 0.001 else f"{value:.1e}")


# ---- the frame around every result ----

def _retried(launch: dict, scale: bool) -> str:
    """A launch that started a server again, with how many servers it started: "3 (A, 2)" in mode levels; in mode
    scaleout "3 (A, 3 servers)" when only its first replica (of the kept try) was, else with the replicas ("3 (A, 5
    servers; replica 2 of the discarded try started again)"), from its server records."""
    index, side, starts = launch["index"], launch["side"], launch.get("starts")
    if not scale:
        return words.RETRIED_LAUNCH.format(index=index, side=side, starts=starts)
    again = [item for item in launch.get("servers") or [] if item.get("start_retries")]
    if len(again) > 1 or (again and (again[0].get("replica") != 1 or again[0].get("discarded"))):
        replicas = ", ".join(words.RETRIED_REPLICA.format(replica=item.get("replica"))
                             + (words.RETRIED_DISCARDED if item.get("discarded") else "") for item in again)
        return words.RETRIED_REPLICAS.format(index=index, side=side, starts=starts, replicas=replicas)
    return words.RETRIED_SERVERS.format(index=index, side=side, starts=starts)


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
    launches = results.get("launches") or []
    done = sum(1 for r in launches if r.get("status") == "ok")
    complete, planned = summary.rounds(results)
    parts.append(line(words.RUN_LINE.format(run=run.get("id"), status=run.get("status"), rounds=complete,
                                            planned=planned, done=done, launches=len(plan["launches"]),
                                            platform=run.get("platform"))))
    a, b = plan["sides"]
    parts.append(line(words.SIDE_LINE.format(a=f"{a['name']} ({a['image']})", b=f"{b['name']} ({b['image']})")))
    failed = [f"{r['index']} ({r['side']}, {r['error_code']})" if r.get("error_code") else str(r["index"])
              for r in launches if r.get("status") != "ok"]
    if failed:
        parts.append(line(words.RUN_FAILED_LAUNCHES.format(launches=", ".join(failed))))
    scale = plan.get("mode") == plans.SCALEOUT
    retried = [_retried(r, scale) for r in launches if r.get("start_retries")]
    if retried:
        parts.append(line((words.RUN_RETRIED_SCALE if scale else words.RUN_RETRIED).format(
            launches=", ".join(retried))))
    timed_out = [(words.LOAD_RETRIED_RAN if r.get("load_retries") else words.LOAD_RETRIED_NOT).format(
        index=r["index"], side=r["side"], code=r.get("error_code") or r.get("status")) for r in launches
        if r.get("load_timeouts")]
    if timed_out:
        ran = [r for r in launches if r.get("load_timeouts") and r.get("load_retries")]
        how = words.LOAD_RETRIED_SCALE if scale else words.LOAD_RETRIED_HOW
        if not ran:
            how = words.LOAD_RETRIED_NONE if len(timed_out) > 1 else words.LOAD_RETRIED_NONE_ONE
        elif len(ran) < len(timed_out):
            how += " " + words.LOAD_RETRIED_SOME
        parts.append(line(words.RUN_LOAD_RETRIED.format(launches=", ".join(timed_out), how=how)))
    troubled = summary.levels_unusable(results)
    unusable = [words.RUN_LEVEL.format(index=u["index"], side=u["side"], concurrency=u["concurrency"])
                for u in troubled if u["concurrency"] is not None]
    if unusable:
        parts.append(line(words.RUN_LEVELS.format(levels=", ".join(unusable))))
    surges = [words.RUN_SURGE.format(index=u["index"], side=u["side"], errored=u["errored"],
                                     successful=u["successful"]) for u in troubled if u["concurrency"] is None]
    if surges:
        parts.append(line(words.RUN_SURGES.format(surges=", ".join(surges))))
    mixed = [words.ENERGY_MIXED_LEVEL.format(concurrency=m["concurrency"], power=summary.launches(m["power_mean"]),
                                             counter=summary.launches(m["energy_counter"]))
             for m in summary.energy_methods_mixed(results)]
    if mixed:
        parts.append(line(words.ENERGY_MIXED.format(levels="; ".join(mixed))))
    if run.get("error_code"):
        parts.append(line(words.RUN_CODE.format(code=run["error_code"])))
    analysis = results.get("analysis") or {}
    lost = [f"{row['case']} (A {row['lost']['A']}, B {row['lost']['B']})"
            for row in analysis.get("comparisons") or [] if any((row.get("lost") or {}).values())]
    if lost:
        parts.append(line(words.RUN_LOST.format(lost="; ".join(lost))))
    twins = [t for t in analysis.get("same_vs_same") or [] if t.get("p_raw") is not None]
    differs = [words.TWIN_DIFFERS_CASE.format(side=t["side"], case=t["case"], p=f"{t['p_raw']:.3g}")
               for t in twins if t["p_raw"] < 0.05]
    if differs:
        parts.append(draw.notice(words.TWIN_DIFFERS.format(cases="; ".join(differs), tested=len(twins))))
    zero = [words.TWIN_ZERO_CASE.format(who=who.format(side=t["side"]), case=t["case"], affected=", ".join(t["cases"]),
                                        rounds=(words.ROUND_MANY if len(t[key]) > 1 else words.ROUND_ONE).format(
                                            n=", ".join(map(str, t[key]))))
            for t in analysis.get("nonpositive_twins") or []
            for who, key in ((words.TWIN_ZERO_SIDE, "side_rounds"), (words.TWIN_ZERO_TWIN, "twin_rounds")) if t[key]]
    if zero:
        parts.append(draw.notice(words.TWIN_ZERO.format(cases="; ".join(zero))))
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
    """Each side's compileCache."""
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
    return {row["case"]: row for row in (results.get("analysis") or {}).get("comparisons") or []}


def _labels(results: dict) -> dict[str, dict]:
    return {case["id"]: case for case in results["plan"]["cases"]}


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
        plan = results["plan"]
        power, rule = plan["power"], (results.get("analysis") or {}).get("rule") or {}
        cases = _labels(results)
        everything = list(_rows(results).values())
        for role, title, headers, p_key in (
                ("primary", words.PRIMARY_TITLE.format(plain=rule.get("plain") or power["plain"]),
                 words.VERDICT_HEADERS, "p_corrected"),
                ("secondary", words.SECONDARY_TITLE, words.SECONDARY_HEADERS, "p_raw")):
            rows = []
            for row in everything:
                if row.get("role", "primary") != role:
                    continue
                unit = cases.get(row["case"], {}).get("unit", "")
                rows.append([cases.get(row["case"], {}).get("label", row["case"]),
                             _num(row.get("baseline_median"), unit), _num(row.get("subject_median"), unit),
                             _ratio(row), _bar(row), str(row["n_rounds"]), _p(row.get(p_key)), verdict_cell(row)])
            if rows:
                parts.append(line(title))
                parts.append(table(headers, rows, raw_last=True))
                if role == "primary" and _cache_note(plan):
                    parts.append(_cache_note(plan))
        parts.append(line(words.METHOD_LINE.format(family=power["family_size"],
                                                   plain=rule.get("plain") or power["plain"],
                                                   alpha=plan["analysis"]["alpha"])))
        timeouts = (results.get("analysis") or {}).get("load_timeouts") or {}
        a, b = timeouts.get("A", 0), timeouts.get("B", 0)
        if a or b:
            parts.append(draw.notice(words.LOAD_TIMEOUTS.format(a=a, b=b)))
            if not (a and b):
                parts.append(draw.notice(words.LOAD_TIMEOUTS_ONE_SIDED.format(side="A" if a else "B")))
        if not power["both_twins"]:
            parts.append(draw.notice(words.TWINS_NONE))
        elif power["must_agree"] is None or not power["reachable"]:
            parts.append(draw.notice(words.POWER_NONE.format(rounds=plan["rounds"], family=power["family_size"],
                                                             needed=power["min_rounds"])))
        else:
            parts.append(line(words.POWER_NOTE.format(rounds=plan["rounds"], family=power["family_size"],
                                                      agree=power["must_agree"],
                                                      most=min(power["max_primary_cases"], power["family_size"]))))
    draw.show(*parts)


def _scored(results: dict, side: str) -> list[dict]:
    return [r for r in results.get("launches") or [] if r.get("kind") == "scored" and r.get("side") == side
            and r.get("status") == "ok"]


def _median(values: list[float]) -> float | None:
    return stats.median(values) if values else None


def startup(run: str | None = None) -> None:
    results, parts = _frame(run)
    if results:
        rows = _rows(results)
        table_rows = []
        for case, label in (("startup/ready_s", words.STARTUP_READY), ("startup/first_answer_s",
                                                                       words.STARTUP_FIRST)):
            row = rows.get(case) or {}
            table_rows.append([label, _num(row.get("baseline_median"), "s"), _num(row.get("subject_median"), "s"),
                               _ratio(row) if row else words.NO_FIGURE, verdict_cell(row)])
        parts.append(table(words.STARTUP_HEADERS, table_rows, raw_last=True))
        if _cache_note(results["plan"]):
            parts.append(_cache_note(results["plan"]))
        if any(rows.get(c, {}).get("role") == "secondary" for c in ("startup/ready_s", "startup/first_answer_s")):
            parts.append(line(words.SECONDARY_NOTE))
        phases = _phase_table(results, words.PHASE_HEADERS, lambda r: (r.get("startup") or {}).get("breakdown"))
        parts += [phases, line(words.PHASE_NOTE)] if phases else [line(words.PHASE_NONE)]
    draw.show(*parts)


def _phase_table(results: dict, headers: tuple[str, ...], breakdown) -> str | None:
    """The median of each start phase per side, over the side's own scored launches; `breakdown(launch)` gives a
    launch's phase list. None when no launch has a timed phase."""
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
                row = rows.get(f"c{c}/tokens_per_joule[gpu]") or {}
                power = {}
                for side in ("A", "B"):
                    values = [lv["energy"]["mean_power_w"] for r in _scored(results, side) for lv in r["levels"]
                              if lv.get("concurrency") == c and isinstance((lv.get("energy") or {}).get(
                                  "mean_power_w"), (int, float))]
                    power[side] = _median(values)
                if not row.get("n_rounds"):
                    missing.append(str(c))
                table_rows.append([str(c), _num(row.get("baseline_median"), "tokens/J"),
                                   _num(row.get("subject_median"), "tokens/J"),
                                   _ratio(row) if row else words.NO_FIGURE, _num(power["A"], "W"),
                                   _num(power["B"], "W"), verdict_cell(row)])
            parts.append(table(words.ENERGY_HEADERS, table_rows, raw_last=True))
            if any((rows.get(f"c{c}/tokens_per_joule[gpu]") or {}).get("role") == "secondary"
                   for c in plan["load"]["concurrencies"]):
                parts.append(line(words.SECONDARY_NOTE))
            parts.append(line(words.ENERGY_LINE.format(
                source=words.ENERGY_SOURCE.get(plan["energy"]["source"], plan["energy"]["source"]),
                label=plan["energy"]["label"], samples=plan["energy"]["min_samples"])))
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
        if plan.get("mode") != plans.SCALEOUT or not scale:
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
                    words.SCALE_GPU_REPLICA.format(replica=n, gpus=words.SCALE_GPU_OR.join(
                        words.SCALE_GPU_REQUEST.format(request=g["gpus"],
                                                       uuids=", ".join(g["uuids"]) or words.SCALE_GPU_NO_UUID)
                        for g in entries)) for n, entries in given.items()))))
        rows = _rows(results)
        declared = [c for c in plan["cases"] if c["id"].startswith(plans.SCALE + "/")]
        ids = [c["id"] for c in declared]
        parts.append(table(words.SCALE_HEADERS, [
            [c["label"], _num((rows.get(c["id"]) or {}).get("baseline_median"), c["unit"]),
             _num((rows.get(c["id"]) or {}).get("subject_median"), c["unit"]),
             _ratio(rows[c["id"]]) if c["id"] in rows else words.NO_FIGURE, verdict_cell(rows.get(c["id"]) or {})]
            for c in declared], raw_last=True))
        if _cache_note(plan):
            parts.append(_cache_note(plan))
        fit = summary.surge_rate_fit(results)
        if fit:
            parts.append(draw.notice(words.SCALE_RATE_NOTES[fit]))
        if any((rows.get(c) or {}).get("role") == "secondary" for c in ids):
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
            parts.append(line(words.SCALE_LATE.format(launches=", ".join(sorted(late, key=int)))))
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
                  "prompt": load["prompt_tokens"], "output": load["output_tokens"], "seed": load["seed"],
                  "server_seed": plan["server"]["seed"]}
        scale = plan.get("mode") == plans.SCALEOUT and plan.get("scale_out")
        if scale:
            parts.append(line(words.SURGE_LOAD_LINE.format(rate=scale["request_rate"], seconds=scale["surge_s"],
                                                           **common)))
        else:
            parts.append(line(words.LOAD_LINE.format(seconds=load["seconds_per_level"],
                                                     levels=", ".join(str(c) for c in load["concurrencies"]),
                                                     **common)))
        parts.append(line(words.PLAN_LINE.format(
            rounds=plan["rounds"], lineup=", ".join(e["key"] for e in plan["lineup"]),
            warmup=words.PLAN_WARMUP if plan.get("warmup") else "", launches=len(plan["launches"]))
            + (words.PLAN_SCALE.format(starts=plans.server_starts(plan)) if scale else "")))
        twins = (results.get("analysis") or {}).get("same_vs_same") or []
        if twins:
            labels = _labels(results)
            parts.append(line(words.TWIN_NOTE))
            parts.append(table(words.TWIN_HEADERS, [
                [t["side"], labels.get(t["case"], {}).get("label", t["case"]), f"{t['ratio']:.3f}",
                 f"{t['floor']:.1%}", str(t["n"]), _p(t.get("p_raw"))] for t in twins]))
    draw.show(*parts)
