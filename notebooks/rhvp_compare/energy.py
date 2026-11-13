# SPDX-License-Identifier: Apache-2.0
"""Energy per token from GPU power: tokens per joule for each concurrency level of a round.

What is measured is GPU-only power: the board power the GPU itself reports. CPU, memory, fans and power-supply
losses are not in it. Every figure carries that label.

Two sources of power samples, each a list of (time, watts):

- DCGM, on OpenShift. The NVIDIA GPU Operator's DCGM exporter publishes the board power in watts and, where the GPU
  supports it, an energy counter since boot. Exporters before 4.9.0 name them `DCGM_FI_DEV_POWER_USAGE` and
  `DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION` (millijoules); 4.9.0 renames them `DCGM_FI_DEV_BOARD_POWER_WATTS` and
  `DCGM_FI_DEV_GPU_ENERGY_JOULES_TOTAL` (joules). Both names are read, the older first. The cluster's monitoring
  scrapes them, and the runner reads the raw samples of a window from the Thanos Querier with a range
  selector (no interpolation). Which series belongs to the round's pod is found by the exporter's pod labels, or by
  the node when it has a single GPU series.
- nvidia-smi, under docker (or anywhere DCGM is not there): `nvidia-smi --query-gpu=power.draw` every 500 ms in a
  container that sees the same GPU, each line timestamped by the container log.

For a level's window [start, end] (from GuideLLM's report):

- with two or more energy-counter samples inside the window: mean power = (E_last - E_first) / (t_last - t_first),
  after dropping repeats of an unchanged counter value (a scrape faster than the exporter's own update repeats the
  last value, and a repeat at either end would stretch or shrink the time the energy is divided by);
- otherwise the mean of the power samples inside the window, which needs at least `min_samples` of them;
- tokens per joule = the level's output tokens per second / mean power (W = J/s), and joules per token its inverse.

With fewer samples the figure is left out with a note, never estimated from one sample. Standard library only, plus
httpx for the Thanos query.
"""

from __future__ import annotations

import math
import ssl
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable, Mapping

from . import logs

LABEL = "GPU-only power"
#: (metric name, factor to watts) for power and (metric name, factor to millijoules) for the energy counter, in the
#: order they are tried: the exporter's names before 4.9.0, then its names from 4.9.0 on.
POWER_METRICS = (("DCGM_FI_DEV_POWER_USAGE", 1.0), ("DCGM_FI_DEV_BOARD_POWER_WATTS", 1.0))
ENERGY_METRICS = (("DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION", 1.0), ("DCGM_FI_DEV_GPU_ENERGY_JOULES_TOTAL", 1000.0))
SERVICE_ACCOUNT = Path("/var/run/secrets/kubernetes.io/serviceaccount")


@dataclass(frozen=True)
class Sample:
    t: float      # seconds since the epoch
    value: float  # watts, or millijoules for the energy counter


# ---- parsing ----

def parse_nvidia_smi(log: str) -> list[Sample]:
    """Samples from a timestamped log of `nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits`. A value
    such as "[N/A]" is skipped; with several GPUs on one line set, only the first value of each line is read."""
    out = []
    for at, text in logs.lines(log):
        if at is None:
            continue
        head = text.split(",")[0].strip().removesuffix("W").strip()
        try:
            watts = float(head)
        except ValueError:
            continue
        if math.isfinite(watts) and watts >= 0:
            out.append(Sample(at, watts))
    return out


def parse_matrix(answer: Mapping) -> list[tuple[dict, list[Sample]]]:
    """The series of a Prometheus/Thanos `matrix` answer: (labels, samples)."""
    if answer.get("status") != "success":
        raise ValueError(f"query failed: {answer.get('errorType') or answer.get('error') or 'unknown error'}")
    data = answer.get("data") or {}
    if data.get("resultType") != "matrix":
        raise ValueError(f"expected a matrix, got {data.get('resultType')!r}")
    out = []
    for series in data.get("result") or []:
        samples = []
        for t, v in series.get("values") or []:
            try:
                value = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                samples.append(Sample(float(t), value))
        out.append((dict(series.get("metric") or {}), samples))
    return out


# ---- the window ----

def inside(samples: Iterable[Sample], start: float, end: float) -> list[Sample]:
    return sorted((s for s in samples if start <= s.t <= end), key=lambda s: s.t)


@dataclass(frozen=True)
class Power:
    mean_w: float | None
    samples: int
    method: str | None   # "energy_counter" or "power_mean"
    note: str | None


def collapse(samples: list[Sample]) -> list[Sample]:
    """Keep the first sample of every run of equal values: the time each counter value first appeared."""
    out: list[Sample] = []
    for sample in samples:
        if not out or sample.value != out[-1].value:
            out.append(sample)
    return out


def window_power(power: Iterable[Sample], start: float, end: float, min_samples: int = 2,
                 counter: Iterable[Sample] = ()) -> Power:
    """Mean GPU power over [start, end]: from the energy counter when it has two distinct samples in the window,
    else the mean of the power samples (at least `min_samples` of them)."""
    in_counter = collapse(inside(counter, start, end))
    if len(in_counter) >= 2 and in_counter[-1].t > in_counter[0].t:
        joules = (in_counter[-1].value - in_counter[0].value) / 1000.0
        if joules >= 0:
            return Power(joules / (in_counter[-1].t - in_counter[0].t), len(in_counter), "energy_counter", None)
    in_power = inside(power, start, end)
    if len(in_power) < max(min_samples, 1):
        return Power(None, len(in_power), None, f"{len(in_power)} power samples in the window; at least "
                     f"{max(min_samples, 1)} are needed")
    return Power(sum(s.value for s in in_power) / len(in_power), len(in_power), "power_mean", None)


def level_energy(level: Mapping, power: Power, source: str) -> dict:
    """The `energy` entry of one level: mean power, energy over the window, tokens per joule, joules per token."""
    rate = level.get("output_tokens_per_s")
    duration = level.get("duration_s")
    out = {"source": source, "label": LABEL, "method": power.method, "samples": power.samples,
           "mean_power_w": power.mean_w, "energy_j": None, "tokens_per_joule": None, "joules_per_token": None,
           "note": power.note}
    if power.mean_w and power.mean_w > 0:
        if isinstance(duration, (int, float)):
            out["energy_j"] = power.mean_w * duration
        if isinstance(rate, (int, float)) and rate > 0:
            out["tokens_per_joule"] = rate / power.mean_w
            out["joules_per_token"] = power.mean_w / rate
        elif isinstance(rate, (int, float)) and rate == 0:  # power drawn, no token out: 0 per joule, not unknown
            out["tokens_per_joule"] = 0.0
    elif power.mean_w is not None and out["note"] is None:
        out["note"] = "mean power was 0 W"
    return out


def none_entry(source: str, note: str) -> dict:
    return {"source": source, "label": LABEL, "method": None, "samples": 0, "mean_power_w": None, "energy_j": None,
            "tokens_per_joule": None, "joules_per_token": None, "note": note}


# ---- DCGM through the Thanos Querier ----

def selectors(pod: str, namespace: str, node: str | None) -> list[tuple[str, str]]:
    """Label selectors to try, in order, with what each means: the exporter's pod labels as Prometheus renames them
    (exported_*), the same labels kept as they are, then the node's GPU series."""
    out = [("pod (exported labels)", f'exported_pod="{pod}",exported_namespace="{namespace}"'),
           ("pod", f'pod="{pod}",namespace="{namespace}"')]
    if node:
        out.append(("node", f'Hostname="{node}"'))
    return out


def bearer(path: Path, token: str | None = None):
    """An httpx auth callable that sends `token`, or else the token in the file at `path`, read again for every
    request: the kubelet replaces a pod's service account token file before the token expires, and a run can last
    two days."""
    def apply(request):
        value = token or (path.read_text(encoding="utf-8").strip() if path.is_file() else "")
        if value:
            request.headers["Authorization"] = f"Bearer {value}"
        return request
    return apply


class Thanos:
    """Raw samples from the Thanos Querier, with the pod's service account token (needs cluster-monitoring-view)."""

    def __init__(self, url: str, transport=None, token: str | None = None, verify=None,
                 token_path: Path = SERVICE_ACCOUNT / "token") -> None:
        import httpx

        if verify is None:
            ca = SERVICE_ACCOUNT / "service-ca.crt"
            verify = ssl.create_default_context(cafile=str(ca)) if ca.is_file() else True
        self._client = httpx.Client(base_url=url, auth=bearer(token_path, token), verify=verify, timeout=30.0,
                                    transport=transport)

    def close(self) -> None:
        self._client.close()

    def range_samples(self, metric: str, selector: str, start: float, end: float) -> list[tuple[dict, list[Sample]]]:
        """Every sample of `metric{selector}` in (start - 1 s, end], evaluated at `end`."""
        import httpx

        seconds = max(1, int(math.ceil(end - start)) + 1)
        try:
            response = self._client.get("/api/v1/query", params={"query": f"{metric}{{{selector}}}[{seconds}s]",
                                                                 "time": f"{end:.3f}"})
        except httpx.HTTPError as error:
            raise ValueError(f"the Thanos Querier did not answer ({type(error).__name__})") from None
        if response.status_code != 200:
            raise ValueError(f"the Thanos Querier answered HTTP {response.status_code}")
        return parse_matrix(response.json())


def dcgm_samples(thanos: Thanos, pod: str, namespace: str, node: str | None, start: float,
                 end: float) -> tuple[list[Sample], list[Sample], str | None, str | None]:
    """(power samples, energy-counter samples, the selector that matched, a note) for one window. The node
    selector is used only when the node has exactly one GPU series, so another GPU's power is never read."""
    note = None
    for meaning, selector in selectors(pod, namespace, node):
        for name, factor in POWER_METRICS:
            try:
                power = thanos.range_samples(name, selector, start, end)
            except ValueError as error:
                return [], [], None, str(error)
            if power:
                break
        if not power:
            continue
        if len(power) > 1:
            note = f"{len(power)} GPU series matched by {meaning}; none was used"
            continue
        watts = [Sample(s.t, s.value * factor) for s in power[0][1]]
        return watts, _counter(thanos, selector, start, end), f"{meaning}, {name}", None
    return [], [], None, note or "no DCGM power series for this pod or its node in the cluster's monitoring"


def _counter(thanos: Thanos, selector: str, start: float, end: float) -> list[Sample]:
    """The energy counter in millijoules under either name, or [] (then the power samples are used)."""
    for name, factor in ENERGY_METRICS:
        try:
            series = thanos.range_samples(name, selector, start, end)
        except ValueError:
            return []
        if len(series) == 1:
            return [Sample(s.t, s.value * factor) for s in series[0][1]]
    return []
