# SPDX-License-Identifier: Apache-2.0
"""The scale-out settings and their rules, for scripts/validate-values.sh and the tests.

The same rules fail a chart render: charts/all/ray-serve-llm/templates/_settings.tpl for the server chart's values
and global.scaleOut, charts/all/notebook/templates/_helpers.tpl (notebook.compareConfig) for global.compare.mode.
Keep the three in step; tests/test_scaleout_contract.py checks that a render and this module give the same message
for the same mistake.

`check(values)` takes the server chart's values as Helm sees them: charts/all/ray-serve-llm/values.yaml with
values-global.yaml, the variant file and the override files merged over it. It returns one message per broken rule,
or an empty list. Standard library only.
"""

from __future__ import annotations

from collections.abc import Mapping

#: global.scaleOut.mode values that are built.
SCALE_OUT_MODES = ("none", "keda")
#: global.scaleOut.mode values that are designed but not built yet: refused with KUBERAY_NOT_BUILT.
SCALE_OUT_DESIGNED = ("kuberay",)
KUBERAY_NOT_BUILT = "global.scaleOut.mode: kuberay mode is designed but not built yet; use none or keda"
#: global.compare.mode values; pairs is the default.
COMPARE_MODES = ("pairs", "scaleout")
#: compileCache.mode values; none is the default.
COMPILE_CACHE_MODES = ("none", "node", "shared")
#: modelStorage.accessMode values; ReadWriteOnce is the default.
ACCESS_MODES = ("ReadWriteOnce", "ReadWriteMany")
#: Why maxReplicas above 1 needs a target.
NO_TARGET = ("server.autoscaling.targetOngoingRequests must be set when maxReplicas is above 1: Ray Serve LLM's "
             "default target is 10^9 ongoing requests per replica, so the replica count would never grow")
MOVED = ("server.minReplicas and server.maxReplicas moved to server.autoscaling.minReplicas and "
         "server.autoscaling.maxReplicas")
#: Why replicas that share a GPU must each claim at most their share of its memory (inference-server/serve_app.py
#: refuses the same at start): vLLM refuses to start when the GPU's free memory is below gpu_memory_utilization
#: times its total memory.
SHARED_GPU_MEMORY = ("server.engineKwargs.gpu_memory_utilization ({memory}) must not exceed server.gpuPerReplica "
                     "({share}) when up to {high} replicas share a GPU: each vLLM replica claims that fraction of the "
                     "whole GPU's memory when it starts, so a later replica would not fit")


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _whole(value) -> bool:
    return _number(value) and float(value).is_integer()


def _kind(value) -> str:
    """The value's kind in Helm's words, for messages."""
    if value is None:
        return "invalid"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "float64"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "map"
    if isinstance(value, list):
        return "slice"
    return type(value).__name__


def _show(value) -> str:
    """The value the way Helm's %v prints it."""
    if value is None:
        return "<nil>"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def scale_out_mode(values: Mapping) -> object:
    """global.scaleOut.mode, or none when it is not set."""
    scale = (values.get("global") or {}).get("scaleOut") or {}
    return scale["mode"] if "mode" in scale else "none"


def check_global(values: Mapping) -> list[str]:
    """The two mode switches in global: global.scaleOut.mode and global.compare.mode."""
    out = []
    mode = scale_out_mode(values)
    if mode in SCALE_OUT_DESIGNED:
        out.append(KUBERAY_NOT_BUILT)
    elif not (isinstance(mode, str) and mode in SCALE_OUT_MODES):
        out.append(f"global.scaleOut.mode must be none or keda (kuberay is designed but not built yet); "
                   f"got {_show(mode)}")
    compare = (values.get("global") or {}).get("compare") or {}
    if "mode" in compare and not (isinstance(compare["mode"], str) and compare["mode"] in COMPARE_MODES):
        out.append(f"global.compare.mode must be pairs or scaleout; got {_show(compare['mode'])}")
    return out


def check_server(values: Mapping) -> list[str]:
    """The server chart's own settings: server.autoscaling, server.gpuPerReplica (and, when replicas share a GPU,
    server.engineKwargs.gpu_memory_utilization), server.metricsPort, compileCache, modelStorage.accessMode and, in
    keda mode, scaleOut."""
    out = []
    server = values.get("server") or {}
    if "minReplicas" in server or "maxReplicas" in server:
        out.append(MOVED)
    scaling = server.get("autoscaling") or {}
    low, high = scaling.get("minReplicas"), scaling.get("maxReplicas")
    if not (_whole(low) and low >= 1):
        out.append(f"server.autoscaling.minReplicas must be a whole number of at least 1; got {_show(low)} "
                   f"({_kind(low)})")
        return out
    if not (_whole(high) and high >= low):
        out.append(f"server.autoscaling.maxReplicas must be a whole number of at least minReplicas ({_show(low)}); "
                   f"got {_show(high)} ({_kind(high)})")
        return out
    target = scaling.get("targetOngoingRequests")
    if target is not None and not (_number(target) and target > 0):
        out.append(f"server.autoscaling.targetOngoingRequests must be a number above 0, or null; got {_show(target)} "
                   f"({_kind(target)})")
        return out
    if high > 1 and target is None:
        out.append(NO_TARGET)
        return out
    share = server.get("gpuPerReplica")
    if not (_number(share) and share > 0):
        out.append(f"server.gpuPerReplica must be a number above 0; got {_show(share)} ({_kind(share)})")
        return out
    gpus = server.get("gpuCount")
    if _number(gpus) and high * share > gpus + 1e-6:
        out.append(f"server.autoscaling.maxReplicas ({_show(high)}) times server.gpuPerReplica ({_show(share)}) is "
                   f"more than server.gpuCount ({_show(gpus)})")
        return out
    memory = (server.get("engineKwargs") or {}).get("gpu_memory_utilization")
    if share < 1 and high > 1 and _number(memory) and memory > share + 1e-9:
        out.append(SHARED_GPU_MEMORY.format(memory=_show(memory), share=_show(share), high=_show(high)))
        return out
    port = server.get("metricsPort")
    if not (_whole(port) and 1 <= port <= 65535 and port != server.get("httpPort")):
        out.append(f"server.metricsPort must be a port number (1 to 65535) other than server.httpPort; got "
                   f"{_show(port)} ({_kind(port)})")
        return out
    cache = (values.get("compileCache") or {}).get("mode")
    if not (isinstance(cache, str) and cache in COMPILE_CACHE_MODES):
        out.append(f"compileCache.mode must be none, node or shared; got {_show(cache)}")
        return out
    access = (values.get("modelStorage") or {}).get("accessMode")
    if not (isinstance(access, str) and access in ACCESS_MODES):
        out.append(f"modelStorage.accessMode must be ReadWriteOnce or ReadWriteMany; got {_show(access)}")
        return out
    if scale_out_mode(values) == "keda":
        pods = values.get("scaleOut") or {}
        low_pods, high_pods = pods.get("minPods"), pods.get("maxPods")
        if not (_whole(low_pods) and low_pods >= 1):
            out.append(f"scaleOut.minPods must be a whole number of at least 1; got {_show(low_pods)}")
        elif not (_whole(high_pods) and high_pods >= low_pods):
            out.append(f"scaleOut.maxPods must be a whole number of at least scaleOut.minPods ({_show(low_pods)}); "
                       f"got {_show(high_pods)}")
    return out


def check(values: Mapping) -> list[str]:
    """Every rule. The global switches are all reported; of the server chart's own settings only the first broken
    rule is, in the order its render checks them (a render stops at its first failure)."""
    return check_global(values) + check_server(values)
