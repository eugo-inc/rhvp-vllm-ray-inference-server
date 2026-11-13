# SPDX-License-Identifier: Apache-2.0
"""What each start of a side's server was launched with: the record a round keeps in `launches`
(results.schema.json, `$defs/launch`).

A side may set its own command, which bypasses the image's entrypoint and every setting that entrypoint makes, and
its own extra environment (`sides[].env` in the config). So for every server start the results keep what the runner
asked for (command and args), whether the image's own entrypoint ran, and the environment the server process saw,
for the names that change how a server runs:

- GLIBC_TUNABLES, and every name that starts with CUDA_, VLLM_, TORCH_ or HF_;
- minus any name that looks like a secret (TOKEN, SECRET, PASSWORD, CREDENTIAL or KEY as a part of the name, such as
  HF_TOKEN or VLLM_API_KEY). Such a name is listed in `env_dropped`; its value is never recorded.

Every other name is left out of the record. Standard library only.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

#: Names kept as they are.
ENV_ALLOW_NAMES = ("GLIBC_TUNABLES",)
#: Prefixes whose names are kept.
ENV_ALLOW_PREFIXES = ("CUDA_", "VLLM_", "TORCH_", "HF_")
#: A name with one of these parts is never recorded, whatever its prefix.
ENV_SECRET = re.compile(r"(?:^|_)(?:[A-Z0-9]*TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?|KEY|APIKEY)(?:_|$)",
                        re.IGNORECASE)
#: Where `env` came from: the server process itself (for example /proc/<pid>/environ in its container), the
#: container's configured environment (the image's ENV plus the runner's), or nowhere (not recorded).
ENV_SOURCES = ("process", "container", "none")
#: Whose entrypoint ran: the image's own, or the side's command in its place.
ENTRYPOINTS = ("image", "override")


def allowed(name: str) -> bool:
    """True when `name` is on the allow list (before the secret rule)."""
    return name in ENV_ALLOW_NAMES or name.startswith(ENV_ALLOW_PREFIXES)


def recorded_env(env: Mapping[str, str]) -> tuple[dict[str, str], list[str]]:
    """The part of `env` a launch record keeps, and the allowed names left out as secrets (sorted)."""
    kept: dict[str, str] = {}
    dropped: list[str] = []
    for name, value in env.items():
        if not allowed(name):
            continue
        if ENV_SECRET.search(name):
            dropped.append(name)
            continue
        kept[name] = "" if value is None else str(value)
    return dict(sorted(kept.items())), sorted(dropped)


def record(replica: int, command: Sequence[str] | None, args: Sequence[str], env: Mapping[str, str] | None,
           env_source: str, started_at: str | None = None, image_entrypoint: Sequence[str] | None = None,
           image_cmd: Sequence[str] | None = None) -> dict:
    """One launch record. `replica` is 1 for a round's first server start and counts up for each replica a
    scale-out round adds. `command` is what the runner set in place of the image's entrypoint (None: the image's own
    ran). `env` is the environment the server process saw, as read from `env_source`; None records nothing."""
    if isinstance(replica, bool) or not isinstance(replica, int) or replica < 1:
        raise ValueError(f"replica must be a whole number of at least 1, got {replica!r}")
    if env_source not in ENV_SOURCES:
        raise ValueError(f"env_source must be one of {', '.join(ENV_SOURCES)}, got {env_source!r}")
    if env is None or env_source == "none":
        kept, dropped, env_source = {}, [], "none"
    else:
        kept, dropped = recorded_env(env)
    return {
        "replica": replica,
        "started_at": started_at,
        "entrypoint": "override" if command else "image",
        "command": list(command) if command else None,
        "args": list(args),
        "image_entrypoint": list(image_entrypoint) if image_entrypoint is not None else None,
        "image_cmd": list(image_cmd) if image_cmd is not None else None,
        "env": kept,
        "env_source": env_source,
        "env_dropped": dropped,
    }
