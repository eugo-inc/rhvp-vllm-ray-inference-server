# SPDX-License-Identifier: Apache-2.0
"""The ``rhvp-scale`` log line: the one definition of its shape.

The server writes one such line each time a replica count of its Ray Serve app changes. The image comparison reads
them back (``scale_events`` in ``notebooks/rhvp_compare/logs.py``, a separate image, so it keeps its own copy of the
field names); ``tests/test_scaleout_contract.py`` keeps that parser in step with this module.

The line is one JSON object, written without spaces, so it always contains ``MARKER``. Log prefixes before it on the
same line (a timestamp, a level, a logger name) are allowed::

    {"event":"rhvp-scale","ts":1794551388.123,"app":"llm","deployment":"LLMServer:qwen2_5-0_5b-instruct","from":1,"to":2,"reason":"target"}

Fields, always all of them, in this order:

- ``event``: ``"rhvp-scale"``.
- ``ts``: when the server saw the change, in seconds since the epoch by the server's own clock, to the millisecond.
- ``app``: the Ray Serve application (``app_name`` in ``llm.yaml``).
- ``deployment``: the Ray Serve deployment inside that app whose count changed.
- ``from``, ``to``: the count before and after, whole numbers of 0 or more.
- ``reason``: which count changed:

  - ``"target"``: the number of replicas Ray Serve aims for, that is the autoscaler's decision. Where Ray does not
    report a target, the running plus the starting replicas.
  - ``"running"``: the number of replicas in the RUNNING state.

Standard library only.
"""

from __future__ import annotations

import json
import math
import time
from typing import Optional

EVENT = "rhvp-scale"
FIELDS = ("event", "ts", "app", "deployment", "from", "to", "reason")
REASONS = ("target", "running")
MARKER = '{"event":"rhvp-scale"'


def scale_event(
    app: str, deployment: str, before: int, after: int, reason: str, ts: Optional[float] = None
) -> str:
    """The ``rhvp-scale`` line for one change of a count from ``before`` to ``after``.

    ``ts`` defaults to now. Raises ``ValueError`` for a value outside the shape above.
    """
    if not isinstance(app, str) or not app.strip():
        raise ValueError(f"app must be a non-empty string, got {app!r}")
    if not isinstance(deployment, str) or not deployment.strip():
        raise ValueError(f"deployment must be a non-empty string, got {deployment!r}")
    for name, count in (("before", before), ("after", after)):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{name} must be a whole number of 0 or more, got {count!r}")
    if reason not in REASONS:
        raise ValueError(f"reason must be one of {', '.join(REASONS)}, got {reason!r}")
    when = time.time() if ts is None else ts
    if isinstance(when, bool) or not isinstance(when, (int, float)) or not math.isfinite(when) or when < 0:
        raise ValueError(f"ts must be seconds since the epoch, got {ts!r}")
    record = {"event": EVENT, "ts": round(float(when), 3), "app": app, "deployment": deployment,
              "from": before, "to": after, "reason": reason}
    return json.dumps(record, separators=(",", ":"), ensure_ascii=True)
