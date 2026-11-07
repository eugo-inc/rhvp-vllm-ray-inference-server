# Copyright (c) 2026, Eugo. Licensed under the BSD 3-Clause License (see LICENSES/BSD-3-Clause.txt).
"""Print the Ray Serve status of the LLM app; exit 0 only when it is RUNNING.

Run it inside the server container, where the Ray cluster started by
serve_app.py is visible, for example::

    oc exec -n rhvp-inference deploy/ray-serve-llm -c server -- python rhvp_status.py

It attaches with ``ray.init(address="auto")``, which finds the Ray processes
running in this container, and prints one word:

- ``RUNNING``: the app is serving (the model is loaded). Exit code 0.
- ``DEPLOY_FAILED``: the app failed to start. Exit code 2.
- ``NO_CONTROLLER``: no Ray cluster or no Ray Serve is running. Exit code 1.
- ``NOT_STARTED``: Serve is running but the app is not deployed. Exit code 1.
- any other Ray Serve app status (``DEPLOYING``, ``UNHEALTHY``,
  ``DELETING``). Exit code 1.

With ``--wait SECONDS`` it checks every 2 seconds until the app is RUNNING or
DEPLOY_FAILED, or the time is up, and prints the last state.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from serve_app import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    ConfigError,
    app_status_name,
    load_config,
)

POLL_INTERVAL_S = 2.0
FINAL_STATES = ("RUNNING", "DEPLOY_FAILED")
EXIT_CODES = {"RUNNING": 0, "DEPLOY_FAILED": 2}


def default_app_name() -> str:
    """The app name from the server's config file, or ``llm``."""
    path = os.environ.get("RHVP_CONFIG") or DEFAULT_CONFIG_PATH
    logging.getLogger("rhvp.serve").setLevel(logging.ERROR)
    try:
        return load_config(path, os.environ)["app_name"]
    except (ConfigError, OSError):
        return "llm"


def exit_code(state: str) -> int:
    return EXIT_CODES.get(state, 1)


def check_once(app_name: str, address: str) -> str:
    import ray
    from ray import serve

    if not ray.is_initialized():
        try:
            ray.init(address=address, log_to_driver=False, logging_level=logging.ERROR)
        except ConnectionError:
            return "NO_CONTROLLER"
    return app_status_name(serve.status(), app_name)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--app", default=None, help="Serve app name (default: app_name from $RHVP_CONFIG, else llm)")
    parser.add_argument("--address", default="auto", help="Ray address (default: %(default)s)")
    parser.add_argument(
        "--wait",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="keep checking until RUNNING or DEPLOY_FAILED, for at most this long",
    )
    args = parser.parse_args(argv)
    app_name = args.app or default_app_name()

    deadline = time.monotonic() + max(args.wait, 0.0)
    try:
        while True:
            try:
                state = check_once(app_name, args.address)
            except Exception as exc:  # noqa: BLE001 - report, do not crash the probe
                print(f"status check failed: {exc}", file=sys.stderr)
                state = "NO_CONTROLLER"
            if state in FINAL_STATES or time.monotonic() >= deadline:
                break
            time.sleep(POLL_INTERVAL_S)
    finally:
        try:
            import ray

            if ray.is_initialized():
                ray.shutdown()
        except Exception:  # noqa: BLE001
            pass

    print(state)
    return exit_code(state)


if __name__ == "__main__":
    sys.exit(main())
