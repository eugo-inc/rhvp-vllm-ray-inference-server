#!/usr/bin/env python3
"""Start an in-cluster image build when the pattern's git branch changed a build's folder.

Called by ansible/rebuild-on-commit.yaml, which the Validated Patterns imperative CronJob
runs every 10 minutes inside a shallow clone of the pattern repository, checked out at
the pattern's target revision. Uses only the Python standard library, `oc` and `git`.

For each BuildConfig it is given, it decides one of these, in order:

1. The BuildConfig does not exist (image.override is set, or Argo CD has not created it
   yet): do nothing.
2. It has no builds yet: do nothing. Its ConfigChange trigger starts the first build.
3. A build is New, Pending or Running: do nothing.
4. The newest successful build used the checked-out commit: do nothing.
5. The newest build did not succeed, and it used the checked-out commit, or a commit
   with the same build folder: do nothing. It would fail the same way; start it by hand
   with `oc start-build <name> -n <namespace>`.
6. The build folder (the BuildConfig's contextDir) is the same in the checked-out commit
   and in the newest successful build's commit: do nothing.
7. Otherwise: `oc start-build <name> -n <namespace> --commit=<checked-out commit>`, and
   return without waiting for the build.

When an older commit cannot be fetched to compare folders, the folder counts as changed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from typing import Callable, Optional

ACTIVE_PHASES = {"New", "Pending", "Running"}


def run(cmd: list[str], cwd: Optional[str] = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)


def build_commit(build: dict) -> str:
    """Git commit a build used: from its status, else the one it was started with."""
    for part in (build.get("status") or {}, build.get("spec") or {}):
        commit = (((part.get("revision") or {}).get("git") or {}).get("commit")) or ""
        if commit:
            return commit
    return ""


def build_phase(build: dict) -> str:
    return (build.get("status") or {}).get("phase") or "New"


def parse_time(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None


def decide(
    builds: list[dict],
    head: str,
    head_time: Optional[datetime],
    folder_changed: Callable[[str], Optional[bool]],
) -> tuple[str, str]:
    """Return ("start" | "skip", reason).

    folder_changed(commit) answers whether the build folder differs between `commit` and
    the checked-out commit: True, False, or None when it cannot tell.
    """
    if not builds:
        return "skip", "no builds yet; the ConfigChange trigger starts the first one"

    builds = sorted(builds, key=lambda b: ((b.get("metadata") or {}).get("creationTimestamp") or "",
                                           (b.get("metadata") or {}).get("name") or ""))
    for b in builds:
        if build_phase(b) in ACTIVE_PHASES:
            return "skip", f"build {b['metadata']['name']} is {build_phase(b)}"

    complete = [b for b in builds if build_phase(b) == "Complete"]
    last_ok_commit = build_commit(complete[-1]) if complete else ""
    if last_ok_commit == head:
        return "skip", f"up to date: build {complete[-1]['metadata']['name']} used {head[:12]}"

    latest = builds[-1]
    if build_phase(latest) != "Complete":
        failed_commit = build_commit(latest)
        name = latest["metadata"]["name"]
        if failed_commit == head:
            return "skip", f"build {name} of {head[:12]} is {build_phase(latest)}; not retried automatically"
        if failed_commit and folder_changed(failed_commit) is False:
            return "skip", (f"build {name} of {failed_commit[:12]} is {build_phase(latest)} and the build "
                            f"folder is unchanged since; not retried automatically")
        if not failed_commit:
            created = parse_time((latest.get("metadata") or {}).get("creationTimestamp") or "")
            if head_time is None or created is None or head_time <= created:
                return "skip", (f"build {name} is {build_phase(latest)} and the checked-out commit is "
                                f"not newer than it; not retried automatically")

    if last_ok_commit and folder_changed(last_ok_commit) is False:
        return "skip", f"build folder unchanged since the last successful build ({last_ok_commit[:12]})"

    return "start", f"new commit {head[:12]} (last successful build: {last_ok_commit[:12] or 'none'})"


def make_folder_changed(repo: str, context_dir: str) -> Callable[[str], Optional[bool]]:
    def folder_changed(commit: str) -> Optional[bool]:
        have = run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=repo, check=False)
        if have.returncode != 0:
            fetched = run(["git", "fetch", "--quiet", "--depth=1", "origin", commit], cwd=repo, check=False)
            if fetched.returncode != 0:
                return None
        diff = run(["git", "diff", "--quiet", commit, "HEAD", "--", context_dir], cwd=repo, check=False)
        if diff.returncode == 0:
            return False
        if diff.returncode == 1:
            return True
        return None

    return folder_changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="git checkout of the pattern repository")
    parser.add_argument("--targets", required=True,
                        help='JSON list of {"namespace", "buildconfig", "context_dir"}')
    parser.add_argument("--oc", default="oc")
    args = parser.parse_args()

    head = run(["git", "rev-parse", "HEAD"], cwd=args.repo).stdout.strip()
    head_time = parse_time(run(["git", "log", "-1", "--format=%cI", "HEAD"], cwd=args.repo).stdout.strip())

    for target in json.loads(args.targets):
        ns, bc, context_dir = target["namespace"], target["buildconfig"], target["context_dir"]
        label = f"{ns}/{bc}"
        try:
            check_target(args.oc, args.repo, ns, bc, context_dir, head, head_time, label)
        except subprocess.CalledProcessError as err:
            # One target's error (a Forbidden start-build when rebuildOnCommit is off in that chart, a transient
            # API error) must not stop the other targets.
            print(f"ERROR {label}: command failed ({err.returncode}): {' '.join(err.cmd)}: "
                  f"{(err.stderr or '').strip()}")
    # Always 0: the imperative CronJob runs its jobs one after another and stops at the first failing one, so a
    # failure here would also stop the jobs after it. Errors are in this job's log as ERROR lines.
    return 0


def check_target(oc: str, repo: str, ns: str, bc: str, context_dir: str, head: str,
                 head_time: Optional[datetime], label: str) -> None:
    found = run([oc, "get", "buildconfig", bc, "-n", ns, "--ignore-not-found", "-o", "name"])
    if not found.stdout.strip():
        print(f"SKIP {label}: no such BuildConfig")
        return
    listed = run([oc, "get", "builds", "-n", ns, "-l", f"openshift.io/build-config.name={bc}", "-o", "json"])
    builds = json.loads(listed.stdout or "{}").get("items") or []
    action, reason = decide(builds, head, head_time, make_folder_changed(repo, context_dir))
    if action == "start":
        started = run([oc, "start-build", bc, "-n", ns, f"--commit={head}", "-o", "name"])
        print(f"STARTED {label}: {started.stdout.strip()} - {reason}")
    else:
        print(f"SKIP {label}: {reason}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as err:
        # Only the git commands before the first target get here. Exit 0 for the same reason as in main().
        sys.stderr.write(f"ERROR command failed ({err.returncode}): {' '.join(err.cmd)}\n{err.stderr}\n")
        sys.exit(0)
