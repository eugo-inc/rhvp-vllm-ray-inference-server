#!/usr/bin/env bash
# Schema-check this pattern's values files without a cluster, helm or containers.
#
# 1. Merges the values files the way the cluster does (values-global.yaml, then
#    variants/<variant>/values-<variant>.yaml, then every clusterGroup.sharedValueFiles
#    entry that exists for the case) for three cases: AWS with global.gpuMachineSet from
#    values-global.yaml, AWS with gpuMachineSet none, and GCP. Each merged tree is checked
#    against the clustergroup chart's values.schema.json. Maps merge; lists and plain
#    values in a later file replace earlier ones. Every file under overrides/ must be
#    used by at least one case, so a misspelt override name fails here.
# 2. Checks values-secret.yaml.template against rhvp.cluster_utils'
#    values-secrets.v2.schema.json, and against the secrets loader's own rule: a field
#    with onMissingValue "error" (the default) needs a non-empty value, path or ini_file;
#    "generate" needs a vaultPolicy and no value or path; "prompt" would stop an
#    unattended install and is refused here.
# 3. Checks pattern-metadata.yaml against the Validated Patterns UI catalog schema,
#    after the catalog's normalization (platform wrapper removed, owners as a list).
#
# Every schema is downloaded from a pinned commit and checked against its sha256.
# Needs python3 with PyYAML and jsonschema (PYTHON=... picks the interpreter), and
# network access to raw.githubusercontent.com.
# render-charts.sh complements this: helm checks the same schema there after merging
# the chart's own defaults.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
cd "${REPO_ROOT}"

exec "${PYTHON}" -I - "${REPO_ROOT}" <<'PY'
import copy
import hashlib
import json
import os
import re
import sys
import urllib.request

import jsonschema
import yaml

REPO = sys.argv[1]
SCHEMAS = {
    # clustergroup chart 0.9.58 (validatedpatterns/clustergroup-chart)
    "clustergroup": (
        "https://raw.githubusercontent.com/validatedpatterns/clustergroup-chart/"
        "c60888f5806972e781fbf987a4acdeddbd464062/values.schema.json",
        "f18a8d938f70f3e78603b2ac8f34c9baf544b7828bba1d73056bc89c2dded2bc",
    ),
    "secrets": (
        "https://raw.githubusercontent.com/validatedpatterns/rhvp.cluster_utils/"
        "3f4536cc72ac728e676bd6331e6703b4f3ad79cc/roles/vault_utils/values-secrets.v2.schema.json",
        "736f22acb7102513e2fb561e33fc48f259d51c8168dac0fefd7c617cfbe96be9",
    ),
    "metadata": (
        "https://raw.githubusercontent.com/validatedpatterns/pattern-ui-catalog/"
        "bbe8244d55a286feffe1212a3c52b2fd07dc06a6/pattern.schema.json",
        "0fae92bef1349bb67adab89a6b06e4cb0caa9b220ca9c07a6962cecc4c356700",
    ),
}
failures = []


def fetch(name):
    url, want = SCHEMAS[name]
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = resp.read()
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        sys.exit(f"validate-values.sh: {url} has sha256 {got}, expected {want}")
    return json.loads(data)


def load(rel):
    with open(os.path.join(REPO, rel)) as fh:
        return yaml.safe_load(fh) or {}


def merge(base, over):
    if isinstance(base, dict) and isinstance(over, dict):
        out = dict(base)
        for key, value in over.items():
            out[key] = merge(base[key], value) if key in base else value
        return out
    return over


def lookup(tree, dotted):
    node = tree
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


TPL = re.compile(r"\{\{-?\s*\$?\.Values\.([A-Za-z0-9_.]+)\s*-?\}\}")


def render_path(template, values):
    def sub(match):
        value = lookup(values, match.group(1))
        if value is None:
            return ""
        return str(value)

    out = TPL.sub(sub, template)
    if "{{" in out:
        sys.exit(f"validate-values.sh: cannot template sharedValueFiles entry {template!r}")
    return out


def filling_defaults(base):
    """A validator class that first fills in schema defaults, like check-jsonschema
    --fill-defaults (the way rhvp.cluster_utils checks its own secrets examples)."""
    validate_properties = base.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        if isinstance(instance, dict):
            for prop, sub in properties.items():
                if isinstance(sub, dict) and "default" in sub:
                    instance.setdefault(prop, sub["default"])
        yield from validate_properties(validator, properties, instance, schema)

    return jsonschema.validators.extend(base, {"properties": set_defaults})


def check(schema, instance, what, fill_defaults=False):
    cls = filling_defaults(jsonschema.Draft7Validator) if fill_defaults else jsonschema.Draft7Validator
    validator = cls(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path))
    for err in errors:
        where = "/".join(str(p) for p in err.absolute_path) or "(top)"
        failures.append(f"{what}: {where}: {err.message}")
    if not errors:
        print(f"ok   {what}")


# ---- 1. clustergroup values ---------------------------------------------------------
cg_schema = fetch("clustergroup")
glob = load("values-global.yaml")
variant = glob["main"]["variant"]
variant_file = f"variants/{variant}/values-{variant}.yaml"
used_overrides = set()
cases = [
    ("AWS", {}),
    ("AWS", {"gpuMachineSet": "none"}),
    ("GCP", {}),
]
for platform, extra_global in cases:
    injected = {"global": {"clusterPlatform": platform, "clusterVersion": "4.20",
                           "localClusterName": "local-cluster", **extra_global}}
    files = ["values-global.yaml", variant_file]
    # Files the clustergroup chart adds itself under the variants layout (all optional).
    for name in (f"values-{platform}.yaml", f"values-{platform}-4.20.yaml",
                 f"values-{platform}-{variant}.yaml", f"values-4.20-{variant}.yaml",
                 "values-local-cluster.yaml"):
        files.append(f"variants/{variant}/{name}")
    base = merge(merge(load("values-global.yaml"), load(variant_file)), injected)
    for entry in (base.get("clusterGroup") or {}).get("sharedValueFiles") or []:
        files.append(render_path(entry, base).lstrip("/"))
    present = [f for f in files if os.path.isfile(os.path.join(REPO, f))]
    merged = {}
    for f in present:
        merged = merge(merged, load(f))
        if f.startswith("overrides/"):
            used_overrides.add(f)
    merged = merge(merged, injected)
    label = f"clustergroup values, {platform}" + "".join(f", {k}={v}" for k, v in extra_global.items())
    check(cg_schema, merged, f"{label} [{' + '.join(present)}]")

for root, _dirs, names in os.walk(os.path.join(REPO, "overrides")):
    for name in names:
        rel = os.path.relpath(os.path.join(root, name), REPO)
        if rel not in used_overrides:
            failures.append(f"{rel}: no sharedValueFiles entry includes it in any checked case")

# ---- 2. secrets template --------------------------------------------------------------
secrets = load("values-secret.yaml.template")
check(fetch("secrets"), copy.deepcopy(secrets), "values-secret.yaml.template (values-secrets.v2 schema)",
      fill_defaults=True)
policies = set((secrets.get("vaultPolicies") or {}).keys()) | {"validatedPatternDefaultPolicy"}
rule_ok = True
for secret in (secrets.get("secrets") or []) + (secrets.get("bootstrap_secrets") or []):
    for field in secret.get("fields") or []:
        where = f"values-secret.yaml.template: {secret.get('name')}.{field.get('name')}"
        kinds = [k for k in ("value", "path", "ini_file") if k in field]
        mode = field.get("onMissingValue", "error")
        problem = None
        if len(kinds) > 1:
            problem = f"sets both {kinds[0]} and {kinds[1]}"
        elif mode == "error":
            if not any(field.get(k) not in (None, "") for k in ("value", "path", "ini_file")):
                problem = "onMissingValue is error (the default) but value, path and ini_file are all empty"
        elif mode == "generate":
            if field.get("value") not in (None, "") or field.get("path") not in (None, ""):
                problem = "onMissingValue is generate but a value or path is set"
            elif field.get("vaultPolicy") not in policies:
                problem = "onMissingValue is generate but vaultPolicy is missing or undefined"
        elif mode == "prompt":
            problem = "onMissingValue prompt stops an unattended install"
        else:
            problem = f"onMissingValue {mode!r} is invalid"
        if problem:
            rule_ok = False
            failures.append(f"{where}: {problem}")
if rule_ok:
    print("ok   values-secret.yaml.template (secrets loader non-empty rule)")

# ---- 3. pattern metadata ---------------------------------------------------------------
meta = load("pattern-metadata.yaml")
catalog = dict(meta)
if isinstance(catalog.get("owners"), str):
    catalog["owners"] = [o.strip() for o in catalog["owners"].split(",") if o.strip()]
catalog["org"] = "eugo-inc"
reqs = {}
for role, pools in (meta.get("requirements") or {}).items():
    reqs[role] = {pool: (spec or {}).get("platform", spec) for pool, spec in (pools or {}).items()}
catalog["requirements"] = reqs
check(fetch("metadata"), catalog, "pattern-metadata.yaml (UI catalog schema, normalized)")
if meta.get("name") != glob["global"]["pattern"]:
    failures.append("pattern-metadata.yaml: name differs from global.pattern in values-global.yaml")

if failures:
    print("\nvalidate-values.sh: FAILED", file=sys.stderr)
    for line in failures:
        print(f"  {line}", file=sys.stderr)
    sys.exit(1)
print("validate-values.sh: OK")
PY
