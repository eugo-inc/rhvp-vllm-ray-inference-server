"""values-secret.yaml.template: the secrets schema and the secrets loader's own rule.

The JSON schema accepts `value: null`, but the loader in rhvp.cluster_utils
(plugins/module_utils/load_secrets_common.py, _validate_error_mode) rejects a field whose
onMissingValue is "error" (the default) when value, path and ini_file are all empty, and
`make install` falls back to this template when the user has no secrets file. So both
checks run here, each with a planted case that must fail.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import jsonschema
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "values-secret.yaml.template"
SCHEMA_URL = (
    "https://raw.githubusercontent.com/validatedpatterns/rhvp.cluster_utils/"
    "3f4536cc72ac728e676bd6331e6703b4f3ad79cc/roles/vault_utils/values-secrets.v2.schema.json"
)
SCHEMA_SHA256 = "736f22acb7102513e2fb561e33fc48f259d51c8168dac0fefd7c617cfbe96be9"


def loader_problems(doc: dict) -> list[str]:
    """The field rules the secrets loader applies before it writes anything to Vault,
    plus one of our own: no prompts, because the install must run unattended."""
    policies = set((doc.get("vaultPolicies") or {}).keys()) | {"validatedPatternDefaultPolicy"}
    problems = []
    for secret in (doc.get("secrets") or []) + (doc.get("bootstrap_secrets") or []):
        for field in secret.get("fields") or []:
            where = f"{secret.get('name')}.{field.get('name')}"
            kinds = [k for k in ("value", "path", "ini_file") if k in field]
            mode = field.get("onMissingValue", "error")
            if len(kinds) > 1:
                problems.append(f"{where}: sets both {kinds[0]} and {kinds[1]}")
            elif mode == "error":
                if not any(field.get(k) not in (None, "") for k in ("value", "path", "ini_file")):
                    problems.append(f"{where}: error mode with no value, path or ini_file")
            elif mode == "generate":
                if field.get("value") not in (None, "") or field.get("path") not in (None, ""):
                    problems.append(f"{where}: generate mode with a value or path")
                elif field.get("vaultPolicy") not in policies:
                    problems.append(f"{where}: generate mode without a known vaultPolicy")
            elif mode == "prompt":
                problems.append(f"{where}: prompt mode stops an unattended install")
            else:
                problems.append(f"{where}: invalid onMissingValue {mode!r}")
    return problems


def default_filling_validator(schema: dict):
    """Fill schema defaults before validating, like `check-jsonschema --fill-defaults`,
    which is how rhvp.cluster_utils checks its own examples against this schema."""
    base = jsonschema.Draft7Validator
    validate_properties = base.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, sub_schema):
        if isinstance(instance, dict):
            for prop, sub in properties.items():
                if isinstance(sub, dict) and "default" in sub:
                    instance.setdefault(prop, sub["default"])
        yield from validate_properties(validator, properties, instance, sub_schema)

    return jsonschema.validators.extend(base, {"properties": set_defaults})(schema)


@pytest.fixture(scope="module")
def template() -> dict:
    return yaml.safe_load(TEMPLATE.read_text())


@pytest.fixture(scope="module")
def schema() -> dict:
    try:
        with urllib.request.urlopen(SCHEMA_URL, timeout=30) as resp:
            data = resp.read()
    except (urllib.error.URLError, TimeoutError) as err:
        pytest.skip(f"cannot download the secrets schema: {err}")
    assert hashlib.sha256(data).hexdigest() == SCHEMA_SHA256, "pinned secrets schema changed"
    return json.loads(data)


def test_template_matches_schema(template, schema):
    errors = list(default_filling_validator(schema).iter_errors(copy.deepcopy(template)))
    assert not errors, [e.message for e in errors]


def test_template_passes_loader_rule(template):
    assert loader_problems(template) == []


def test_hf_token_entry_matches_the_chart(template):
    """hfmodel/hftoken with no vaultPrefixes lands at secret/data/hub/hfmodel, the key the
    ray-serve-llm ExternalSecret reads."""
    [secret] = [s for s in template["secrets"] if s["name"] == "hfmodel"]
    assert "vaultPrefixes" not in secret
    [field] = secret["fields"]
    assert field["name"] == "hftoken"
    assert field["value"] == "none"
    chart_values = yaml.safe_load((REPO / "charts/all/ray-serve-llm/values.yaml").read_text())
    assert chart_values["hfmodel"]["key"] == "secret/data/hub/hfmodel"
    assert chart_values["hfmodel"]["property"] == "hftoken"


def test_planted_null_value_passes_schema_but_fails_loader_rule(template, schema):
    planted = copy.deepcopy(template)
    planted["secrets"][0]["fields"][0]["value"] = None
    assert not list(default_filling_validator(schema).iter_errors(copy.deepcopy(planted)))
    assert loader_problems(planted)


@pytest.mark.parametrize(
    "field",
    [
        {"name": "x", "value": ""},
        {"name": "x"},
        {"name": "x", "onMissingValue": "generate"},
        {"name": "x", "onMissingValue": "generate", "vaultPolicy": "validatedPatternDefaultPolicy", "value": "v"},
        {"name": "x", "onMissingValue": "prompt", "value": None},
        {"name": "x", "value": "v", "path": "/tmp/f"},
    ],
)
def test_planted_bad_fields_fail_loader_rule(field):
    assert loader_problems({"version": "2.0", "secrets": [{"name": "s", "fields": [field]}]})


def test_validate_values_script_passes():
    """The same checks as CI (.github/workflows/jsonschema.yaml), end to end."""
    proc = subprocess.run(
        [str(REPO / "scripts/validate-values.sh")],
        env={**os.environ, "PYTHON": sys.executable},
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0 and "urlopen error" in proc.stderr:
        pytest.skip("validate-values.sh could not download its schemas")
    assert proc.returncode == 0, proc.stdout + proc.stderr
