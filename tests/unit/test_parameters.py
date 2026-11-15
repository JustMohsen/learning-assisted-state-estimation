"""configs/parameters.yaml: complete, documented, loadable, overridable."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import BaseModel

from lase.filter.params import load_filter_config
from lase.parameters import DEFAULT_PARAMETERS_PATH, Parameters, deep_merge, load_parameters
from lase.utils.paths import repo_root

PATH = repo_root() / DEFAULT_PARAMETERS_PATH


def _model_keys(model: type[BaseModel], prefix: str = "") -> set[str]:
    keys = set()
    for name, field in model.model_fields.items():
        ann = field.annotation
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            keys |= _model_keys(ann, f"{prefix}{name}.")
        else:
            keys.add(f"{prefix}{name}")
    return keys


def _yaml_keys(d: dict[str, Any], prefix: str = "") -> set[str]:
    keys = set()
    for k, v in d.items():
        if isinstance(v, dict):
            keys |= _yaml_keys(v, f"{prefix}{k}.")
        else:
            keys.add(f"{prefix}{k}")
    return keys


def test_file_lists_every_parameter_explicitly() -> None:
    data = yaml.safe_load(PATH.read_text())
    missing = _model_keys(Parameters) - _yaml_keys(data)
    extra = _yaml_keys(data) - _model_keys(Parameters)
    assert not missing, f"parameters missing from {PATH.name}: {sorted(missing)}"
    assert not extra, f"unknown keys in {PATH.name}: {sorted(extra)}"


def test_every_value_line_is_commented() -> None:
    for i, line in enumerate(PATH.read_text().splitlines(), 1):
        body = line.split("#", 1)[0]
        if ":" in body and body.split(":", 1)[1].strip():  # a "key: value" line
            assert "#" in line or body.strip().startswith("name:"), (
                f"line {i} has no comment: {line!r}"
            )


def test_loads_and_matches_filter_config() -> None:
    p = load_parameters()
    assert isinstance(p, Parameters)
    assert load_filter_config("baseline") == p.filter
    assert p.filter.nonholonomic.enabled


def test_overrides_merge(tmp_path: Path) -> None:
    ov = tmp_path / "exp.yaml"
    ov.write_text("filter:\n  gnss:\n    sigma_m: 9.0\n")
    p = load_parameters(overrides=[ov])
    assert p.filter.gnss.sigma_m == 9.0
    assert p.filter.gnss.min_interval_s == load_parameters().filter.gnss.min_interval_s
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}


def test_unknown_override_key_rejected(tmp_path: Path) -> None:
    ov = tmp_path / "bad.yaml"
    ov.write_text("filter:\n  gnss:\n    sigma_mm: 9.0\n")
    with pytest.raises(Exception, match="sigma_mm"):
        load_parameters(overrides=[ov])
