import json
from pathlib import Path

import numpy as np
import pydantic
import pytest
import torch

from lase.utils.config import Config, dump_config, load_config
from lase.utils.paths import DATA_DIR_ENV, data_dir, repo_root
from lase.utils.provenance import create_run_dir
from lase.utils.seeding import seed_everything


class _Example(Config):
    name: str
    rate_hz: float = 5.0


def test_config_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("name: test\nrate_hz: 10\n")
    cfg = load_config(path, _Example)
    assert cfg == _Example(name="test", rate_hz=10.0)
    out = tmp_path / "out.yaml"
    dump_config(cfg, out)
    assert load_config(out, _Example) == cfg


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("name: test\nrate_hzz: 10\n")
    with pytest.raises(pydantic.ValidationError):
        load_config(path, _Example)


def test_config_is_frozen() -> None:
    cfg = _Example(name="x")
    with pytest.raises(pydantic.ValidationError):
        cfg.name = "y"  # type: ignore[misc]


def test_seed_everything_reproducible() -> None:
    seed_everything(123)
    a = (np.random.rand(3), torch.rand(3))  # noqa: NPY002
    seed_everything(123)
    b = (np.random.rand(3), torch.rand(3))  # noqa: NPY002
    np.testing.assert_array_equal(a[0], b[0])
    assert torch.equal(a[1], b[1])


def test_data_dir_default_and_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(DATA_DIR_ENV, raising=False)
    assert data_dir() == repo_root() / "local_data"
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    assert data_dir() == tmp_path
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path / "missing"))
    with pytest.raises(RuntimeError):
        data_dir()


def test_create_run_dir(tmp_path: Path) -> None:
    run = create_run_dir("unit", _Example(name="r"), seed=7, base_dir=tmp_path)
    assert (run / "config.yaml").is_file()
    record = json.loads((run / "provenance.json").read_text())
    assert record["seed"] == 7
    assert "commit" in record["git"]
