import importlib

import pytest

import lase

SUBPACKAGES = [
    "data",
    "geometry",
    "sim",
    "filter",
    "learning",
    "comparators",
    "training",
    "evaluation",
    "utils",
]


def test_version() -> None:
    assert lase.__version__


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackages_import(name: str) -> None:
    importlib.import_module(f"lase.{name}")
