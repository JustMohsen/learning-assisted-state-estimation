"""Recording-day split manifest (train / validation / test)."""

from __future__ import annotations

import datetime as dt
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import model_validator

from lase.utils.config import Config, load_config
from lase.utils.paths import repo_root

DEFAULT_SPLIT_PATH = Path("configs/data/split.yaml")


class Split(StrEnum):
    """Dataset partition."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class SplitManifest(Config):
    """Recording dates per partition. Partitions must be disjoint and duplicate-free."""

    name: str
    train: tuple[dt.date, ...]
    validation: tuple[dt.date, ...]
    test: tuple[dt.date, ...]

    @model_validator(mode="after")
    def _check_disjoint(self) -> Self:
        all_dates = self.train + self.validation + self.test
        dupes = sorted(d for d, n in Counter(all_dates).items() if n > 1)
        if dupes:
            raise ValueError(f"Dates appear more than once across or within splits: {dupes}")
        return self

    def dates(self, split: Split | None = None) -> tuple[dt.date, ...]:
        """Dates of one partition, or all dates (chronologically sorted) if ``split`` is None."""
        if split is None:
            return tuple(sorted(self.train + self.validation + self.test))
        return tuple(getattr(self, split.value))

    def split_of(self, date: dt.date) -> Split:
        """Partition containing ``date``.

        Raises:
            KeyError: If the date is not in the manifest (unknown recordings never join silently).
        """
        for split in Split:
            if date in self.dates(split):
                return split
        raise KeyError(f"{date} is not in split manifest '{self.name}'.")


def load_split(path: str | Path | None = None) -> SplitManifest:
    """Load the split manifest (default: ``configs/data/split.yaml``)."""
    return load_config(path or repo_root() / DEFAULT_SPLIT_PATH, SplitManifest)
