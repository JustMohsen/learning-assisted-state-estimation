import datetime as dt

import pydantic
import pytest

from lase.data.splits import Split, SplitManifest, load_split


def test_default_manifest_counts() -> None:
    m = load_split()
    assert len(m.train) == 22
    assert len(m.validation) == 2
    assert len(m.test) == 3
    assert len(m.dates()) == 27
    assert list(m.dates()) == sorted(m.dates())


def test_split_of() -> None:
    m = load_split()
    assert m.split_of(dt.date(2013, 1, 10)) is Split.VALIDATION
    assert m.split_of(dt.date(2012, 11, 4)) is Split.TEST
    assert m.split_of(dt.date(2012, 1, 8)) is Split.TRAIN
    with pytest.raises(KeyError):
        m.split_of(dt.date(2000, 1, 1))


def test_overlap_rejected() -> None:
    d = dt.date(2012, 1, 8)
    with pytest.raises(pydantic.ValidationError):
        SplitManifest(name="bad", train=(d,), validation=(d,), test=())
