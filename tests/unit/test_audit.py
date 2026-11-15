import numpy as np

from lase.data.audit import gap_spans, spans_where


def test_spans_where_min_duration() -> None:
    t = np.arange(0, 20_000_001, 1_000_000, dtype=np.int64)  # 0..20 s
    m = np.zeros(len(t), dtype=bool)
    m[2:9] = True  # 2..8 s (6 s)
    m[12:14] = True  # 12..13 s (1 s)
    assert spans_where(t, m, 5.0) == [(2_000_000, 8_000_000)]


def test_gap_spans() -> None:
    t = np.array([0, 1_000_000, 8_000_000, 9_000_000], dtype=np.int64)
    assert gap_spans(t, 5.0) == [(1_000_000, 8_000_000)]
