import datetime as dt
import io
import tarfile
from pathlib import Path

import pytest

from lase.data.download import (
    FileKind,
    FileManifest,
    FileRecord,
    extract_members,
    fetch,
    list_archive,
    remote_relpath,
    sha256_file,
)

D = dt.date(2013, 1, 10)


def test_remote_relpaths() -> None:
    assert remote_relpath(FileKind.SENSOR, D) == "sensor_data/2013-01-10_sen.tar.gz"
    assert remote_relpath(FileKind.GROUND_TRUTH, D) == "ground_truth/groundtruth_2013-01-10.csv"
    assert remote_relpath(FileKind.COVARIANCE, D) == "covariance/cov_2013-01-10.csv"


def _make_tar(path: Path) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in [("2013-01-10/ms25.csv", b"1,2,3\n"), ("2013-01-10/x.txt", b"x")]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        evil = tarfile.TarInfo("../../evil.csv")
        evil.size = 1
        tar.addfile(evil, io.BytesIO(b"e"))


def test_extract_selected_and_flat(tmp_path: Path) -> None:
    tar_path = tmp_path / "a.tar.gz"
    _make_tar(tar_path)
    assert ("2013-01-10/ms25.csv", 6) in list_archive(tar_path)
    out = extract_members(tar_path, tmp_path / "out", ["ms25.csv", "evil.csv"])
    names = sorted(p.name for p in out)
    assert names == ["evil.csv", "ms25.csv"]
    assert all(p.parent == tmp_path / "out" for p in out)  # nothing escapes dest_dir
    assert (tmp_path / "out" / "ms25.csv").read_bytes() == b"1,2,3\n"
    assert not (tmp_path / "out" / "x.txt").exists()


def test_manifest_roundtrip(tmp_path: Path) -> None:
    m = FileManifest(files={"a/b.csv": FileRecord(size=3, sha256="ab")})
    m.save(tmp_path / "m.yaml")
    assert FileManifest.load(tmp_path / "m.yaml") == m
    assert FileManifest.load(tmp_path / "missing.yaml") == FileManifest()


def test_fetch_records_then_verifies_existing_file(tmp_path: Path) -> None:
    rel = remote_relpath(FileKind.COVARIANCE, D)
    f = tmp_path / rel
    f.parent.mkdir(parents=True)
    f.write_bytes(b"data")
    m = FileManifest()
    assert fetch(FileKind.COVARIANCE, D, tmp_path, m)[1] == "recorded"
    assert m.files[rel].sha256 == sha256_file(f)
    assert fetch(FileKind.COVARIANCE, D, tmp_path, m)[1] == "verified"
    f.write_bytes(b"tampered")
    with pytest.raises(OSError, match="Checksum mismatch"):
        fetch(FileKind.COVARIANCE, D, tmp_path, m)
