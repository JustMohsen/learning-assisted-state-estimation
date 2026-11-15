"""Download, verify and extract NCLT files.

NCLT publishes no checksums, so integrity uses trust-on-first-use: the SHA-256 and
size of each file are recorded in ``configs/data/nclt_files.yaml`` on first download
and verified on every later run. Raw downloads are never modified.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import shutil
import tarfile
import urllib.request
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path, PurePosixPath

import yaml
from pydantic import BaseModel, ConfigDict

from lase.utils.paths import repo_root

BASE_URL = "https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu"
DEFAULT_MANIFEST_PATH = Path("configs/data/nclt_files.yaml")
_CHUNK = 1 << 20


class FileKind(StrEnum):
    """Downloadable NCLT file types used by this project."""

    SENSOR = "sensor_data"
    GROUND_TRUTH = "ground_truth"
    COVARIANCE = "covariance"


def remote_relpath(kind: FileKind, date: dt.date) -> str:
    """Path of a file relative to :data:`BASE_URL` (also its path under ``raw/``)."""
    d = date.isoformat()
    names = {
        FileKind.SENSOR: f"sensor_data/{d}_sen.tar.gz",
        FileKind.GROUND_TRUTH: f"ground_truth/groundtruth_{d}.csv",
        FileKind.COVARIANCE: f"covariance/cov_{d}.csv",
    }
    return names[kind]


class FileRecord(BaseModel):
    """Recorded size and checksum of one downloaded file."""

    model_config = ConfigDict(extra="forbid")

    size: int
    sha256: str


class FileManifest(BaseModel):
    """Checksums of downloaded files, keyed by relative path (mutable; written by the downloader)."""

    model_config = ConfigDict(extra="forbid")

    base_url: str = BASE_URL
    files: dict[str, FileRecord] = {}

    @classmethod
    def load(cls, path: Path | None = None) -> FileManifest:
        """Load the manifest, or return an empty one if it does not exist yet."""
        path = path or repo_root() / DEFAULT_MANIFEST_PATH
        if not path.exists():
            return cls()
        with path.open(encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f) or {})

    def save(self, path: Path | None = None) -> None:
        """Write the manifest with files sorted by path."""
        path = path or repo_root() / DEFAULT_MANIFEST_PATH
        data = {
            "base_url": self.base_url,
            "files": {k: self.files[k].model_dump() for k in sorted(self.files)},
        }
        header = (
            "# SHA-256 and size of each NCLT file, recorded on first download (trust on first use)\n"
            "# and verified on later runs. Written by scripts/download_nclt.py.\n"
        )
        with path.open("w", encoding="utf-8") as f:
            f.write(header)
            yaml.safe_dump(data, f, sort_keys=False)


def sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def remote_size(url: str, timeout: float = 60.0) -> int:
    """Content length of a remote file (HTTP HEAD)."""
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return int(resp.headers["Content-Length"])


def download_file(url: str, dest: Path, timeout: float = 60.0) -> Path:
    """Download ``url`` to ``dest``, resuming a partial ``<dest>.part`` if present.

    The file is written to ``<dest>.part`` and renamed only once its size matches
    the server's content length.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = remote_size(url, timeout)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > total:
        part.unlink()
        have = 0
    if have < total:
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if have and resp.status != 206:  # server ignored the range: restart
                have = 0
            with part.open("ab" if have else "wb") as out:
                shutil.copyfileobj(resp, out, _CHUNK)
    if part.stat().st_size != total:
        raise OSError(f"Incomplete download of {url}: {part.stat().st_size} of {total} bytes.")
    part.replace(dest)
    return dest


def fetch(
    kind: FileKind,
    date: dt.date,
    raw_dir: Path,
    manifest: FileManifest,
) -> tuple[Path, str]:
    """Ensure one file is present and verified; download it if needed.

    Returns:
        The local path and a status: ``"verified"``, ``"downloaded"`` or ``"recorded"``
        (already present, checksum newly recorded).

    Raises:
        OSError: If a present or downloaded file does not match its recorded checksum.
    """
    rel = remote_relpath(kind, date)
    dest = raw_dir / rel
    status = "verified"
    if not dest.exists():
        download_file(f"{manifest.base_url}/{rel}", dest)
        status = "downloaded"
    digest = sha256_file(dest)
    record = manifest.files.get(rel)
    if record is None:
        manifest.files[rel] = FileRecord(size=dest.stat().st_size, sha256=digest)
        status = "recorded" if status == "verified" else status
    elif record.sha256 != digest or record.size != dest.stat().st_size:
        raise OSError(f"Checksum mismatch for {rel}; delete it and download again.")
    return dest, status


def list_archive(tar_path: Path) -> list[tuple[str, int]]:
    """Member names and sizes of a sensor archive."""
    with tarfile.open(tar_path, "r:gz") as tar:
        return [(m.name, m.size) for m in tar.getmembers() if m.isfile()]


def extract_members(tar_path: Path, dest_dir: Path, wanted: Iterable[str]) -> list[Path]:
    """Extract selected files (matched by base name) flat into ``dest_dir``.

    Only regular files whose base name is in ``wanted`` are extracted; paths inside
    the archive are ignored, so nothing can be written outside ``dest_dir``.

    Returns:
        Paths of the extracted files.
    """
    wanted_set = set(wanted)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            name = PurePosixPath(member.name).name
            if not member.isfile() or name not in wanted_set:
                continue
            src = tar.extractfile(member)
            if src is None:
                continue
            target = dest_dir / name
            tmp = target.with_name(target.name + ".part")
            with src, tmp.open("wb") as f:
                shutil.copyfileobj(src, f, _CHUNK)
            tmp.replace(target)
            out.append(target)
    return out
