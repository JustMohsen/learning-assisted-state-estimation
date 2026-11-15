"""Download, verify and extract NCLT sensor archives, ground truth and covariances.

Examples:
    uv run python scripts/download_nclt.py --split validation
    uv run python scripts/download_nclt.py --dates 2013-01-10 2012-01-22
    uv run python scripts/download_nclt.py --split all

Files go to ``local_data/nclt/raw/`` (unmodified) and the sensor CSVs are extracted to
``local_data/nclt/extracted/<date>/``. Checksums are recorded in
``configs/data/nclt_files.yaml`` on first download and verified afterwards.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time

from lase.data.download import FileKind, FileManifest, extract_members, fetch, list_archive
from lase.data.splits import Split, load_split
from lase.utils.paths import nclt_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--split", choices=[s.value for s in Split] + ["all"])
    sel.add_argument("--dates", nargs="+", type=dt.date.fromisoformat)
    p.add_argument(
        "--kinds",
        nargs="+",
        choices=[k.value for k in FileKind],
        default=[k.value for k in FileKind],
    )
    p.add_argument("--no-extract", action="store_true", help="Skip extracting sensor archives.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    manifest_split = load_split()
    if args.dates:
        dates = tuple(args.dates)
        for d in dates:
            manifest_split.split_of(d)  # raises for dates outside the manifest
    elif args.split == "all":
        dates = manifest_split.dates()
    else:
        dates = manifest_split.dates(Split(args.split))

    root = nclt_dir()
    raw = root / "raw"
    manifest = FileManifest.load()
    kinds = [FileKind(k) for k in args.kinds]
    t0 = time.monotonic()
    for i, date in enumerate(dates, 1):
        for kind in kinds:
            path, status = fetch(kind, date, raw, manifest)
            manifest.save()  # persist after every file so interruptions lose nothing
            print(
                f"[{i}/{len(dates)}] {date} {kind.value:13s} {status:10s} "
                f"{path.stat().st_size / 1e6:8.1f} MB  ({time.monotonic() - t0:6.0f} s)",
                flush=True,
            )
            if kind is FileKind.SENSOR and not args.no_extract:
                names = [name.rsplit("/", 1)[-1] for name, _ in list_archive(path)]
                out = extract_members(path, root / "extracted" / date.isoformat(), names)
                size = sum(p.stat().st_size for p in out)
                print(f"    extracted {len(out)} files, {size / 1e6:.1f} MB", flush=True)
    print(f"Done: {len(dates)} dates in {time.monotonic() - t0:.0f} s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
