# local_data/

Local data directory. **Everything here except this README is gitignored and never committed.** Each clone downloads its own copy.

The code finds this directory automatically (`lase.utils.paths.data_dir()`). To keep the data elsewhere, set `LASE_DATA_DIR` (see `.env.example`).

## Layout

```text
local_data/
└── nclt/
    ├── raw/                              # downloads, kept unmodified
    │   ├── sensor_data/<date>_sen.tar.gz # IMU, FOG, wheels, GPS, RTK, odometry, ... (compressed)
    │   ├── ground_truth/groundtruth_<date>.csv   # reference 6-DOF pose (~148 Hz)
    │   └── covariance/cov_<date>.csv     # reference graph-node covariances
    ├── extracted/<date>/                 # CSVs extracted from the sensor archives
    └── download.log                      # log of the last scripted download
```

## Getting the data

Source: [NCLT dataset](https://robots.engin.umich.edu/nclt/) (University of Michigan). Phase 1 only needs the sensor archives, ground truth and covariances, not the images, LiDAR or camera data.

Download size for all 27 sessions (checked 15 November 2026): about **2.75 GB** of sensor archives, **2.66 GB** of ground-truth CSVs and **10 MB** of covariances, roughly **5.4 GB** in total before extraction.

**Scripted download (recommended):**

```bash
uv run python scripts/download_nclt.py --split all        # all 27 sessions (~8 min at ~12 MB/s)
uv run python scripts/download_nclt.py --split validation # or train / test
uv run python scripts/download_nclt.py --dates 2013-01-10 # specific dates
```

It resumes interrupted downloads, verifies each file against the SHA-256 recorded in `configs/data/nclt_files.yaml`, and extracts all 12 sensor CSVs into `nclt/extracted/<date>/`. On disk: about 5.5 GB raw plus 6.9 GB extracted.

**Manual download** (alternative), per recording date `<date>` (e.g. `2013-01-10`):

```text
https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/sensor_data/<date>_sen.tar.gz
https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/ground_truth/groundtruth_<date>.csv
https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/covariance/cov_<date>.csv
```

Save each file under `nclt/raw/` in the matching subdirectory shown above. The recording dates and their train/validation/test split are listed in [PROJECT_SCOPE.md](../PROJECT_SCOPE.md), section 4.4.
