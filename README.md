# AURORA data access

Explore the AURORA object store with a reproducible Python environment, a bounded
inventory, and a Jupyter tutorial. Start with
[`tutorial_aurora_data_access.ipynb`](tutorial_aurora_data_access.ipynb).

This is a fork of [TONe ICO data access](https://github.com/Troll-Observing-Network/integrated-cloud-observatory-data-access),
starting at `7c3bed71ea339c8826e6cb5377d971e437cc1624`. Its original tutorial is
retained as `tutorial_tone_ico_data_access.ipynb`. The AURORA examples read ordinary
Zarr stores; they do not assume the Icechunk format used by TONe.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from this repository:

```bash
uv sync --locked
```

Python 3.14 is selected by `.python-version`. `uv.lock` pins the environment,
including the Xarray, Zarr and S3 reader versions tested together.

## Authentication

Use an authorised AWS credentials profile, preferably with only `ListBucket` and
`GetObject` permissions on the required AURORA prefixes. Store credentials in your
normal private AWS configuration, never in this repository or a notebook cell.
Alternatively, provide `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and, if needed,
`AWS_SESSION_TOKEN` through your existing secure environment setup.

```bash
export AWS_PROFILE=aurora-readonly
export AURORA_S3_ENDPOINT=https://gamb2le-o.s3-ext.jc.rl.ac.uk
export AURORA_S3_BUCKET=gamb2le-o
uv run python -m aurora_access
uv run jupyter lab
```

The endpoint and bucket above are defaults. Omit `AWS_PROFILE` when using environment
credentials. `AURORA_S3_ANONYMOUS=1` explicitly selects anonymous requests, but
anonymous listing was denied during initial verification. This project does not
create credentials, change access policies, or write to S3. Existing production
credentials used for local verification were held in process memory; their
permissions have not been represented as read-only credentials.

## Inventory

`uv run python -m aurora_access --output reports` writes `inventory.json` and
`availability.md` locally. These outputs are ignored by Git because they may contain
internal archive names and object listings.

| Area | Prefix within `gamb2le-o` |
|---|---|
| Original instrument files | `data/incoming/aurora-cloud/raw/` |
| Derived operational products | `data/output/aurora-cloud/products/` |
| Model evaluation | `data/output/aurora-cloud/model-evaluation/` |
| Internal archive evidence | `data/internal/aurora-cloud/` |

Discovery uses paginated, delimiter-based listings, stops at Zarr roots, and does
not enumerate array chunks. It traverses product families to three levels; other
areas receive shallow listings. Backup stores are excluded from traversal.
Collections with more than three child prefixes at depth two are sampled using
the last three lexicographic names; all child names remain recorded. This order
is not a claim about measurement dates. The default cap is 160 visited prefixes;
`--max-prefixes` can change it. Any incomplete pagination, unvisited prefixes,
access denial, missing metadata, or transient failures are reported explicitly.

Counts and bytes describe objects directly listed at one level, never recursive
archive totals. Metadata-derived coverage is labelled as unverified. Object
modification times are not observation times. Metadata readability is not a check
of every chunk or proof that the archive matches the instrument source.

## Tutorial

The examples select `h1_t` from Vaisala MET, `beta_att` from CL61, and `BatterySOC`
from power telemetry. They inspect at most 200,000 recent timestamps and use the
first hour of the newest prior elapsed UTC day containing observations in that
tail. This does not prove a complete day of data. A too-short, unordered or
nonunique time tail produces an explicit blocker rather than a guessed selection.

Variables and CL61 range (0–3,000 m) are selected before computation. The estimated
decoded selection must be below 100 MiB. Remote chunk transfer can exceed this
amount; the limit is not a bandwidth cap. Metadata reads are capped at 8 MiB per
object, and the recent time coordinate adds about 1.6 MiB decoded. Keep `HOURS=1`
initially; increase it up to 24 only after inspecting chunk sizes.

Missing values stay missing and scatter plots do not join across time gaps.
Unknown units are labelled as absent from source metadata. CL61 range is not
silently converted to height. Finite values do not establish calibration or
scientific quality. These mutable operational products are distinct from original
instrument files and validated scientific NetCDF products. The code licence does
not establish permission to redistribute observations.

The module offers `Store.list_level`, `Store.metadata`, `Store.open_dataset`,
`inventory`, `select_window`, and `example`. `open_dataset` is lazy and suppresses
creation of a full time index. Only ordinary Zarr is supported by the AURORA reader;
unknown formats remain inventory entries. No Icechunk repositories were assumed
or converted.

## Reproduce verification

```bash
uv run python -m pytest -q
uv run jupyter nbconvert --execute --to notebook \
  tutorial_aurora_data_access.ipynb --output tutorial_aurora_data_access.executed.ipynb \
  --output-dir reports --ExecutePreprocessor.timeout=600
uv run jupyter nbconvert --to html reports/tutorial_aurora_data_access.executed.ipynb \
  --output-dir reports
```

Execute from the repository root so the local module imports correctly. The
notebook saves an inventory, plot PNGs, and `validation.json` with tested locations,
UTC windows, finite-value counts and elapsed time. Read the status table: execution
can finish with a reported source-access blocker. Only `readable` examples have
successfully loaded finite observations. Public notebooks are committed without
outputs; executed evidence stays in `reports/`.

Tests cover pagination, listing limits, metadata format detection, access failures,
transient error classification, retry bounds, half-open UTC selections, retained
missing values, height selection and memory limits. They use synthetic data and do
not require object-store credentials.
