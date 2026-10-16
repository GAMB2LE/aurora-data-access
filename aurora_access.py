"""Read-only discovery and bounded Xarray access to the AURORA object store."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from zarr.storage import FsspecStore
from zarr.core.sync import sync

PRODUCTS = 'data/output/aurora-cloud/products/'
ROOTS = {
    'raw': 'data/incoming/aurora-cloud/raw/',
    'products': PRODUCTS,
    'model-evaluation': 'data/output/aurora-cloud/model-evaluation/',
    'internal': 'data/internal/aurora-cloud/',
}
EXAMPLES = {
    'meteorology': (PRODUCTS + 'vaisalamet/vaisalamet.zarr/', 'h1_t'),
    'ceilometer': (PRODUCTS + 'cl61/gamb2le_depolarisation_lidar_ceilometer_aurora.zarr/', 'beta_att'),
    'power': (PRODUCTS + 'power/power.zarr/', 'BatterySOC'),
}
MAX_BYTES = 100 * 1024**2


class AccessError(RuntimeError):
    """A bounded read could not be completed; no remote changes were made."""


def error_status(error):
    if isinstance(error, ClientError):
        code = error.response['Error']['Code']
        if code in {'AccessDenied', '403', 'InvalidAccessKeyId', 'SignatureDoesNotMatch'}:
            return 'access_denied'
        if code in {'NoSuchKey', '404', 'NotFound'}:
            return 'not_found'
        if code in {'SlowDown', 'RequestTimeout', '500', '502', '503', '504'}:
            return 'inconclusive_transient'
    if isinstance(error, (TimeoutError, ConnectionError)) or 'Timeout' in type(error).__name__:
        return 'inconclusive_transient'
    return 'unreadable_' + type(error).__name__


@dataclass(frozen=True)
class Settings:
    endpoint: str = 'https://gamb2le-o.s3-ext.jc.rl.ac.uk'
    bucket: str = 'gamb2le-o'
    profile: str | None = None
    anonymous: bool = False

    @classmethod
    def from_env(cls):
        return cls(endpoint=os.getenv('AURORA_S3_ENDPOINT', cls.endpoint),
                   bucket=os.getenv('AURORA_S3_BUCKET', cls.bucket),
                   profile=os.getenv('AWS_PROFILE'),
                   anonymous=os.getenv('AURORA_S3_ANONYMOUS', '0') == '1')


class Store:
    """Only exposes listing, metadata reads and read-only Zarr opening."""

    def __init__(self, settings=None):
        self.settings = settings or Settings.from_env()
        s = self.settings
        session = boto3.Session(profile_name=s.profile)
        options = dict(connect_timeout=10, read_timeout=30,
                       retries={'mode': 'standard', 'total_max_attempts': 3},
                       s3={'addressing_style': 'path'})
        self.client = session.client('s3', endpoint_url=s.endpoint,
                                    config=Config(signature_version=UNSIGNED if s.anonymous else 's3v4', **options))
        self._fs = None

    @property
    def fs(self):
        if self._fs is None:
            s = self.settings
            self._fs = s3fs.S3FileSystem(
                anon=s.anonymous, **({"profile": s.profile} if s.profile else {}),
                client_kwargs={'endpoint_url': s.endpoint},
                config_kwargs={'connect_timeout': 10, 'read_timeout': 30,
                               'retries': {'mode': 'standard', 'total_max_attempts': 3},
                               's3': {'addressing_style': 'path'}},
                max_concurrency=2, asynchronous=True, skip_instance_cache=True)
        return self._fs

    def close(self):
        self.client.close()
        if self._fs is not None and self._fs._s3 is not None:
            sync(self._fs._s3.close())
            self._fs = None

    def list_level(self, prefix, max_pages=20):
        """List one hierarchy level, never recursively enumerate array chunks."""
        directories, objects, token = [], [], None
        for _ in range(max_pages):
            args = dict(Bucket=self.settings.bucket, Prefix=prefix, Delimiter='/', MaxKeys=1000)
            if token:
                args['ContinuationToken'] = token
            page = self.client.list_objects_v2(**args)
            directories.extend(p['Prefix'] for p in page.get('CommonPrefixes', []))
            objects.extend({'key': o['Key'], 'bytes': o['Size'],
                            'modified_utc': o['LastModified'].isoformat()}
                           for o in page.get('Contents', []))
            if not page.get('IsTruncated'):
                return directories, objects, True
            token = page.get('NextContinuationToken')
            if not token:
                raise AccessError('Truncated listing did not supply a continuation token')
        return directories, objects, False

    def json_object(self, key, limit=8 * 1024**2):
        try:
            obj = self.client.get_object(Bucket=self.settings.bucket, Key=key)
        except ClientError as exc:
            if error_status(exc) == 'not_found':
                return None
            raise
        with obj['Body'] as body:
            if obj['ContentLength'] > limit:
                raise AccessError('Metadata exceeds 8 MiB read limit')
            return json.loads(body.read(limit + 1))

    def metadata(self, prefix):
        prefix = prefix.rstrip('/') + '/'
        meta = self.json_object(prefix + '.zmetadata')
        if meta is not None:
            entries = meta['metadata']
            arrays = {}
            for key, value in entries.items():
                if key.endswith('/.zarray'):
                    name = key.removesuffix('/.zarray')
                    attrs = entries.get(name + '/.zattrs', {})
                    arrays[name] = {'shape': value['shape'], 'chunks': value['chunks'],
                                    'dtype': value['dtype'], 'dimensions': attrs.get('_ARRAY_DIMENSIONS', []),
                                    'units': attrs.get('units'), 'long_name': attrs.get('long_name')}
            attrs = entries.get('.zattrs', {})
            return {'format': 'zarr-v2-consolidated', 'arrays': arrays,
                    'coverage': {k: attrs[k] for k in ('time_coverage_start', 'time_coverage_end') if k in attrs},
                    'coverage_basis': 'metadata_attributes_not_verified'}
        if self.json_object(prefix + 'zarr.json') is not None:
            return {'format': 'zarr-v3', 'arrays': {}, 'coverage': {}, 'coverage_basis': 'unknown'}
        if self.json_object(prefix + '.zgroup') is not None:
            return {'format': 'zarr-v2-unconsolidated', 'arrays': {}, 'coverage': {}, 'coverage_basis': 'unknown'}
        return {'format': 'unidentified', 'arrays': {}, 'coverage': {}, 'coverage_basis': 'unknown'}

    def open_dataset(self, prefix, metadata=None):
        m = metadata or self.metadata(prefix)
        if not m['format'].startswith('zarr-'):
            raise AccessError('Not a verified ordinary Zarr dataset; do not assume Icechunk')
        mapper = FsspecStore(self.fs, read_only=True, path=self.settings.bucket + '/' + prefix.rstrip('/'))
        return xr.open_zarr(mapper, consolidated=m['format'] == 'zarr-v2-consolidated',
                            chunks={}, create_default_indexes=False)


def inventory(store, max_prefixes=160, product_depth=3):
    """Discover dataset roots; raw/internal inventory is intentionally shallow."""
    rows, queue = [], deque((area, root, 0) for area, root in ROOTS.items())
    visited = 0
    while queue and visited < max_prefixes:
        area, prefix, depth = queue.popleft()
        visited += 1
        row = {'area': area, 'prefix': prefix, 'uri': f's3://{store.settings.bucket}/{prefix}'}
        try:
            if prefix.rstrip('/').endswith('.zarr'):
                row.update(store.metadata(prefix))
                row['status'] = 'metadata_readable' if row['format'] != 'unidentified' else 'metadata_missing'
            else:
                dirs, objects, complete = store.list_level(prefix)
                row.update(status='listed' if complete else 'partial_listing',
                           child_prefixes=dirs, listed_object_count=len(objects),
                           listed_bytes=sum(o['bytes'] for o in objects),
                           sample_objects=objects[:5], format='directory', coverage_basis='unknown')
                limit = product_depth if area == 'products' else 1
                if complete and depth < limit:
                    eligible = [p for p in dirs if not any(x in p.lower() for x in ('backup', '.zarr-repair', '/issues/'))]
                    # A collection of dated stores is one dataset family: sample its latest
                    # three lexicographic members, with every member retained in child_prefixes.
                    if depth >= 2 and len(eligible) > 3:
                        row['child_sample_basis'] = 'last_three_lexicographic_prefixes_not_measurement_times'
                        row['children_not_inspected'] = len(eligible) - 3
                        eligible = sorted(eligible)[-3:]
                    queue.extend((area, p, depth + 1) for p in eligible)
                elif dirs:
                    row['deeper_listing'] = 'not_scanned'
        except Exception as exc:
            row['status'] = error_status(exc)
        rows.append(row)
    seen = {r['prefix'] for r in rows}
    for prefix, _ in EXAMPLES.values():
        if prefix not in seen:
            rows.append({'area': 'products', 'prefix': prefix, 'status': 'configured_not_observed'})
    return {'observed_at_utc': datetime.now(timezone.utc).isoformat(),
            'endpoint': store.settings.endpoint, 'bucket': store.settings.bucket,
            'scope': 'bounded prefix and metadata discovery; not a full object or coverage audit',
            'prefix_limit_reached': bool(queue), 'unvisited_prefixes': len(queue), 'datasets': rows}


def select_window(ds, variable, start, end, max_bytes=MAX_BYTES, max_range=3000):
    """Select [start, end) from a dataset with an already bounded time axis."""
    if variable not in ds:
        raise AccessError(f'Variable {variable!r} is absent')
    times = pd.DatetimeIndex(ds.time.values)
    mask = (times >= pd.Timestamp(start)) & (times < pd.Timestamp(end))
    idx = np.flatnonzero(mask)
    if not len(idx):
        raise AccessError('No timestamps in the requested window')
    array = ds[variable].isel(time=idx)
    if 'range' in array.dims:
        values = array['range'].values
        array = array.isel(range=np.flatnonzero((values >= 0) & (values <= max_range)))
    if array.size == 0:
        raise AccessError('Selection contains no values')
    estimate = array.nbytes + sum(c.nbytes for c in array.coords.values())
    if estimate > max_bytes:
        raise AccessError(f'Selection needs approximately {estimate} bytes, above {max_bytes}')
    return array, estimate


def example(store, name, hours=1):
    """Use a bounded recent time tail, avoiding loading the entire time index."""
    if not 0 < hours <= 24:
        raise ValueError('hours must be between 0 and 24')
    prefix, variable = EXAMPLES[name]
    meta = store.metadata(prefix)
    ds = store.open_dataset(prefix, meta)
    try:
        if 'time' not in ds or not ds.sizes.get('time'):
            raise AccessError('No time coordinate')
        # 200,000 timestamps = about 1.6 MiB decoded, separate from selection budget.
        tail = ds.isel(time=slice(max(0, ds.sizes['time'] - 200_000), None))
        times = pd.DatetimeIndex(tail.time.values)
        if times.hasnans or not times.is_monotonic_increasing or not times.is_unique:
            raise AccessError('Recent timestamps are missing, duplicated or unordered')
        boundary = min(times[-1].floor('D'), pd.Timestamp.now(tz='UTC').tz_localize(None).floor('D'))
        day = boundary - pd.Timedelta(days=1)
        # Use the newest prior calendar day that has observations in this tail.
        candidates = times[times < boundary]
        if not len(candidates):
            raise AccessError('No previous UTC day in the bounded time tail')
        day = candidates[-1].floor('D')
        if times[0] > day and ds.sizes['time'] > 200_000:
            raise AccessError('Time tail does not reach day start; use an explicit bounded historical read')
        tail = tail.assign_coords(time=times)
        array, estimate = select_window(tail, variable, day, day + pd.Timedelta(hours=hours))
        array = array.compute(scheduler='threads', num_workers=2)
        finite = int(np.isfinite(array.values).sum())
        if not finite:
            raise AccessError('Selection contains no finite observations')
        summary = {'example': name, 'uri': f's3://{store.settings.bucket}/{prefix}',
                   'variable': variable, 'units': array.attrs.get('units', 'not supplied by source'),
                   'window_start_utc': day.isoformat(), 'window_end_exclusive_utc': (day + pd.Timedelta(hours=hours)).isoformat(),
                   'first_observation_utc': str(array.time.values[0]), 'last_observation_utc': str(array.time.values[-1]),
                   'selected_bytes_estimate': estimate, 'finite_values': finite, 'total_values': array.size,
                   'time_tail_first_utc': str(times[0]), 'time_tail_last_utc': str(times[-1]),
                   'note': 'Previous elapsed calendar day, not proof of complete daily coverage. Missing values retained.'}
        return array, summary
    finally:
        ds.close()


def save_inventory(result, directory='reports'):
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'inventory.json').write_text(json.dumps(result, indent=2))
    lines = ['# AURORA availability inventory', '', result['observed_at_utc'], '', result['scope'], '',
             '| Prefix | Status | Format |', '|---|---|---|']
    lines += [f"| `{r['prefix']}` | {r['status']} | {r.get('format', 'unknown')} |" for r in result['datasets']]
    lines += ['', f"Unvisited queued prefixes: {result['unvisited_prefixes']}.",
              'Counts and bytes refer only to objects directly listed at that level, not recursive totals.',
              'Metadata readability does not verify every chunk. Modified times are not measurement times.']
    (out / 'availability.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='reports')
    parser.add_argument('--max-prefixes', type=int, default=160)
    args = parser.parse_args()
    if args.max_prefixes < 1:
        parser.error('--max-prefixes must be positive')
    store = Store()
    try:
        result = inventory(store, max_prefixes=args.max_prefixes)
    finally:
        store.close()
    save_inventory(result, args.output)
    print(f"Saved {len(result['datasets'])} discovery rows to {args.output}")
    if any(r['status'] not in {'listed', 'metadata_readable'} for r in result['datasets']) or result['prefix_limit_reached']:
        print('Some entries are incomplete or unreadable; inspect availability.md.')


if __name__ == '__main__':
    main()
