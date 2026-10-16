import io
from datetime import datetime, timezone

from botocore.exceptions import ClientError
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from aurora_access import AccessError, Settings, Store, error_status, inventory, select_window


def fake_store(client):
    store = Store.__new__(Store)
    store.settings = Settings()
    store.client = client
    return store


class Pages:
    def __init__(self):
        self.calls = []

    def list_objects_v2(self, **kwargs):
        self.calls.append(kwargs)
        if 'ContinuationToken' not in kwargs:
            return {'CommonPrefixes': [{'Prefix': 'root/a/'}], 'IsTruncated': True,
                    'NextContinuationToken': 'second'}
        return {'Contents': [{'Key': 'root/file.nc', 'Size': 10,
                             'LastModified': datetime(2026, 1, 1, tzinfo=timezone.utc)}]}


def test_paginated_listing_and_cap():
    client = Pages()
    dirs, objects, complete = fake_store(client).list_level('root/')
    assert dirs == ['root/a/'] and objects[0]['bytes'] == 10 and complete
    assert client.calls[1]['ContinuationToken'] == 'second'
    assert all(c['Delimiter'] == '/' for c in client.calls)
    assert fake_store(Pages()).list_level('root/', max_pages=1)[2] is False


def test_access_denied_not_converted_to_missing_metadata():
    class Denied:
        def get_object(self, **kwargs):
            raise ClientError({'Error': {'Code': 'AccessDenied'}}, 'GetObject')
    with pytest.raises(ClientError):
        fake_store(Denied()).metadata('dataset.zarr/')
    result = inventory(fake_store(Denied()), max_prefixes=1)
    assert result['prefix_limit_reached']


def test_missing_metadata_and_format_detection():
    store = fake_store(None)
    store.json_object = lambda key: None
    assert store.metadata('x.zarr')['format'] == 'unidentified'
    store.json_object = lambda key: {'zarr_format': 3} if key.endswith('zarr.json') else None
    assert store.metadata('x.zarr')['format'] == 'zarr-v3'


def test_metadata_byte_limit_closes_body():
    body = io.BytesIO(b'{}')
    class Large:
        def get_object(self, **kwargs):
            return {'Body': body, 'ContentLength': 9 * 1024**2}
    with pytest.raises(AccessError):
        fake_store(Large()).json_object('x')
    assert body.closed


def test_selection_is_half_open_and_preserves_nan():
    ds = xr.Dataset({'temperature': ('time', [1., np.nan, 3., 4.])},
                    coords={'time': pd.date_range('2026-01-01', periods=4, freq='h')})
    selected, estimate = select_window(ds, 'temperature', '2026-01-01', '2026-01-01T02:00')
    assert selected.size == 2 and np.isnan(selected[1]) and estimate == 32
    with pytest.raises(AccessError, match='above'):
        select_window(ds, 'temperature', '2026-01-01', '2026-01-02', max_bytes=1)
    with pytest.raises(AccessError, match='No timestamps'):
        select_window(ds, 'temperature', '2025-01-01', '2025-01-02')


def test_height_selection_before_budget():
    ds = xr.Dataset({'beta': (('time', 'range'), np.ones((2, 4)))},
                    coords={'time': pd.date_range('2026-01-01', periods=2, freq='h'),
                            'range': [0, 1000, 3000, 5000]})
    selected, _ = select_window(ds, 'beta', '2026-01-01', '2026-01-02')
    assert selected.sizes['range'] == 3


def test_transient_failure_is_inconclusive():
    assert error_status(ClientError({'Error': {'Code': '504'}}, 'ListObjectsV2')) == 'inconclusive_transient'
    assert error_status(TimeoutError()) == 'inconclusive_transient'


def test_retries_and_timeouts_are_bounded(monkeypatch):
    captured = {}
    class Session:
        def client(self, service, **kwargs):
            captured.update(kwargs)
    monkeypatch.setattr('boto3.Session', lambda **kwargs: Session())
    Store()
    config = captured['config']
    assert config.retries['total_max_attempts'] == 3
    assert config.connect_timeout == 10 and config.read_timeout == 30


def test_inventory_reports_denial_and_never_descends_into_chunks():
    class Discovery:
        settings = Settings()
        def list_level(self, prefix):
            if prefix.endswith('raw/'):
                raise ClientError({'Error': {'Code': 'AccessDenied'}}, 'ListObjectsV2')
            if prefix.endswith('products/'):
                return [prefix + 'example.zarr/'], [], True
            assert not prefix.endswith('.zarr/')
            return [], [], True
        def metadata(self, prefix):
            return {'format': 'zarr-v2-consolidated', 'arrays': {}, 'coverage': {}}
    result = inventory(Discovery())
    assert any(r['status'] == 'access_denied' for r in result['datasets'])
    assert any(r['status'] == 'metadata_readable' for r in result['datasets'])


def test_large_family_records_sampling():
    class Discovery:
        settings = Settings()
        def list_level(self, prefix):
            if prefix.endswith('products/'):
                return [prefix + 'model/'], [], True
            if prefix.endswith('model/'):
                return [prefix + 'issues/'], [], True
            return [], [], True
        def metadata(self, prefix):
            raise AssertionError('Excluded issue archives must not be traversed')
    result = inventory(Discovery())
    assert not any(r['prefix'].endswith('issues/') for r in result['datasets'])


def test_example_uses_previous_elapsed_day_and_closes_dataset():
    from aurora_access import example
    closed = []
    ds = xr.Dataset({'h1_t': ('time', np.arange(48, dtype=float))},
                    coords={'time': pd.date_range('2026-01-01', periods=48, freq='h')})
    ds.set_close(lambda: closed.append(True))
    class Local:
        settings = Settings()
        def metadata(self, prefix):
            return {}
        def open_dataset(self, prefix, metadata):
            return ds
    values, report = example(Local(), 'meteorology')
    assert report['window_start_utc'] == '2026-01-01T00:00:00'
    assert values.size == 1 and report['finite_values'] == 1
    assert closed == [True]


def test_example_rejects_unordered_time_and_all_missing_data():
    from aurora_access import example
    class Local:
        settings = Settings()
        def metadata(self, prefix):
            return {}
        def open_dataset(self, prefix, metadata):
            return self.ds
    store = Local()
    store.ds = xr.Dataset({'h1_t': ('time', [np.nan, np.nan])},
                          coords={'time': pd.to_datetime(['2026-01-01', '2026-01-02'])})
    with pytest.raises(AccessError, match='no finite'):
        example(store, 'meteorology')
    store.ds = store.ds.isel(time=[1, 0])
    with pytest.raises(AccessError, match='unordered'):
        example(store, 'meteorology')
