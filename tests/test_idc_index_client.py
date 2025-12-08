import polars as pl
import pytest

from idc_series_preview.idc_index_client import get_series_url


class DummyScan:
    """Minimal stand-in for pl.scan_parquet returning a lazy frame with known data."""

    def __init__(self, urls):
        self.urls = urls

    def filter(self, *_args, **_kwargs):
        return self

    def select(self, *_args, **_kwargs):
        return self

    def collect(self):
        return pl.DataFrame({"series_aws_url": self.urls})


@pytest.fixture(autouse=True)
def patch_scan_parquet(monkeypatch):
    """Monkeypatch pl.scan_parquet used inside get_series_url."""
    def _scan_parquet(_url):
        return DummyScan(["s3://idc-open-data/series/*"])

    monkeypatch.setattr("idc_series_preview.idc_index_client.pl.scan_parquet", _scan_parquet)
    yield


def test_get_series_url_strips_wildcard_and_normalizes_trailing_slash():
    url = get_series_url("dummy")
    assert url == "s3://idc-open-data/series/"
