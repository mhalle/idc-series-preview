"""Lightweight client for IDC index parquet lookups."""

from typing import Optional

import polars as pl

from .constants import IDC_INDEX_URL


def get_series_url(series_uid: str) -> Optional[str]:
    """
    Resolve the storage URL for a series using the published IDC index parquet.

    Parameters
    ----------
    series_uid : str
        SeriesInstanceUID to look up.

    Returns
    -------
    Optional[str]
        The series_aws_url base (trailing slash) if found; otherwise None.
    """
    if not series_uid:
        return None

    try:
        df = (
            pl.scan_parquet(IDC_INDEX_URL)
            .filter(pl.col("SeriesInstanceUID") == series_uid)
            .select("series_aws_url")
            .collect()
        )
        if df.height == 0:
            return None
        url = df["series_aws_url"][0]
        if isinstance(url, str) and url.endswith("/*"):
            url = url[:-2]
        if url and not url.endswith("/"):
            url = url + "/"
        return url
    except Exception:
        return None
