import logging

from idc_series_preview.series_spec import parse_and_normalize_series


def test_parse_accepts_dotted_dicom_uid(tmp_path):
    logger = logging.getLogger("test")
    root = "s3://idc-open-data"
    uid = "1.2.276.0.7230010.3.1.3.313263360.37570.1706311149.189517"

    cache_dir = tmp_path / "cache"
    resolved_uid, series_url = parse_and_normalize_series(uid, root, logger, cache_dir)

    assert resolved_uid == uid
    assert series_url.startswith(root)
    assert series_url.endswith("/")
