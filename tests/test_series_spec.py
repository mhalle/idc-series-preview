import logging

from idc_series_preview.series_spec import parse_and_normalize_series


def test_parse_accepts_dotted_dicom_uid():
    logger = logging.getLogger("test")
    root = "s3://idc-open-data"
    uid = "1.2.276.0.7230010.3.1.3.313263360.37570.1706311149.189517"

    resolved_root, resolved_uid = parse_and_normalize_series(uid, root, logger)

    assert resolved_root == root
    assert resolved_uid == uid
