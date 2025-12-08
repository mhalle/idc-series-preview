import polars as pl
from pydicom.dataset import Dataset

from idc_series_preview.index_cache import _generate_parquet_table


def make_dataset(
    sop_uid: str,
    position: list[float] | None,
    instance_number: int,
    pixel_spacing: list[float] | None = None,
    window_center: list[float] | None = None,
    window_width: list[float] | None = None,
):
    ds = Dataset()
    ds.SOPInstanceUID = sop_uid
    ds.InstanceNumber = instance_number
    if position is not None:
        ds.ImagePositionPatient = position
    if pixel_spacing is not None:
        ds.PixelSpacing = pixel_spacing
    if window_center is not None:
        ds.WindowCenter = window_center
    if window_width is not None:
        ds.WindowWidth = window_width
    return ds


def test_generate_parquet_table_preserves_list_tags():
    dataset = make_dataset(
        "1.2.3",
        [0.0, 0.0, 0.0],
        1,
        pixel_spacing=[0.5, 0.5],
        window_center=[40.0, 80.0],
        window_width=[400.0, 800.0],
    )

    df = _generate_parquet_table({"1.2.3": dataset}, "series", "s3://bucket")

    assert "PixelSpacing" in df.columns
    assert df.schema["PixelSpacing"] == pl.List(pl.Float32)
    assert df["PixelSpacing"].to_list()[0] == [0.5, 0.5]

    assert df.schema["WindowCenter"] == pl.List(pl.Float32)
    assert df["WindowCenter"].to_list()[0] == [40.0, 80.0]

    assert df.schema["WindowWidth"] == pl.List(pl.Float32)
    assert df["WindowWidth"].to_list()[0] == [400.0, 800.0]

    assert df["ImagePositionPatient"].to_list()[0] == [0.0, 0.0, 0.0]
    assert df["_index_normalized"].to_list() == [0.0]


def test_generate_parquet_table_normalized_index_multiple_slices():
    ds1 = make_dataset("1", [5.0, 0.0, 0.0], 1)
    ds2 = make_dataset("2", [1.0, 0.0, 0.0], 2)

    df = _generate_parquet_table({"1": ds1, "2": ds2}, "series", "s3://bucket")

    values = sorted(df["_index_normalized"].to_list())
    assert values == [0.0, 1.0]
    assert set(df["SOPInstanceUID"].to_list()) == {"1", "2"}
