import polars as pl
from pydicom.dataset import Dataset, FileMetaDataset
import struct

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


def test_generate_parquet_table_stores_pixel_metadata_for_native_with_raw_header():
    ds = make_dataset("1.2.3", [0.0, 0.0, 0.0], 1)
    ds.Rows = 2
    ds.Columns = 4
    ds.BitsAllocated = 1
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = "1.2.840.10008.1.2.1"  # Explicit VR Little Endian (native)
    ds.file_meta = file_meta

    prefix = b"\x01" * 10
    pixel_tag = b"\xe0\x7f\x10\x00"  # PixelData
    vr = b"OB"
    reserved = b"\x00\x00"
    length = struct.pack("<I", 4)
    raw_bytes = prefix + pixel_tag + vr + reserved + length + b"\x00" * 4

    df = _generate_parquet_table(
        {"1.2.3": ds},
        "series",
        "s3://bucket",
        {"1.2.3": raw_bytes},
    )

    assert df["_pixel_data_offset"].to_list()[0] == len(prefix) + 12
    # frame size: ceil(2*4*1/8) = 1 byte
    assert df["_frame_size"].to_list()[0] == 1
    assert df["_transfer_syntax_uid"].to_list()[0] == "1.2.840.10008.1.2.1"


def test_generate_parquet_table_pixel_metadata_none_for_compressed():
    ds = make_dataset("9.9.9", [0.0, 0.0, 0.0], 1)
    ds.Rows = 2
    ds.Columns = 4
    ds.BitsAllocated = 1
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = "1.2.840.10008.1.2.4.50"  # JPEG Baseline (compressed)
    ds.file_meta = file_meta

    df = _generate_parquet_table({"9.9.9": ds}, "series", "s3://bucket")

    assert df["_pixel_data_offset"].to_list()[0] is None
    assert df["_frame_size"].to_list()[0] is None
    assert df["_transfer_syntax_uid"].to_list()[0] == "1.2.840.10008.1.2.4.50"
