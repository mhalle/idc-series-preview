import argparse
import logging
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from PIL import Image

from idc_series_preview.cli_core import (
    build_index_command,
    contrast_mosaic_command,
    get_index_command,
    _setup_contrast_mosaic_subcommand,
    _setup_mosaic_subcommand,
    mosaic_command,
)


class StubSeriesIndex:
    def __init__(self, series_spec, root, cache_dir, use_cache=True, force_rebuild=False):
        self.series_uid = series_spec if isinstance(series_spec, str) else "stub-series"
        self.instance_count = 1
        self.index_dataframe = pl.DataFrame(
            {
                "Index": [0],
                "IndexNormalized": [0.0],
                "SeriesUID": [self.series_uid],
                "PrimaryAxis": ["Z"],
                "PrimaryPosition": [0.0],
                "DataURL": [f"{root.rstrip('/')}/{self.series_uid}/instance.dcm"],
                "PixelSpacing": [[0.5, 0.5]],
            }
        )
        index_path = Path(cache_dir) / "indices" / f"{self.series_uid}_index.parquet"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_dataframe.write_parquet(str(index_path))


@pytest.fixture(autouse=True)
def patch_series_index(monkeypatch):
    monkeypatch.setattr("idc_series_preview.api.SeriesIndex", StubSeriesIndex)


def test_get_index_command_jsonl_preserves_lists(tmp_path):
    cache_dir = tmp_path / "cache"
    output_path = tmp_path / "out.jsonl"
    args = SimpleNamespace(
        series="demo-series",
        root="s3://bucket",
        cache_dir=str(cache_dir),
        verbose=False,
        output=str(output_path),
        format="jsonl",
        rebuild=False,
    )
    logger = logging.getLogger("test")

    rc = get_index_command(args, logger)
    assert rc == 0
    assert output_path.is_file()

    content = output_path.read_text().strip()
    assert '"PixelSpacing":[0.5,0.5]' in content


def test_build_index_command_respects_cache_dir(tmp_path):
    cache_dir = tmp_path / "cache"
    args = SimpleNamespace(
        series=["series-a"],
        root="s3://bucket",
        cache_dir=str(cache_dir),
        output=None,
        rebuild=False,
        all=False,
        verbose=False,
    )
    logger = logging.getLogger("test")

    rc = build_index_command(args, logger)
    assert rc == 0

    expected = cache_dir / "indices" / "series-a_index.parquet"
    assert expected.is_file()


def test_mosaic_command_shrinks_rows_for_unique_slices(monkeypatch, tmp_path):
    requested_positions: list[list[float]] = []

    class DummySeriesIndex:
        def __init__(self, seriesuid, root, cache_dir=None, use_cache=True):  # noqa: D401
            self.instance_count = 2

        def get_instances(self, positions, max_workers=None, headers_only=False):
            requested_positions.append(list(positions))
            return [
                SimpleNamespace(instance_uid=f"uid{i}", dataset=object())
                for i in range(len(positions))
            ]

    class DummyRenderer:
        def __init__(self, image_width, window_settings):
            pass

        def render_instance(self, dataset):
            return Image.new('L', (8, 8))

    class RecordingMosaicRenderer:
        created = []

        def __init__(self, tile_width, tile_height, image_width):
            self.tile_width = tile_width
            self.tile_height = tile_height
            self.images = None
            RecordingMosaicRenderer.created.append(self)

        def tile_images(self, images):
            self.images = list(images)
            return Image.new('L', (8, 8))

    saved = []

    def fake_save_image(image, output, quality=85):
        saved.append((output, quality))
        return True

    monkeypatch.setattr("idc_series_preview.api.SeriesIndex", DummySeriesIndex)
    monkeypatch.setattr("idc_series_preview.image_utils.InstanceRenderer", DummyRenderer)
    monkeypatch.setattr("idc_series_preview.image_utils.MosaicRenderer", RecordingMosaicRenderer)
    monkeypatch.setattr("idc_series_preview.cli_core.save_image", fake_save_image)

    args = SimpleNamespace(
        seriesuid="series",
        output=str(tmp_path / "mosaic.webp"),
        root="s3://bucket",
        image_width=128,
        contrast="auto",
        quality=75,
        verbose=False,
        start=0.0,
        end=1.0,
        tile_width=4,
        tile_height=4,
        cache_dir=None,
        no_cache=False,
    )
    logger = logging.getLogger("test")

    rc = mosaic_command(args, logger)
    assert rc == 0
    assert saved  # ensure image attempted to save

    # Only two unique slices exist, so we should only request two positions
    assert requested_positions
    assert len(requested_positions[-1]) == 2

    renderer = RecordingMosaicRenderer.created[-1]
    assert renderer.tile_height == 1  # shrink from 4 to 1 row
    assert len(renderer.images) == 2


def test_contrast_mosaic_shrinks_rows_for_unique_slices(monkeypatch, tmp_path):
    requested_positions: list[list[float]] = []

    class DummySeriesIndex:
        def __init__(self, seriesuid, root, cache_dir=None, use_cache=True):
            self.instance_count = 2

        def get_instances(self, positions, max_workers=None, headers_only=False):
            requested_positions.append(list(positions))
            return [
                SimpleNamespace(instance_uid=f"uid{i}", dataset=object())
                for i in range(len(positions))
            ]

    class DummyRenderer:
        def __init__(self, image_width, window_settings):
            pass

        def render_instance(self, dataset):
            return Image.new('L', (6, 6))

    class RecordingGridRenderer:
        created = []

        def __init__(self, tile_width, tile_height, image_width):
            self.tile_width = tile_width
            self.tile_height = tile_height
            self.images = None
            RecordingGridRenderer.created.append(self)

        def tile_images(self, images):
            self.images = list(images)
            return Image.new('L', (6, 6))

    saved = []

    def fake_save_image(image, output, quality=85):
        saved.append((output, quality))
        return True

    monkeypatch.setattr("idc_series_preview.api.SeriesIndex", DummySeriesIndex)
    monkeypatch.setattr("idc_series_preview.image_utils.InstanceRenderer", DummyRenderer)
    monkeypatch.setattr("idc_series_preview.image_utils.MosaicRenderer", RecordingGridRenderer)
    monkeypatch.setattr("idc_series_preview.cli_core.save_image", fake_save_image)

    args = SimpleNamespace(
        seriesuid="series",
        output=str(tmp_path / "grid.webp"),
        root="s3://bucket",
        image_width=128,
        contrast=["lung", "bone"],
        quality=75,
        verbose=False,
        position=None,
        slice_offset=0,
        start=0.0,
        end=1.0,
        tile_height=5,
        cache_dir=None,
        no_cache=False,
    )
    logger = logging.getLogger("test")

    rc = contrast_mosaic_command(args, logger)
    assert rc == 0
    assert saved

    assert requested_positions
    assert len(requested_positions[-1]) == 2

    renderer = RecordingGridRenderer.created[-1]
    assert renderer.tile_height == 2
    assert len(renderer.images) == 4


def _build_single_command_parser(setup_fn):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True
    setup_fn(subparsers)
    return parser


def test_mosaic_short_flag_parses_tile_width():
    parser = _build_single_command_parser(_setup_mosaic_subcommand)
    args = parser.parse_args(["mosaic", "series", "out.webp", "-t", "7"])
    assert args.tile_width == 7


def test_contrast_mosaic_short_flag_parses_tile_height():
    parser = _build_single_command_parser(_setup_contrast_mosaic_subcommand)
    args = parser.parse_args([
        "contrast-mosaic",
        "series",
        "out.webp",
        "-c",
        "lung",
        "--start",
        "0.0",
        "--end",
        "0.5",
        "-t",
        "3",
    ])
    assert args.tile_height == 3
