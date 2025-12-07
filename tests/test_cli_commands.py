import argparse
import json
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
    image_command,
    header_command,
    video_command,
    _map_quality_to_crf,
)
from idc_series_preview.constants import DEFAULT_IMAGE_QUALITY


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


def test_header_command_writes_json(monkeypatch, tmp_path):
    class DummySeriesIndex:
        def __init__(self, *args, **kwargs):
            self.index_dataframe = pl.DataFrame(
                {
                    "IndexNormalized": [0.0, 0.5, 1.0],
                    "SeriesUID": ["a", "b", "c"],
                    "Custom": ["first", "middle", "last"],
                    "WindowWidth": [1500, 1400, 1300],
                }
            )

    monkeypatch.setattr("idc_series_preview.api.SeriesIndex", DummySeriesIndex)

    output_path = tmp_path / "header.json"
    args = SimpleNamespace(
        seriesuid="series",
        root="s3://bucket",
        cache_dir=None,
        no_cache=False,
        position=0.6,
        slice_offset=0,
        output=str(output_path),
        indent=2,
        tags=[],
        verbose=False,
    )
    logger = logging.getLogger("test")

    rc = header_command(args, logger)
    assert rc == 0
    data = json.loads(output_path.read_text())
    assert data["Custom"] == "middle"
    assert data["WindowWidth"] == 1400


def test_header_command_respects_slice_offset(monkeypatch, capsys):
    class DummySeriesIndex:
        def __init__(self, *args, **kwargs):
            self.index_dataframe = pl.DataFrame(
                {
                    "IndexNormalized": [0.0, 0.5, 1.0],
                    "Value": [1, 2, 3],
                }
            )

    monkeypatch.setattr("idc_series_preview.api.SeriesIndex", DummySeriesIndex)

    args = SimpleNamespace(
        seriesuid="series",
        root="s3://bucket",
        cache_dir=None,
        no_cache=False,
        position=0.4,
        slice_offset=1,
        output=None,
        indent=0,
        tags=[],
        verbose=False,
    )
    logger = logging.getLogger("test")

    rc = header_command(args, logger)
    assert rc == 0
    captured = capsys.readouterr().out
    assert '"Value": 3' in captured or '"Value":3' in captured


def test_header_command_filters_tags(monkeypatch, tmp_path, caplog):
    class DummySeriesIndex:
        def __init__(self, *args, **kwargs):
            self.index_dataframe = pl.DataFrame(
                {
                    "IndexNormalized": [0.0, 0.5],
                    "SeriesUID": ["one", "two"],
                    "WindowWidth": [1200, 1100],
                    "WindowCenter": [600, 550],
                }
            )

    monkeypatch.setattr("idc_series_preview.api.SeriesIndex", DummySeriesIndex)

    output_path = tmp_path / "tags.json"
    args = SimpleNamespace(
        seriesuid="series",
        root="s3://bucket",
        cache_dir=None,
        no_cache=False,
        position=0.4,
        slice_offset=0,
        output=str(output_path),
        indent=2,
        tags=["WindowWidth", "MissingTag"],
        verbose=True,
    )
    logger = logging.getLogger("test")

    with caplog.at_level(logging.WARNING):
        rc = header_command(args, logger)
    assert rc == 0
    data = json.loads(output_path.read_text())
    assert list(data.keys()) == ["WindowWidth"]
    assert data["WindowWidth"] == 1100
    assert "MissingTag" in caplog.text


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


def test_video_command_streams_slices_to_ffmpeg(monkeypatch, tmp_path):
    requested = []

    class DummySeriesIndex:
        def __init__(self, seriesuid, root, cache_dir=None, use_cache=True):
            self.instance_count = 3
            self.index_dataframe = pl.DataFrame(
                {
                    "Index": [0, 1, 2],
                    "IndexNormalized": [0.0, 0.5, 1.0],
                }
            )

        def get_instances(self, slice_numbers, max_workers=None, headers_only=False):
            requested.append(list(slice_numbers))
            return [
                SimpleNamespace(instance_uid=f"uid{idx}", dataset=SimpleNamespace(value=idx))
                for idx in slice_numbers
            ]

    class DummyRenderer:
        def __init__(self, image_width, window_settings):
            pass

        def render_instance(self, dataset):
            color = int(dataset.value) * 40
            return Image.new("RGB", (2, 1), color=(color, 0, 0))

    class RecordingPipe:
        def __init__(self):
            self.frames = []
            self.closed = False

        def write(self, data):
            self.frames.append(data)

        def close(self):
            self.closed = True

    class RecordingProcess:
        def __init__(self):
            self.stdin_pipe = RecordingPipe()
            self.stdin = self.stdin_pipe
            self.returncode = 0

        def communicate(self):
            return (b"", b"")

    ffmpeg_calls = []

    def fake_start_ffmpeg_process(width, height, fps, output, crf):
        process = RecordingProcess()
        ffmpeg_calls.append(
            {
                "width": width,
                "height": height,
                "fps": fps,
                "output": output,
                "crf": crf,
                "process": process,
            }
        )
        return process

    monkeypatch.setattr("idc_series_preview.api.SeriesIndex", DummySeriesIndex)
    monkeypatch.setattr("idc_series_preview.image_utils.InstanceRenderer", DummyRenderer)
    monkeypatch.setattr("idc_series_preview.cli_core._start_ffmpeg_process", fake_start_ffmpeg_process)

    args = SimpleNamespace(
        seriesuid="series",
        output=str(tmp_path / "video.mp4"),
        root="s3://bucket",
        image_width=64,
        contrast=None,
        verbose=False,
        cache_dir=None,
        no_cache=False,
        start=0.0,
        end=1.0,
        fps=24.0,
        frames=None,
        quality=DEFAULT_IMAGE_QUALITY,
    )
    logger = logging.getLogger("test")

    rc = video_command(args, logger)
    assert rc == 0
    assert requested == [[0, 1, 2]]
    assert ffmpeg_calls
    call = ffmpeg_calls[0]
    assert call["width"] == 2
    assert call["height"] == 1
    assert call["fps"] == 24.0
    assert call["crf"] == _map_quality_to_crf(DEFAULT_IMAGE_QUALITY)
    frames = call["process"].stdin_pipe.frames
    assert len(frames) == 3
    assert [frame[0] for frame in frames] == [0, 40, 80]

    requested.clear()
    ffmpeg_calls.clear()
    args.quality = 0
    rc = video_command(args, logger)
    assert rc == 0
    assert ffmpeg_calls
    assert ffmpeg_calls[0]["crf"] == _map_quality_to_crf(0)


def test_video_command_rejects_invalid_fps(monkeypatch, tmp_path):
    class FailSeriesIndex:
        def __init__(self, *args, **kwargs):
            raise AssertionError("SeriesIndex should not be instantiated for invalid fps")

    monkeypatch.setattr("idc_series_preview.api.SeriesIndex", FailSeriesIndex)

    args = SimpleNamespace(
        seriesuid="series",
        output=str(tmp_path / "video.mp4"),
        root="s3://bucket",
        image_width=64,
        contrast=None,
        verbose=False,
        cache_dir=None,
        no_cache=False,
        start=0.0,
        end=1.0,
        fps=0.0,
        frames=None,
        quality=DEFAULT_IMAGE_QUALITY,
    )
    logger = logging.getLogger("test")

    rc = video_command(args, logger)
    assert rc == 1


def test_video_command_rejects_invalid_quality(tmp_path):
    args = SimpleNamespace(
        seriesuid="series",
        output=str(tmp_path / "video.mp4"),
        root="s3://bucket",
        image_width=64,
        contrast=None,
        verbose=False,
        cache_dir=None,
        no_cache=False,
        start=0.0,
        end=1.0,
        fps=24.0,
        frames=None,
        quality=120,
    )
    logger = logging.getLogger("test")

    rc = video_command(args, logger)
    assert rc == 1


def test_video_command_supports_frames_option(monkeypatch, tmp_path):
    calls = []

    class DummySeriesIndex:
        def __init__(self, *args, **kwargs):
            self.instance_count = 5

        def get_instances(self, **kwargs):
            calls.append(kwargs)
            return [
                SimpleNamespace(instance_uid=f"uid{i}", dataset=SimpleNamespace(value=i))
                for i in range(len(kwargs.get("positions") or kwargs.get("slice_numbers")))
            ]

    class DummyRenderer:
        def __init__(self, *args, **kwargs):
            pass

        def render_instance(self, dataset):
            return Image.new("RGB", (2, 2), color=(dataset.value, 0, 0))

    def fake_start_ffmpeg_process(*args, **kwargs):
        class P:
            def __init__(self):
                self.stdin = SimpleNamespace(write=lambda data: None, close=lambda: None)
                self.returncode = 0

            def communicate(self):
                return (b"", b"")

        return P()

    class DummyInterpolator:
        def __init__(self, _count):
            pass

        def interpolate_unique(self, count, start, end):
            return ([start, end][:count], 0)

    monkeypatch.setattr("idc_series_preview.api.SeriesIndex", DummySeriesIndex)
    monkeypatch.setattr("idc_series_preview.image_utils.InstanceRenderer", DummyRenderer)
    monkeypatch.setattr("idc_series_preview.cli_core._start_ffmpeg_process", fake_start_ffmpeg_process)
    monkeypatch.setattr("idc_series_preview.api.PositionInterpolator", DummyInterpolator)

    args = SimpleNamespace(
        seriesuid="series",
        output=str(tmp_path / "video.mp4"),
        root="s3://bucket",
        image_width=64,
        contrast=None,
        verbose=False,
        cache_dir=None,
        no_cache=False,
        start=0.0,
        end=1.0,
        fps=24.0,
        frames=2,
        quality=80,
    )
    logger = logging.getLogger("test")

    rc = video_command(args, logger)
    assert rc == 0
    assert calls
    assert "positions" in calls[0]
    assert calls[0]["positions"] == [0.0, 1.0]


def test_video_command_rejects_invalid_frame_count(tmp_path):
    args = SimpleNamespace(
        seriesuid="series",
        output=str(tmp_path / "video.mp4"),
        root="s3://bucket",
        image_width=64,
        contrast=None,
        verbose=False,
        cache_dir=None,
        no_cache=False,
        start=0.0,
        end=1.0,
        fps=24.0,
        frames=0,
        quality=DEFAULT_IMAGE_QUALITY,
    )
    logger = logging.getLogger("test")

    rc = video_command(args, logger)
    assert rc == 1


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
