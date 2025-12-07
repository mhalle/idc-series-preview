from click.testing import CliRunner

from idc_series_preview.cli_click import cli


def test_mosaic_cli_invokes_core(monkeypatch):
    captured = {}

    def fake_mosaic(args, logger):
        captured["seriesuid"] = args.seriesuid
        captured["tile_width"] = args.tile_width
        return 0

    monkeypatch.setattr("idc_series_preview.cli_click.mosaic_command", fake_mosaic)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "mosaic",
            "38902e14-b11f-4548-910e-771ee757dc82",
            "out.webp",
            "--root",
            "s3://custom",
            "--tile-width",
            "4",
        ],
    )

    assert result.exit_code == 0
    assert captured["seriesuid"] == "38902e14-b11f-4548-910e-771ee757dc82"
    assert captured["tile_width"] == 4


def test_mosaic_cli_maps_nonzero_exit_to_click_exception(monkeypatch):
    def fake_mosaic(args, logger):
        return 2

    monkeypatch.setattr("idc_series_preview.cli_click.mosaic_command", fake_mosaic)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["mosaic", "series", "out.webp"],
    )

    assert result.exit_code == 1
    assert "failed with exit code 2" in result.output


def test_cache_options_conflict(monkeypatch):
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "mosaic",
            "series",
            "out.webp",
            "--cache-dir",
            "/tmp/cache",
            "--no-cache",
        ],
    )

    assert result.exit_code == 2
    assert "--cache-dir cannot be combined with --no-cache" in result.output


def test_video_cli_invokes_core(monkeypatch):
    captured = {}

    def fake_video(args, logger):
        captured["fps"] = args.fps
        captured["start"] = args.start
        captured["output"] = args.output
        captured["frames"] = args.frames
        captured["quality"] = args.quality
        return 0

    monkeypatch.setattr("idc_series_preview.cli_click.video_command", fake_video)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "video",
            "series",
            "video.mp4",
            "--fps",
            "30",
            "--start",
            "0.1",
            "--end",
            "0.9",
            "--frames",
            "12",
        ],
    )

    assert result.exit_code == 0
    assert captured["fps"] == 30
    assert captured["start"] == 0.1
    assert captured["output"] == "video.mp4"
    assert captured["frames"] == 12
    assert captured["quality"] == 60


def test_video_cli_accepts_quality_option(monkeypatch):
    captured = {}

    def fake_video(args, logger):
        captured["quality"] = args.quality
        return 0

    monkeypatch.setattr("idc_series_preview.cli_click.video_command", fake_video)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "video",
            "series",
            "video.mp4",
            "--quality",
            "18",
        ],
    )

    assert result.exit_code == 0
    assert captured["quality"] == 18


def test_header_cli_invokes_core(monkeypatch):
    captured = {}

    def fake_header(args, logger):
        captured["position"] = args.position
        captured["slice_offset"] = args.slice_offset
        captured["output"] = args.output
        captured["tags"] = args.tags
        return 0

    monkeypatch.setattr("idc_series_preview.cli_click.header_command", fake_header)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "header",
            "series",
            "-p",
            "0.5",
            "--slice-offset",
            "1",
            "--output",
            "header.json",
            "--tag",
            "SeriesUID",
            "--tag",
            "WindowWidth",
        ],
    )

    assert result.exit_code == 0
    assert captured["position"] == 0.5
    assert captured["slice_offset"] == 1
    assert captured["output"] == "header.json"
    assert captured["tags"] == ["SeriesUID", "WindowWidth"]
