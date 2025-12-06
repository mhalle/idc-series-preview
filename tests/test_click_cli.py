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
