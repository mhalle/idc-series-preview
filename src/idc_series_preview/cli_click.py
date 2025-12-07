"""Click-based command-line interface for idc-series-preview."""

from __future__ import annotations

import logging
from argparse import Namespace
from typing import Any, Callable, Optional

import click

from .constants import (
    DEFAULT_IMAGE_QUALITY,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_MOSAIC_TILE_SIZE,
)
from .cli_core import (
    setup_logging,
    mosaic_command,
    image_command,
    header_command,
    video_command,
    contrast_mosaic_command,
    build_index_command,
    get_index_command,
    clear_index_command,
)


CommandCallable = Callable[[Namespace, logging.Logger], int]


def _invoke_command(func: CommandCallable, **kwargs: Any) -> None:
    """Invoke existing argparse command and map errors to Click exceptions."""
    args = Namespace(**kwargs)
    setup_logging(getattr(args, "verbose", False))
    logger = logging.getLogger(__name__)
    rc = func(args, logger)
    if rc != 0:
        raise click.ClickException(f"{func.__name__} failed with exit code {rc}")


def common_options(
    *,
    include_contrast: bool = True,
    image_width_default: Optional[int] = DEFAULT_IMAGE_WIDTH,
    include_quality: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator adding shared root/image/quality/verbose options."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        func = click.option(
            "-v",
            "--verbose",
            is_flag=True,
            help="Enable detailed logging",
        )(func)
        if include_quality:
            func = click.option(
                "-q",
                "--quality",
                type=int,
                default=DEFAULT_IMAGE_QUALITY,
                show_default=True,
                help="Output image quality 0-100. Recommended 70+ for JPEG",
            )(func)
        if include_contrast:
            func = click.option(
                "-c",
                "--contrast",
                help="Contrast preset ('ct-lung', 'bone', 'auto', 'embedded', or custom WW/WL such as '1500/500')",
            )(func)
        func = click.option(
            "-w",
            "--image-width",
            type=int,
            default=image_width_default,
            show_default=image_width_default is not None,
            help="Width of each output image in pixels. Height scales proportionally.",
        )(func)
        func = click.option(
            "--root",
            default="s3://idc-open-data",
            show_default=True,
            help="Root path for DICOM files (S3, HTTP, or local path).",
        )(func)
        return func

    return decorator


def cache_options(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator adding cache directory / no-cache flags."""
    func = click.option(
        "--no-cache",
        is_flag=True,
        default=False,
        help="Disable header index caching.",
    )(func)
    func = click.option(
        "--cache-dir",
        type=click.Path(path_type=str),
        help="Directory to store/load cached indices.",
    )(func)
    return func


def _validate_cache_flags(cache_dir: Optional[str], no_cache: bool) -> None:
    if cache_dir and no_cache:
        raise click.BadOptionUsage(
            option_name="--cache-dir",
            message="--cache-dir cannot be combined with --no-cache",
        )


@click.group()
def cli() -> None:
    """Preview DICOM series stored on S3, HTTP, or local files using Click CLI."""


@cli.command("mosaic")
@click.argument("seriesuid")
@click.argument("output")
@common_options()
@cache_options
@click.option(
    "-s",
    "--start",
    type=float,
    default=0.0,
    show_default=True,
    help="Start of normalized z-position range (0.0-1.0).",
)
@click.option(
    "-e",
    "--end",
    type=float,
    default=1.0,
    show_default=True,
    help="End of normalized z-position range (0.0-1.0).",
)
@click.option(
    "-t",
    "--tile-width",
    type=int,
    default=DEFAULT_MOSAIC_TILE_SIZE,
    show_default=True,
    help="Number of images per row in the mosaic.",
)
@click.option(
    "--tile-height",
    type=int,
    default=None,
    help="Number of images per column in the mosaic. Defaults to tile width.",
)
def mosaic_click(
    *,
    seriesuid: str,
    output: str,
    root: str,
    image_width: Optional[int],
    contrast: Optional[str],
    quality: int,
    verbose: bool,
    start: float,
    end: float,
    tile_width: int,
    tile_height: Optional[int],
    cache_dir: Optional[str],
    no_cache: bool,
) -> None:
    """Generate a tiled mosaic grid from a DICOM series."""
    _validate_cache_flags(cache_dir, no_cache)
    _invoke_command(
        mosaic_command,
        seriesuid=seriesuid,
        output=output,
        root=root,
        image_width=image_width,
        contrast=contrast,
        quality=quality,
        verbose=verbose,
        start=start,
        end=end,
        tile_width=tile_width,
        tile_height=tile_height,
        cache_dir=cache_dir,
        no_cache=no_cache,
    )


@cli.command("image")
@click.argument("seriesuid")
@click.argument("output")
@common_options(image_width_default=None)
@cache_options
@click.option(
    "-p",
    "--position",
    type=float,
    required=True,
    help="Normalized z-position (0.0-1.0) of the slice to extract.",
)
@click.option(
    "--slice-offset",
    type=int,
    default=0,
    show_default=True,
    help="Offset from --position by number of slices (e.g., 1 for next slice).",
)
def image_click(
    *,
    seriesuid: str,
    output: str,
    root: str,
    image_width: Optional[int],
    contrast: Optional[str],
    quality: int,
    verbose: bool,
    cache_dir: Optional[str],
    no_cache: bool,
    position: float,
    slice_offset: int,
) -> None:
    """Extract a single image from a DICOM series at a specific position."""
    _validate_cache_flags(cache_dir, no_cache)
    _invoke_command(
        image_command,
        seriesuid=seriesuid,
        output=output,
        root=root,
        image_width=image_width,
        contrast=contrast,
        quality=quality,
        verbose=verbose,
        cache_dir=cache_dir,
        no_cache=no_cache,
        position=position,
        slice_offset=slice_offset,
    )


@cli.command("header")
@click.argument("seriesuid")
@cache_options
@click.option(
    "--root",
    default="s3://idc-open-data",
    show_default=True,
    help="Root path for DICOM files (S3, HTTP, or local path).",
)
@click.option(
    "-p",
    "--position",
    type=float,
    required=True,
    help="Normalized z-position (0.0-1.0) used to select the instance.",
)
@click.option(
    "--slice-offset",
    type=int,
    default=0,
    show_default=True,
    help="Offset from --position by number of slices.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=str),
    help="Optional JSON file destination; defaults to stdout.",
)
@click.option(
    "--indent",
    type=int,
    default=2,
    show_default=True,
    help="JSON indentation (0 disables pretty printing).",
)
@click.option(
    "-t",
    "--tag",
    multiple=True,
    help="Restrict output to specific DICOM keywords (repeatable).",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable detailed logging.",
)
def header_click(
    *,
    seriesuid: str,
    root: str,
    cache_dir: Optional[str],
    no_cache: bool,
    position: float,
    slice_offset: int,
    output: Optional[str],
    indent: int,
    tag: tuple[str, ...],
    verbose: bool,
) -> None:
    """Print header metadata for a single instance selected by position."""
    _validate_cache_flags(cache_dir, no_cache)
    _invoke_command(
        header_command,
        seriesuid=seriesuid,
        root=root,
        cache_dir=cache_dir,
        no_cache=no_cache,
        position=position,
        slice_offset=slice_offset,
        output=output,
        indent=indent,
        tags=list(tag),
        verbose=verbose,
    )


@cli.command("video")
@click.argument("seriesuid")
@click.argument("output")
@common_options(include_quality=False, image_width_default=None)
@cache_options
@click.option(
    "-s",
    "--start",
    type=float,
    default=0.0,
    show_default=True,
    help="Start of normalized z-position range (0.0-1.0).",
)
@click.option(
    "-e",
    "--end",
    type=float,
    default=1.0,
    show_default=True,
    help="End of normalized z-position range (0.0-1.0).",
)
@click.option(
    "--fps",
    type=float,
    default=24.0,
    show_default=True,
    help="Frames per second for the output video.",
)
@click.option(
    "--frames",
    type=int,
    help="Sample exactly this many slices evenly spaced within the range.",
)
@click.option(
    "-q",
    "--quality",
    type=click.IntRange(0, 100),
    default=DEFAULT_IMAGE_QUALITY,
    show_default=True,
    help="Video quality 0-100 (higher is better). Internally maps to libx264 CRF 40-10.",
)
def video_click(
    *,
    seriesuid: str,
    output: str,
    root: str,
    image_width: Optional[int],
    contrast: Optional[str],
    verbose: bool,
    cache_dir: Optional[str],
    no_cache: bool,
    start: float,
    end: float,
    fps: float,
    frames: Optional[int],
    quality: int,
) -> None:
    """Render the slices of a DICOM series into an MP4 video."""
    _validate_cache_flags(cache_dir, no_cache)
    _invoke_command(
        video_command,
        seriesuid=seriesuid,
        output=output,
        root=root,
        image_width=image_width,
        contrast=contrast,
        verbose=verbose,
        cache_dir=cache_dir,
        no_cache=no_cache,
        start=start,
        end=end,
        fps=fps,
        frames=frames,
        quality=quality,
    )


@cli.command("contrast-mosaic")
@click.argument("seriesuid")
@click.argument("output")
@common_options(include_contrast=False)
@cache_options
@click.option(
    "-c",
    "--contrast",
    multiple=True,
    required=True,
    help="Contrast settings (repeatable) such as 'ct-lung', 'auto', or '1500/500'.",
)
@click.option(
    "-p",
    "--position",
    type=float,
    help="Normalized z-position for single-instance mode (0.0-1.0).",
)
@click.option(
    "--slice-offset",
    type=int,
    default=0,
    show_default=True,
    help="Slice offset applied when --position is used.",
)
@click.option(
    "-s",
    "--start",
    type=float,
    help="Start of normalized z-position range (range mode only).",
)
@click.option(
    "-e",
    "--end",
    type=float,
    help="End of normalized z-position range (range mode only).",
)
@click.option(
    "-t",
    "--tile-height",
    type=int,
    default=2,
    show_default=True,
    help="Number of instances per column (y-axis) when sampling a range.",
)
def contrast_mosaic_click(
    *,
    seriesuid: str,
    output: str,
    root: str,
    image_width: Optional[int],
    quality: int,
    verbose: bool,
    cache_dir: Optional[str],
    no_cache: bool,
    contrast: tuple[str, ...],
    position: Optional[float],
    slice_offset: int,
    start: Optional[float],
    end: Optional[float],
    tile_height: int,
) -> None:
    """Create a grid of a DICOM instance under multiple contrast settings."""
    _validate_cache_flags(cache_dir, no_cache)
    _invoke_command(
        contrast_mosaic_command,
        seriesuid=seriesuid,
        output=output,
        root=root,
        image_width=image_width,
        quality=quality,
        verbose=verbose,
        cache_dir=cache_dir,
        no_cache=no_cache,
        contrast=list(contrast),
        position=position,
        slice_offset=slice_offset,
        start=start,
        end=end,
        tile_height=tile_height,
    )


@cli.command("build-index")
@click.argument("series", nargs=-1)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=str),
    help="Output directory for a single series index.",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=str),
    help="Cache directory for multiple series (default platform cache).",
)
@click.option(
    "--root",
    default="s3://idc-open-data",
    show_default=True,
    help="Root path for DICOM files.",
)
@click.option(
    "--rebuild",
    is_flag=True,
    help="Force regeneration of each series index.",
)
@click.option(
    "--all",
    is_flag=True,
    help="With --rebuild, rebuild every cached index located in the cache directory.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable detailed logging",
)
def build_index_click(
    *,
    series: tuple[str, ...],
    output: Optional[str],
    cache_dir: Optional[str],
    root: str,
    verbose: bool,
    rebuild: bool,
    all: bool,
) -> None:
    """Build DICOM series indices (cached headers for fast access)."""
    _invoke_command(
        build_index_command,
        series=list(series),
        output=output,
        cache_dir=cache_dir,
        root=root,
        rebuild=rebuild,
        all=all,
        verbose=verbose,
    )


@cli.command("get-index")
@click.argument("series")
@click.argument("output", required=False)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=str),
    help="Directory to store/load index files.",
)
@click.option(
    "--root",
    default="s3://idc-open-data",
    show_default=True,
    help="Root path for DICOM files.",
)
@click.option(
    "--rebuild",
    is_flag=True,
    help="Force regeneration even when cached index exists.",
)
@click.option(
    "--format",
    type=click.Choice(sorted(["parquet", "json", "jsonl"])),
    help="Explicit export format when OUTPUT is provided.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable detailed logging",
)
def get_index_click(
    *,
    series: str,
    output: Optional[str],
    cache_dir: Optional[str],
    root: str,
    verbose: bool,
    rebuild: bool,
    format: Optional[str],
) -> None:
    """Get or create a DICOM series index and return its path."""
    _invoke_command(
        get_index_command,
        series=series,
        output=output,
        cache_dir=cache_dir,
        root=root,
        rebuild=rebuild,
        format=format,
        verbose=verbose,
    )


@cli.command("clear-index")
@click.argument("series", nargs=-1)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=str),
    help="Cache directory containing index files.",
)
@click.option(
    "--root",
    default="s3://idc-open-data",
    show_default=True,
    help="Root path used when resolving series inputs.",
)
@click.option(
    "--all",
    is_flag=True,
    help="Remove every cached index from the cache directory.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable detailed logging",
)
def clear_index_click(
    *,
    series: tuple[str, ...],
    cache_dir: Optional[str],
    root: str,
    verbose: bool,
    all: bool,
) -> None:
    """Delete cached index files for specific series or all cached entries."""
    _invoke_command(
        clear_index_command,
        series=list(series),
        cache_dir=cache_dir,
        root=root,
        all=all,
        verbose=verbose,
    )


__all__ = ["cli"]
