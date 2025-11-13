#!/usr/bin/env python3
"""
DICOM Series Preview

Preview DICOM series stored on S3, HTTP, or local files with intelligent sampling
and visualization. Supports both tiled mosaics and individual image extraction.
"""

import argparse
import sys
import logging
import math
import shutil
import json
from pathlib import Path

from .image_utils import MosaicRenderer, save_image
from .contrast import ContrastPresets
from .index_cache import get_cache_directory
from .constants import DEFAULT_IMAGE_WIDTH, DEFAULT_MOSAIC_TILE_SIZE, DEFAULT_IMAGE_QUALITY
from .workers import optimal_workers


SUPPORTED_INDEX_FORMATS = {"parquet", "json", "jsonl"}


def setup_logging(verbose=False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.WARNING
    format_str = '%(levelname)s: %(message)s' if not verbose else '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=level,
        format=format_str
    )


def add_common_arguments(parser):
    """Add shared arguments to a parser (series, output, root, contrast, quality, verbose)."""
    parser.add_argument(
        "seriesuid",
        help="DICOM Series UID or full path. Can be: series UID (e.g., 38902e14-b11f-4548-910e-771ee757dc82), "
             "partial UID prefix (e.g., 38902e14*, 389*), or full path (e.g., s3://idc-open-data/38902e14-b11f-4548-910e-771ee757dc82). "
             "Full paths override --root parameter."
    )
    parser.add_argument(
        "output",
        help="Output image path (.webp or .jpg)"
    )

    # Storage arguments
    parser.add_argument(
        "--root",
        default="s3://idc-open-data",
        help="Root path for DICOM files (S3, HTTP, or local path). Default: s3://idc-open-data"
    )

    # Image scaling (shared)
    parser.add_argument(
        "-w", "--image-width",
        type=int,
        default=DEFAULT_IMAGE_WIDTH,
        help=f"Width of each image tile in pixels. Height will be proportionally scaled. Default: {DEFAULT_IMAGE_WIDTH}"
    )

    # Contrast parameters (shared)
    parser.add_argument(
        "-c", "--contrast",
        help="Contrast settings: CT preset (ct-lung, ct-bone, ct-brain, ct-abdomen, ct-liver, ct-mediastinum, ct-soft-tissue), "
             "MR preset (mr-t1, mr-t2, mr-proton), legacy alias (lung, bone, brain, etc.), "
             "shortcut (soft for ct-soft-tissue, media for ct-mediastinum, t1/t2/proton for MR), "
             "'auto' for auto-detection, 'embedded' for DICOM file window/level, "
             "or custom window/level values (e.g., '1500/500' or '1500,-500'). "
             "Supports both slash (medical standard) and comma separators."
    )

    # Output format arguments
    parser.add_argument(
        "-q", "--quality",
        type=int,
        default=DEFAULT_IMAGE_QUALITY,
        help=f"Output image quality 0-100. Default: {DEFAULT_IMAGE_QUALITY} for WebP, 70+ recommended for JPEG"
    )

    # Utility arguments
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable detailed logging"
    )


def add_range_arguments(parser):
    """Add range selection arguments (--start, --end)."""
    parser.add_argument(
        "-s", "--start",
        type=float,
        default=0.0,
        help="Start of normalized z-position range (0.0-1.0). Default: 0.0 (beginning of series)"
    )
    parser.add_argument(
        "-e", "--end",
        type=float,
        default=1.0,
        help="End of normalized z-position range (0.0-1.0). Default: 1.0 (end of series)"
    )


def add_cache_arguments(parser):
    """Add caching arguments (--cache-dir, --no-cache)."""
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--cache-dir",
        metavar="PATH",
        help="Directory to store/load DICOM series index files. "
             "Overrides default cache location. Index files are stored in "
             "{CACHE_DIR}/indices/{SERIESUID}_index.parquet and loaded/generated automatically."
    )
    cache_group.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable index caching. Fetches headers fresh from storage on every run. "
             "By default, caching is enabled using IDC_SERIES_PREVIEW_CACHE_DIR env var "
             "or platform-specific cache directory."
    )


def _split_format_prefix(output_value: str):
    """Return (format, path) if the value uses format:path syntax."""
    if not output_value:
        return None, None

    if ":" not in output_value:
        return None, output_value

    prefix, path = output_value.split(":", 1)
    if prefix in SUPPORTED_INDEX_FORMATS and path:
        return prefix, path

    return None, output_value


def _infer_format_from_suffix(path: Path):
    """Infer index export format based on file suffix."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return "parquet"
    if suffix == ".json":
        return "json"
    if suffix in (".jsonl", ".ndjson"):
        return "jsonl"
    return None


def _resolve_get_index_output(args, logger):
    """Determine requested format/path for get-index output."""
    output_value = getattr(args, "output", None)
    format_arg = getattr(args, "format", None)

    if not output_value:
        if format_arg:
            logger.error("--format requires an output destination")
            return None
        return ("print", None)

    prefix_format, path_str = _split_format_prefix(output_value)
    output_path = Path(path_str)
    target_format = format_arg or prefix_format or _infer_format_from_suffix(output_path) or "parquet"

    if target_format not in SUPPORTED_INDEX_FORMATS:
        logger.error(f"Unsupported output format: {target_format}")
        return None

    return (target_format, output_path)


def parse_contrast_arg(contrast_str: str) -> dict | str | None:
    """
    Parse contrast argument which can be a preset name, "auto", "embedded", or window/level values.

    Args:
        contrast_str: Contrast specification (e.g., "lung", "soft", "auto", "embedded", "1500/500", "1500,-500")

    Returns:
        - Dict with 'window_width' and 'window_center' for custom values or presets
        - "auto" string for auto-detection from pixel statistics
        - "embedded" string for using DICOM file's window/level
        - Raises ValueError if format is invalid
    """
    contrast_str = contrast_str.strip()
    contrast_str_lower = contrast_str.lower()

    # Check if it's a special keyword
    if contrast_str_lower in ["auto", "embedded"]:
        return contrast_str_lower

    # Check if it's a preset or shortcut (case-insensitive)
    if contrast_str_lower in ContrastPresets.PRESETS.keys() or contrast_str_lower in ContrastPresets.SHORTCUTS.keys():
        return ContrastPresets.get_preset(contrast_str)

    # Check if it's window/level format (slash-separated, medical imaging standard)
    if "/" in contrast_str:
        try:
            parts = contrast_str.split("/")
            if len(parts) != 2:
                raise ValueError(
                    f"Window/level format requires exactly 2 values, got {len(parts)}"
                )
            window_width = float(parts[0].strip())
            window_level = float(parts[1].strip())
            return {
                "window_width": window_width,
                "window_center": window_level,
            }
        except ValueError as e:
            raise ValueError(f"Invalid window/level format: {e}")

    # Check if it's window,level format (comma-separated, alternative)
    if "," in contrast_str:
        try:
            parts = contrast_str.split(",")
            if len(parts) != 2:
                raise ValueError(
                    f"Window,level format requires exactly 2 values, got {len(parts)}"
                )
            window_width = float(parts[0].strip())
            window_level = float(parts[1].strip())
            return {
                "window_width": window_width,
                "window_center": window_level,
            }
        except ValueError as e:
            raise ValueError(f"Invalid window,level format: {e}")

    # Build list of valid options
    valid_presets = sorted(list(ContrastPresets.PRESETS.keys()))
    valid_shortcuts = sorted(list(ContrastPresets.SHORTCUTS.keys()))
    preset_list = ", ".join(valid_presets)
    shortcuts_list = ", ".join(valid_shortcuts)

    # Not a valid format
    raise ValueError(
        f"Invalid contrast specification: '{contrast_str}'. "
        f"Must be a preset ({preset_list}), shortcut ({shortcuts_list}), 'auto', 'embedded', "
        f"or window/level format (e.g., '1500/500' or '1500,-500' for negative values)"
    )


def _validate_output_format(output_path):
    """
    Validate output file format.

    Args:
        output_path: Path object for output file

    Returns:
        True if valid, False otherwise
    """
    return output_path.suffix.lower() in ['.webp', '.jpg', '.jpeg']


def _get_window_settings_from_args(args):
    """
    Extract window/center settings from command-line arguments.

    Supports:
    - Contrast preset (lung, bone, brain, etc.)
    - Custom window,level values (e.g., "1500,500")
    - Auto-detection ("auto")

    Args:
        args: Parsed command-line arguments

    Returns:
        Window settings dict, "auto" string, or None
    """
    if hasattr(args, "contrast") and args.contrast:
        try:
            return parse_contrast_arg(args.contrast)
        except ValueError as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Invalid contrast argument: {e}")
            return None
    return None


def mosaic_command(args, logger):
    """Generate a tiled mosaic from a DICOM series."""
    try:
        from .api import SeriesIndex, PositionInterpolator

        # Validate output format
        output_path = Path(args.output)
        if not _validate_output_format(output_path):
            logger.error("Output file must be .webp or .jpg/.jpeg")
            return 1

        # Validate range parameters
        if not (0.0 <= args.start <= 1.0):
            logger.error("--start must be between 0.0 and 1.0")
            return 1
        if not (0.0 <= args.end <= 1.0):
            logger.error("--end must be between 0.0 and 1.0")
            return 1
        if args.start > args.end:
            logger.error("--start must be less than or equal to --end")
            return 1

        # Determine tile dimensions
        tile_width = max(1, args.tile_width)
        if args.tile_height is not None and args.tile_height < 1:
            logger.error("--tile-height must be >= 1 when provided")
            return 1
        tile_height = args.tile_height if args.tile_height else tile_width

        if tile_width < 1:
            logger.error("--tile-width must be >= 1")
            return 1

        # Get window/center settings
        window_settings = _get_window_settings_from_args(args)

        if args.verbose:
            logger.info(f"Generating DICOM series mosaic")
            logger.info(f"Series UID: {args.seriesuid}")
            logger.info(f"Root: {args.root}")
            logger.info(f"Tile grid: {tile_width}x{tile_height}")
            logger.info(f"Tile width: {args.image_width}px")
            if args.start > 0.0 or args.end < 1.0:
                logger.info(f"Range: {args.start:.1%} to {args.end:.1%} of series")

        # Create SeriesIndex with optional cache support
        try:
            use_cache = not getattr(args, 'no_cache', False)
            cache_dir = getattr(args, 'cache_dir', None)
            series_index = SeriesIndex(
                args.seriesuid,
                root=args.root,
                cache_dir=cache_dir,
                use_cache=use_cache,
            )
        except ValueError as e:
            logger.error(f"Failed to initialize series index: {e}")
            return 1

        if args.verbose:
            logger.info(f"Series has {series_index.instance_count} instances")

        # Generate evenly-spaced positions for the mosaic, removing duplicates
        requested_tiles = tile_width * tile_height
        interp = PositionInterpolator(series_index.instance_count)
        positions, _ = interp.interpolate_unique(
            requested_tiles,
            start=args.start,
            end=args.end,
        )

        if not positions:
            logger.error(
                "Requested range did not produce any slices. Verify --start/--end."
            )
            return 1

        unique_count = len(positions)
        effective_tile_height = max(1, math.ceil(unique_count / tile_width))

        if args.verbose and unique_count < requested_tiles:
            logger.info(
                "Range %.1f%%-%.1f%% provides %d unique slices; "
                "shrinking mosaic height to %d rows"
                % (args.start * 100, args.end * 100, unique_count, effective_tile_height)
            )

        # Retrieve and render images
        if args.verbose:
            logger.info(f"Retrieving and rendering {len(positions)} images...")

        # Optimize worker count for this mosaic size (min 5, max 10)
        mosaic_workers = optimal_workers(len(positions), max_workers=10, min_workers=5)
        if args.verbose:
            logger.debug(f"Using {mosaic_workers} workers for {len(positions)} images")

        try:
            instances = series_index.get_instances(
                positions=positions,
                max_workers=mosaic_workers,
                headers_only=False,
            )
        except ValueError as e:
            logger.error(f"Failed to retrieve instances: {e}")
            return 1

        if not instances:
            logger.error("No instances retrieved for series")
            return 1

        from .image_utils import InstanceRenderer, MosaicRenderer

        renderer = InstanceRenderer(image_width=args.image_width, window_settings=window_settings)
        images = []
        for instance in instances:
            img = renderer.render_instance(instance.dataset)
            if img is None:
                logger.error(f"Failed to render image for instance {instance.instance_uid}")
                return 1
            images.append(img)

        if args.verbose:
            logger.info(f"Retrieved and rendered {len(images)} images")
            logger.info("Tiling images into mosaic...")
        generator = MosaicRenderer(
            tile_width=tile_width,
            tile_height=effective_tile_height,
            image_width=args.image_width,
        )

        output_image = generator.tile_images(images)

        if not output_image:
            logger.error("Failed to tile images")
            return 1

        # Save output
        if args.verbose:
            logger.info(f"Saving mosaic to {args.output}...")
        if not save_image(
            output_image,
            args.output,
            quality=args.quality
        ):
            logger.error("Failed to save mosaic")
            return 1

        if args.verbose:
            logger.info("Done!")
        return 0

    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1


def image_command(args, logger):
    """Extract a single image from a DICOM series at a specific position."""
    try:
        from .api import SeriesIndex

        # Validate output format
        output_path = Path(args.output)
        if not _validate_output_format(output_path):
            logger.error("Output file must be .webp or .jpg/.jpeg")
            return 1

        # Validate position parameter
        if not (0.0 <= args.position <= 1.0):
            logger.error("--position must be between 0.0 and 1.0")
            return 1

        # Validate slice-offset parameter
        if args.slice_offset != 0:
            if args.verbose:
                logger.info(f"Will apply slice offset: {args.slice_offset}")

        # Get window/center settings
        window_settings = _get_window_settings_from_args(args)

        if args.verbose:
            logger.info(f"Extracting single image from DICOM series")
            logger.info(f"Series UID: {args.seriesuid}")
            logger.info(f"Root: {args.root}")
            logger.info(f"Position: {args.position:.1%}")

        # Create SeriesIndex with optional cache support
        try:
            use_cache = not getattr(args, 'no_cache', False)
            cache_dir = getattr(args, 'cache_dir', None)
            series_index = SeriesIndex(
                args.seriesuid,
                root=args.root,
                cache_dir=cache_dir,
                use_cache=use_cache,
            )
        except ValueError as e:
            logger.error(f"Failed to initialize series index: {e}")
            return 1

        # Retrieve instance at position with optional offset
        if args.verbose:
            logger.info("Retrieving DICOM instance...")
        try:
            instance = series_index.get_instance(
                position=args.position,
                slice_offset=args.slice_offset
            )
        except ValueError as e:
            logger.error(f"Failed to retrieve instance: {e}")
            return 1

        if args.verbose:
            logger.info(f"Retrieved instance {instance.instance_uid}")

        # Generate and render image
        if args.verbose:
            logger.info("Generating image...")
        try:
            from .image_utils import InstanceRenderer

            image_width = args.image_width if args.image_width is not None else 2048
            renderer = InstanceRenderer(image_width=image_width, window_settings=window_settings)
            output_image = renderer.render_instance(instance.dataset)
        except ValueError as e:
            logger.error(f"Failed to generate image: {e}")
            return 1

        if not output_image:
            logger.error("Failed to generate image")
            return 1

        # Save output
        if args.verbose:
            logger.info(f"Saving image to {args.output}...")
        if not save_image(
            output_image,
            args.output,
            quality=args.quality
        ):
            logger.error("Failed to save image")
            return 1

        if args.verbose:
            logger.info("Done!")
        return 0

    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1


def contrast_mosaic_command(args, logger):
    """Generate a grid comparing a DICOM instance(s) under multiple contrast settings.

    Grid layout: contrasts on x-axis (columns), instances on y-axis (rows).
    """
    try:
        from .api import SeriesIndex, PositionInterpolator

        # Validate output format
        output_path = Path(args.output)
        if not _validate_output_format(output_path):
            logger.error("Output file must be .webp or .jpg/.jpeg")
            return 1

        # Validate mutually exclusive position vs range selection
        has_position = args.position is not None
        has_range = args.start is not None or args.end is not None

        if has_position and has_range:
            logger.error("Cannot use both --position and --start/--end together")
            return 1

        if not has_position and not has_range:
            logger.error("Must specify either --position or --start/--end")
            return 1

        # Validate contrast settings
        if not args.contrast or len(args.contrast) == 0:
            logger.error("At least one -c/--contrast setting is required")
            return 1

        parsed_contrasts = []
        for contrast_str in args.contrast:
            try:
                parsed = parse_contrast_arg(contrast_str)
                parsed_contrasts.append(parsed)
            except ValueError as e:
                logger.error(f"Invalid contrast argument: {e}")
                return 1

        num_contrasts = len(parsed_contrasts)

        # Create SeriesIndex with optional cache support
        try:
            use_cache = not getattr(args, 'no_cache', False)
            cache_dir = getattr(args, 'cache_dir', None)
            series_index = SeriesIndex(
                args.seriesuid,
                root=args.root,
                cache_dir=cache_dir,
                use_cache=use_cache,
            )
        except ValueError as e:
            logger.error(f"Failed to initialize series index: {e}")
            return 1

        # Determine instance selection mode
        instances = []
        tile_height = args.tile_height if args.tile_height and args.tile_height > 0 else 1

        if has_position:
            # Single position mode
            if not (0.0 <= args.position <= 1.0):
                logger.error("--position must be between 0.0 and 1.0")
                return 1

            if args.slice_offset != 0:
                if args.verbose:
                    logger.info(f"Will apply slice offset: {args.slice_offset}")

            tile_height = 1

            if args.verbose:
                logger.info(f"Generating contrast grid from DICOM series")
                logger.info(f"Series UID: {args.seriesuid}")
                logger.info(f"Root: {args.root}")
                logger.info(f"Position: {args.position:.1%}")
                logger.info(f"Contrast variations: {len(parsed_contrasts)}")

            # Retrieve single instance at position with optional offset
            if args.verbose:
                logger.info("Retrieving DICOM instance...")
            try:
                instance = series_index.get_instance(
                    position=args.position,
                    slice_offset=args.slice_offset
                )
                instances = [instance]
            except ValueError as e:
                logger.error(f"Failed to retrieve instance: {e}")
                return 1

            if args.verbose:
                logger.info(f"Retrieved instance {instances[0].instance_uid}")

        else:
            # Range selection mode
            if args.slice_offset != 0:
                logger.error("--slice-offset is not allowed with --start/--end")
                return 1

            # Validate range parameters
            if not (0.0 <= args.start <= 1.0):
                logger.error("--start must be between 0.0 and 1.0")
                return 1
            if not (0.0 <= args.end <= 1.0):
                logger.error("--end must be between 0.0 and 1.0")
                return 1
            if args.start > args.end:
                logger.error("--start must be less than or equal to --end")
                return 1

            if args.verbose:
                logger.info(f"Generating contrast grid from DICOM series")
                logger.info(f"Series UID: {args.seriesuid}")
                logger.info(f"Root: {args.root}")
                logger.info(f"Range: {args.start:.1%} to {args.end:.1%}")
                logger.info(f"Requested tile height (instances): {tile_height}")
                logger.info(f"Contrast variations: {len(parsed_contrasts)}")

            # Generate evenly-spaced positions across range, removing duplicates
            interp = PositionInterpolator(series_index.instance_count)
            positions, _ = interp.interpolate_unique(
                tile_height,
                start=args.start,
                end=args.end,
            )

            if not positions:
                logger.error(
                    "Requested range did not produce any slices. Verify --start/--end."
                )
                return 1

            effective_height = len(positions)
            if args.verbose and effective_height < tile_height:
                logger.info(
                    "Range %.1f%%-%.1f%% provides %d unique slices; "
                    "shrinking grid height to %d rows"
                    % (args.start * 100, args.end * 100, effective_height, effective_height)
                )

            # Retrieve instances at positions
            if args.verbose:
                logger.info("Retrieving DICOM instances...")
            try:
                instances = series_index.get_instances(positions=positions, headers_only=False)
            except ValueError as e:
                logger.error(f"Failed to retrieve instances: {e}")
                return 1

            if not instances:
                logger.error(f"No DICOM instances found in range {args.start:.1%}-{args.end:.1%}")
                return 1

            tile_height = len(instances)

            if args.verbose:
                logger.info(f"Retrieved {len(instances)} instances")

        from .image_utils import InstanceRenderer, MosaicRenderer

        if args.verbose:
            logger.info(f"Grid layout: {num_contrasts}x{tile_height} (contrasts x instances)")
            logger.info("Generating contrast grid...")

        image_grid = []
        for inst_idx, instance in enumerate(instances):
            for contrast_idx, contrast_settings in enumerate(parsed_contrasts):
                contrast_str = args.contrast[contrast_idx]
                if args.verbose:
                    logger.info(
                        f"Instance {inst_idx+1}/{len(instances)}, "
                        f"contrast {contrast_idx+1}/{num_contrasts}: {contrast_str}"
                    )

                renderer = InstanceRenderer(
                    image_width=args.image_width,
                    window_settings=contrast_settings,
                )
                img = renderer.render_instance(instance.dataset)
                if img is None:
                    logger.error(
                        f"Failed to render instance {inst_idx+1} contrast {contrast_str}"
                    )
                    return 1
                image_grid.append(img)

        generator = MosaicRenderer(
            tile_width=num_contrasts,
            tile_height=tile_height,
            image_width=args.image_width,
        )

        output_image = generator.tile_images(image_grid)

        if not output_image:
            logger.error("Failed to tile images into grid")
            return 1

        # Save output
        if args.verbose:
            logger.info(f"Saving contrast grid to {args.output}...")
        if not save_image(
            output_image,
            args.output,
            quality=args.quality
        ):
            logger.error("Failed to save contrast grid")
            return 1

        if args.verbose:
            logger.info("Done!")
        return 0

    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1


def get_index_command(args, logger):
    """Get or create a DICOM series index and return its path."""
    try:
        from .api import SeriesIndex

        # Determine cache directory (CLI override or default)
        cache_dir = Path(args.cache_dir) if getattr(args, 'cache_dir', None) else get_cache_directory()
        cache_dir_str = str(cache_dir)

        if args.verbose:
            logger.info(f"Ensuring index for series: {args.series}")
            logger.info(f"Root: {args.root}")
            logger.info(f"Cache directory: {cache_dir}")

        try:
            series_index = SeriesIndex(
                args.series,
                root=args.root,
                cache_dir=cache_dir_str,
                use_cache=True,
                force_rebuild=getattr(args, 'rebuild', False),
            )
        except ValueError as e:
            logger.error(f"Failed to initialize series index: {e}")
            return 1

        index_path = cache_dir / "indices" / f"{series_index.series_uid}_index.parquet"

        output_request = _resolve_get_index_output(args, logger)
        if output_request is None:
            return 1

        if output_request[0] == "print":
            if args.verbose:
                logger.info(f"Index ready: {index_path}")
            print(index_path)
            return 0

        export_format, export_path = output_request
        export_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if export_format == "parquet":
                shutil.copy2(index_path, export_path)
            else:
                df = series_index.index_dataframe
                if export_format == "json":
                    df.write_json(str(export_path))
                elif export_format == "jsonl":
                    df.write_ndjson(str(export_path))
        except Exception as e:
            logger.error(f"Failed to export index to {export_format}: {e}")
            return 1

        if args.verbose:
            logger.info(f"Index exported to {export_path} ({export_format})")
        print(export_path)
        return 0

    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1


def clear_index_command(args, logger):
    """Delete cached index files for specified series or all cached entries."""
    try:
        from .index_cache import iterate_cached_series, get_index_path
        from .series_spec import parse_and_normalize_series

        cache_dir = Path(args.cache_dir) if getattr(args, 'cache_dir', None) else get_cache_directory()

        if args.all:
            if args.series:
                logger.error("--all cannot be combined with explicit series arguments")
                return 1

            cached = iterate_cached_series(cache_dir)
            if not cached:
                logger.info("No cached indices found")
                return 0

            removed = 0
            for series_uid, path in cached:
                if path.exists():
                    path.unlink()
                    removed += 1
                    if args.verbose:
                        logger.info(f"Deleted {path}")

            logger.info(f"Removed {removed} cached indices from {cache_dir}")
            return 0

        if not args.series:
            logger.error("Specify one or more SERIES arguments or use --all")
            return 1

        removed = 0
        for series_spec in args.series:
            result = parse_and_normalize_series(series_spec, args.root, logger)
            if result is None:
                continue
            _, series_uid = result
            index_path = get_index_path(series_uid, cache_dir)
            if index_path.exists():
                index_path.unlink()
                removed += 1
                if args.verbose:
                    logger.info(f"Deleted {index_path}")
            else:
                logger.warning(f"No cached index found for {series_uid}")

        if removed == 0:
            logger.warning("No indices were removed")
        else:
            logger.info(f"Removed {removed} cached indices")
        return 0

    except Exception as e:
        logger.exception(f"Error clearing index cache: {e}")
        return 1


def _setup_mosaic_subcommand(subparsers):
    """
    Setup mosaic subcommand with all its arguments.

    Args:
        subparsers: The subparsers object from ArgumentParser
    """
    mosaic_parser = subparsers.add_parser(
        "mosaic",
        help="Generate a tiled mosaic grid from a DICOM series"
    )
    add_common_arguments(mosaic_parser)
    add_range_arguments(mosaic_parser)
    add_cache_arguments(mosaic_parser)
    mosaic_parser.add_argument(
        "-t", "--tile-width",
        type=int,
        default=DEFAULT_MOSAIC_TILE_SIZE,
        help=f"Number of images per row in mosaic. Default: {DEFAULT_MOSAIC_TILE_SIZE}"
    )
    mosaic_parser.add_argument(
        "--tile-height",
        type=int,
        help="Number of images per column in mosaic. Default: same as --tile-width"
    )
    mosaic_parser.set_defaults(func=mosaic_command)


def _setup_image_subcommand(subparsers):
    """
    Setup image subcommand with all its arguments.

    Args:
        subparsers: The subparsers object from ArgumentParser
    """
    image_parser = subparsers.add_parser(
        "image",
        help="Extract a single image from a DICOM series at a specific position"
    )

    # Add positional arguments
    image_parser.add_argument(
        "seriesuid",
        help="DICOM Series UID or full path. Can be: series UID (e.g., 38902e14-b11f-4548-910e-771ee757dc82), "
             "partial UID prefix (e.g., 38902e14*, 389*), or full path (e.g., s3://idc-open-data/38902e14-b11f-4548-910e-771ee757dc82). "
             "Full paths override --root parameter."
    )
    image_parser.add_argument(
        "output",
        help="Output image path (.webp or .jpg)"
    )

    # Storage arguments
    image_parser.add_argument(
        "--root",
        default="s3://idc-open-data",
        help="Root path for DICOM files (S3, HTTP, or local path). Default: s3://idc-open-data"
    )

    # Image scaling with short alias
    image_parser.add_argument(
        "-w", "--image-width",
        type=int,
        default=None,  # None means use natural size (no scaling)
        help="Width of image in pixels. Height will be proportionally scaled. "
             "Default: None (natural DICOM image size)"
    )

    # Contrast parameters
    image_parser.add_argument(
        "-c", "--contrast",
        help="Contrast settings: CT preset (ct-lung, ct-bone, ct-brain, ct-abdomen, ct-liver, ct-mediastinum, ct-soft-tissue), "
             "MR preset (mr-t1, mr-t2, mr-proton), legacy alias (lung, bone, brain, etc.), "
             "shortcut (soft for ct-soft-tissue, media for ct-mediastinum, t1/t2/proton for MR), "
             "'auto' for auto-detection, 'embedded' for DICOM file window/level, "
             "or custom window/level values (e.g., '1500/500' or '1500,-500'). "
             "Supports both slash (medical standard) and comma separators."
    )

    # Output format arguments
    image_parser.add_argument(
        "-q", "--quality",
        type=int,
        default=DEFAULT_IMAGE_QUALITY,
        help=f"Output image quality 0-100. Default: {DEFAULT_IMAGE_QUALITY} for WebP, 70+ recommended for JPEG"
    )

    # Utility arguments
    image_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable detailed logging"
    )

    # Cache arguments
    add_cache_arguments(image_parser)

    # Position arguments
    image_parser.add_argument(
        "-p", "--position",
        type=float,
        required=True,
        help="Extract image at normalized z-position (0.0-1.0). 0.0=superior, 1.0=inferior"
    )
    image_parser.add_argument(
        "--slice-offset",
        type=int,
        default=0,
        help="Offset from --position by number of slices (e.g., 1 for next slice, -1 for previous). Default: 0"
    )

    image_parser.set_defaults(func=image_command)


def _setup_contrast_mosaic_subcommand(subparsers):
    """
    Setup contrast-mosaic subcommand with all its arguments.

    Grid layout: contrasts on x-axis (columns), instances on y-axis (rows).
    Use either --position for single instance or --start/--end for range of instances.

    Args:
        subparsers: The subparsers object from ArgumentParser
    """
    contrast_parser = subparsers.add_parser(
        "contrast-mosaic",
        help="Create a grid of a DICOM instance(s) under multiple contrast settings (contrasts on x-axis, instances on y-axis)"
    )

    # Add positional arguments (series UID and output)
    contrast_parser.add_argument(
        "seriesuid",
        help="DICOM Series UID or full path. Can be: series UID (e.g., 38902e14-b11f-4548-910e-771ee757dc82), "
             "partial UID prefix (e.g., 38902e14*, 389*), or full path (e.g., s3://idc-open-data/38902e14-b11f-4548-910e-771ee757dc82). "
             "Full paths override --root parameter."
    )
    contrast_parser.add_argument(
        "output",
        help="Output image path (.webp or .jpg)"
    )

    # Storage arguments
    contrast_parser.add_argument(
        "--root",
        default="s3://idc-open-data",
        help="Root path for DICOM files (S3, HTTP, or local path). Default: s3://idc-open-data"
    )

    # Image sizing
    contrast_parser.add_argument(
        "-w", "--image-width",
        type=int,
        default=DEFAULT_IMAGE_WIDTH,
        help=f"Width of each image in pixels. Height will be proportionally scaled. Default: {DEFAULT_IMAGE_WIDTH}"
    )

    # Instance selection: position mode (single instance)
    contrast_parser.add_argument(
        "-p", "--position",
        type=float,
        help="Extract single image at normalized z-position (0.0-1.0). 0.0=superior, 1.0=inferior. "
             "Cannot be used with --start/--end."
    )
    contrast_parser.add_argument(
        "--slice-offset",
        type=int,
        default=0,
        help="Offset from --position by number of slices (only valid with --position). Default: 0"
    )

    # Instance selection: range mode (multiple instances)
    contrast_parser.add_argument(
        "-s", "--start",
        type=float,
        help="Start of normalized z-position range (0.0-1.0) for selecting multiple instances. "
             "Cannot be used with --position."
    )
    contrast_parser.add_argument(
        "-e", "--end",
        type=float,
        help="End of normalized z-position range (0.0-1.0) for selecting multiple instances. "
             "Cannot be used with --position."
    )

    # Vertical tiling (instances)
    contrast_parser.add_argument(
        "-t", "--tile-height",
        type=int,
        default=2,
        help="Number of instances per column (y-axis). Only used with --start/--end. Default: 2"
    )

    # Contrast settings (repeatable, always horizontal)
    contrast_parser.add_argument(
        "-c", "--contrast",
        action="append",
        help="Contrast settings (repeatable, x-axis): CT preset (ct-lung, ct-bone, ct-brain, ct-abdomen, ct-liver, ct-mediastinum, ct-soft-tissue), "
             "MR preset (mr-t1, mr-t2, mr-proton), legacy alias (lung, bone, brain, etc.), "
             "shortcut (soft, media, t1, t2, proton), 'auto', 'embedded', or custom window/level. "
             "Formats: '1500/500' (slash, medical standard) or '1500,500' (comma). "
             "Negative values supported (e.g., '1500/-500'). At least one -c/--contrast is required."
    )

    # Output format arguments
    contrast_parser.add_argument(
        "-q", "--quality",
        type=int,
        default=DEFAULT_IMAGE_QUALITY,
        help=f"Output image quality 0-100. Default: {DEFAULT_IMAGE_QUALITY} for WebP, 70+ recommended for JPEG"
    )

    # Index caching arguments
    add_cache_arguments(contrast_parser)

    # Utility arguments
    contrast_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable detailed logging"
    )

    contrast_parser.set_defaults(func=contrast_mosaic_command)


def build_index_command(args, logger):
    """
    Build DICOM series indices by capturing headers and saving to Parquet format.

    Can work in two modes:
    1. Multiple series with --cache-dir: saves to {cache_dir}/indices/{seriesuid}_index.parquet
    2. Single series with -o: saves to {output_dir}/indices/{seriesuid}_index.parquet

    Args:
        args: Parsed command arguments
        logger: Logger instance

    Returns:
        0 on success, 1 on error
    """
    try:
        from .api import SeriesIndex

        # Determine cache/output directory (either cache-dir, output dir, or default cache)
        if getattr(args, "output", None):
            output_dir = Path(args.output)
            if len(args.series) != 1:
                logger.error("When using -o, specify exactly one series")
                return 1
        elif getattr(args, "cache_dir", None):
            output_dir = Path(args.cache_dir)
        else:
            output_dir = get_cache_directory()

        cache_dir_str = str(output_dir)

        if args.rebuild and args.all:
            if args.series:
                logger.error("--all cannot be combined with explicit series arguments")
                return 1
            from .index_cache import iterate_cached_series

            cached = list(iterate_cached_series(Path(cache_dir_str)))
            if not cached:
                logger.error("No cached indices found to rebuild")
                return 1
            series_list = [series for series, _ in cached]
            if args.verbose:
                logger.info(f"Rebuilding {len(series_list)} cached indices")
        else:
            series_list = args.series
            if not series_list:
                logger.error("Specify at least one SERIES argument or use --rebuild --all")
                return 1

        if args.verbose:
            logger.info(f"Building indices for {len(series_list)} series")
            logger.info(f"Cache/output directory: {output_dir}")

        success_count = 0
        for series_spec in series_list:
            try:
                if args.verbose:
                    logger.info(f"Resolving and indexing series '{series_spec}'")

                series_index = SeriesIndex(
                    series_spec,
                    root=args.root,
                    cache_dir=cache_dir_str,
                    use_cache=True,
                    force_rebuild=getattr(args, 'rebuild', False),
                )

                index_path = output_dir / "indices" / f"{series_index.series_uid}_index.parquet"
                if args.verbose:
                    logger.info(
                        f"Index ready: {index_path} "
                        f"({series_index.instance_count} instances)"
                    )
                success_count += 1

            except Exception as e:
                logger.error(f"Error processing series {series_spec}: {e}")

        if success_count == 0:
            return 1

        logger.info(f"Successfully built {success_count}/{len(series_list)} indices")
        return 0

    except Exception as e:
        logger.exception(f"Error building indices: {e}")
        return 1


def _setup_build_index_subcommand(subparsers):
    """
    Setup build-index subcommand with all its arguments.

    Supports two modes:
    1. Multiple series with --cache-dir: idc-series-preview build-index SERIES1 SERIES2 ... --cache-dir /path
    2. Single series with -o: idc-series-preview build-index SERIES -o /output/dir

    Args:
        subparsers: The subparsers object from ArgumentParser
    """
    index_parser = subparsers.add_parser(
        "build-index",
        help="Build DICOM series indices (cached headers for fast access)"
    )

    # Positional argument: one or more series UIDs/paths
    index_parser.add_argument(
        "series",
        nargs="*",
        metavar="SERIES",
        help="DICOM Series UID(s) or path(s). Can be: series UID (e.g., 38902e14-b11f-4548-910e-771ee757dc82), "
             "partial UID prefix (e.g., 38902e14*, 389*), or full path (e.g., s3://idc-open-data/38902e14-b11f-4548-910e-771ee757dc82). "
             "Full paths override --root parameter. Required unless using --rebuild --all."
    )

    # Output options (mutually exclusive)
    output_group = index_parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "-o", "--output",
        metavar="DIR",
        help="Output directory for index file (single series only). "
             "Index will be saved as DIR/indices/{SERIESUID}_index.parquet"
    )
    output_group.add_argument(
        "--cache-dir",
        metavar="DIR",
        help="Cache directory for index files (multiple series). "
             "Indices will be saved as CACHE_DIR/indices/{SERIESUID}_index.parquet"
    )

    # Storage arguments
    index_parser.add_argument(
        "--root",
        default="s3://idc-open-data",
        help="Root path for DICOM files (S3, HTTP, or local path). Default: s3://idc-open-data"
    )

    index_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force regeneration of each series index even if it already exists in the cache."
    )

    index_parser.add_argument(
        "--all",
        action="store_true",
        help="When used with --rebuild (and no series arguments), rebuild every cached index in the cache directory."
    )

    # Utility arguments
    index_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable detailed logging"
    )

    index_parser.set_defaults(func=build_index_command)


def _setup_get_index_subcommand(subparsers):
    """
    Setup get-index subcommand to retrieve or build an index.

    Args:
        subparsers: The subparsers object from ArgumentParser
    """
    get_index_parser = subparsers.add_parser(
        "get-index",
        help="Get or create a DICOM series index and return its path"
    )

    # Positional argument: series UID/path
    get_index_parser.add_argument(
        "series",
        metavar="SERIES",
        help="DICOM Series UID or path. Can be: series UID (e.g., 38902e14-b11f-4548-910e-771ee757dc82), "
             "partial UID prefix (e.g., 38902e14*, 389*), or full path (e.g., s3://idc-open-data/38902e14-b11f-4548-910e-771ee757dc82). "
             "Full paths override --root parameter."
    )

    get_index_parser.add_argument(
        "output",
        nargs="?",
        metavar="OUTPUT",
        help="Optional destination for the index. Supports format prefixes such as 'jsonl:/tmp/out.jsonl' "
             "or paths whose extension implies the format (parquet, json, jsonl)."
    )

    # Storage arguments
    get_index_parser.add_argument(
        "--root",
        default="s3://idc-open-data",
        help="Root path for DICOM files (S3, HTTP, or local path). Default: s3://idc-open-data"
    )

    # Cache directory
    get_index_parser.add_argument(
        "--cache-dir",
        metavar="PATH",
        help="Directory to store/load DICOM series index files. "
             "Overrides default cache location. Index files are stored in "
             "{CACHE_DIR}/indices/{SERIESUID}_index.parquet and "
             "loaded/generated automatically."
    )

    get_index_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force regeneration of the series index even when a cached file already exists."
    )

    get_index_parser.add_argument(
        "--format",
        choices=sorted(SUPPORTED_INDEX_FORMATS),
        help="Explicit index export format when --output is provided. "
             "Overrides prefix/extension detection."
    )

    # Utility arguments
    get_index_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable detailed logging"
    )

    get_index_parser.set_defaults(func=get_index_command)


def _setup_clear_index_subcommand(subparsers):
    """Setup clear-index subcommand to delete cached indices."""
    clear_parser = subparsers.add_parser(
        "clear-index",
        help="Delete cached index files for specific series or all cached entries"
    )

    clear_parser.add_argument(
        "series",
        nargs="*",
        metavar="SERIES",
        help="Optional series UIDs/paths to remove from the cache"
    )

    clear_parser.add_argument(
        "--root",
        default="s3://idc-open-data",
        help="Root path used when resolving SERIES inputs"
    )

    clear_parser.add_argument(
        "--cache-dir",
        metavar="PATH",
        help="Cache directory containing index files (default: platform cache)"
    )

    clear_parser.add_argument(
        "--all",
        action="store_true",
        help="Remove every cached index from the cache directory"
    )

    clear_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable detailed logging"
    )

    clear_parser.set_defaults(func=clear_index_command)


def _setup_parser():
    """
    Setup and configure the main argument parser with all subcommands.

    Returns:
        Configured ArgumentParser with mosaic, image, and contrast-mosaic subcommands
    """
    parser = argparse.ArgumentParser(
        description="Preview DICOM series stored on S3, HTTP, or local files with intelligent sampling and visualization.",
        prog="idc-series-preview"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    subparsers.required = True

    # Setup subcommands
    _setup_mosaic_subcommand(subparsers)
    _setup_image_subcommand(subparsers)
    _setup_contrast_mosaic_subcommand(subparsers)
    _setup_build_index_subcommand(subparsers)
    _setup_get_index_subcommand(subparsers)
    _setup_clear_index_subcommand(subparsers)

    return parser


def main():
    parser = _setup_parser()
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    return args.func(args, logger)


if __name__ == "__main__":
    sys.exit(main())
