#!/usr/bin/env python3
"""
DICOM Series Preview

Preview DICOM series stored on S3, HTTP, or local files with intelligent sampling
and visualization. Supports both tiled mosaics and individual image extraction.
"""

import logging
import math
import shutil
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

            renderer = InstanceRenderer(image_width=args.image_width, window_settings=window_settings)
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
