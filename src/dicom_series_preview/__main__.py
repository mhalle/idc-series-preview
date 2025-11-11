#!/usr/bin/env python3
"""
DICOM Series Preview

Preview DICOM series stored on S3, HTTP, or local files with intelligent sampling
and visualization. Supports both tiled mosaics and individual image extraction.
"""

import argparse
import sys
import logging
from pathlib import Path

from .image_utils import MosaicGenerator, save_image
from .contrast import ContrastPresets
from .index_cache import _generate_parquet_table, get_cache_directory
from .retriever import DICOMRetriever
from .series_spec import parse_and_normalize_series
from .constants import DEFAULT_IMAGE_WIDTH, DEFAULT_MOSAIC_TILE_SIZE, DEFAULT_IMAGE_QUALITY


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
        "--contrast",
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
             "By default, caching is enabled using DICOM_SERIES_PREVIEW_CACHE_DIR env var "
             "or platform-specific cache directory."
    )


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

        # Determine tile height
        tile_height = args.tile_height if args.tile_height else args.tile_width

        # Get window/center settings
        window_settings = _get_window_settings_from_args(args)

        if args.verbose:
            logger.info(f"Generating DICOM series mosaic")
            logger.info(f"Series UID: {args.seriesuid}")
            logger.info(f"Root: {args.root}")
            logger.info(f"Tile grid: {args.tile_width}x{tile_height}")
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

        # Generate evenly-spaced positions for the mosaic
        num_images = args.tile_width * tile_height
        interp = PositionInterpolator(series_index.instance_count)
        positions = interp.interpolate(num_images, start=args.start, end=args.end)

        # Retrieve and render images
        if args.verbose:
            logger.info(f"Retrieving and rendering {len(positions)} images...")
        try:
            images = series_index.get_images(
                positions=positions,
                contrast=window_settings,
                image_width=args.image_width,
            )
        except ValueError as e:
            logger.error(f"Failed to retrieve images: {e}")
            return 1

        if not images:
            logger.error(f"No images retrieved for series")
            return 1

        if args.verbose:
            logger.info(f"Retrieved {len(images)} images")

        # Tile the images into mosaic
        if args.verbose:
            logger.info("Tiling images into mosaic...")
        generator = MosaicGenerator(
            tile_width=args.tile_width,
            tile_height=tile_height,
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
            # Use natural size (large default) if no width specified
            image_width = args.image_width if args.image_width is not None else 2048
            output_image = instance.get_image(
                contrast=window_settings,
                image_width=image_width,
            )
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
            logger.error("At least one --contrast setting is required")
            return 1

        parsed_contrasts = []
        for contrast_str in args.contrast:
            try:
                parsed = parse_contrast_arg(contrast_str)
                parsed_contrasts.append(parsed)
            except ValueError as e:
                logger.error(f"Invalid contrast argument: {e}")
                return 1

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
        tile_height = args.tile_height

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
                logger.info(f"Tile height (instances): {tile_height}")
                logger.info(f"Contrast variations: {len(parsed_contrasts)}")

            # Generate evenly-spaced positions across range
            interp = PositionInterpolator(series_index.instance_count)
            positions = interp.interpolate(tile_height, start=args.start, end=args.end)

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

            if args.verbose:
                logger.info(f"Retrieved {len(instances)} instances")

        # Grid layout: contrasts on x-axis, instances on y-axis
        num_contrasts = len(parsed_contrasts)
        generator = MosaicGenerator(
            tile_width=num_contrasts,
            tile_height=tile_height,
            image_width=args.image_width
        )

        if args.verbose:
            logger.info(f"Grid layout: {num_contrasts}x{tile_height} (contrasts x instances)")
            logger.info("Generating contrast grid...")

        # Create images: for each instance, apply all contrasts
        # Store in row-major order: all contrasts for instance 0, then all for instance 1, etc.
        all_images = []
        for inst_idx, instance in enumerate(instances):
            for contrast_idx, contrast_settings in enumerate(parsed_contrasts):
                contrast_str = args.contrast[contrast_idx]
                if args.verbose:
                    logger.info(
                        f"Instance {inst_idx+1}/{len(instances)}, "
                        f"contrast {contrast_idx+1}/{num_contrasts}: {contrast_str}"
                    )

                try:
                    img = instance.get_image(
                        contrast=contrast_settings,
                        image_width=args.image_width,
                    )
                    if not img:
                        logger.error(f"Failed to generate image for instance {inst_idx+1}, contrast {contrast_str}")
                        return 1
                    all_images.append(img)
                except ValueError as e:
                    logger.error(f"Failed to generate image for instance {inst_idx+1}, contrast {contrast_str}: {e}")
                    return 1

        # Tile the images into grid
        output_image = generator.tile_images(all_images)

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
        # Parse and normalize series specification
        result = parse_and_normalize_series(args.series, args.root, logger)
        if result is None:
            return 1
        root_path, series_uid = result

        # Determine output directory
        if hasattr(args, 'cache_dir') and args.cache_dir:
            output_dir = Path(args.cache_dir)
        else:
            output_dir = get_cache_directory()

        # Determine index path
        index_path = output_dir / "indices" / f"{series_uid}_index.parquet"

        if args.verbose:
            logger.info(f"Looking for index for series {series_uid}")

        # Check if index already exists
        if index_path.exists():
            logger.info(f"Index found: {index_path}")
            return 0

        # Index doesn't exist, build it
        if args.verbose:
            logger.info(f"Building index for series {series_uid}")

        # Fetch headers using retriever
        try:
            retriever = DICOMRetriever(root_path)
            instance_uids = retriever.list_instances(series_uid)
            if not instance_uids:
                logger.error(f"No DICOM instances found for series {series_uid}")
                return 1

            if args.verbose:
                logger.info(f"Found {len(instance_uids)} instances")

            # Fetch headers in parallel using progressive range requests
            urls = [f"{series_uid}/{uid}" for uid in instance_uids]
            datasets_list = retriever.get_instances(urls, headers_only=True)

            # Build dict mapping uid to dataset
            datasets_by_uid = {}
            for uid, dataset in zip(instance_uids, datasets_list):
                if dataset is not None:
                    datasets_by_uid[uid] = dataset

            if not datasets_by_uid:
                logger.error(f"Failed to fetch any headers from {len(instance_uids)} instances")
                return 1

            # Construct storage root with series UID
            storage_root = f"{root_path}/{series_uid}/"

            # Generate parquet table
            if args.verbose:
                logger.info("Generating index parquet table...")
            df = _generate_parquet_table(datasets_by_uid, series_uid, storage_root)
            index_path.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(str(index_path))

        except Exception as e:
            logger.error(f"Failed to generate index: {e}")
            return 1

        logger.info(f"Index saved: {index_path}")
        return 0

    except Exception as e:
        logger.exception(f"Error: {e}")
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
        "--tile-width",
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
        "--contrast",
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
        "--tile-height",
        type=int,
        default=2,
        help="Number of instances per column (y-axis). Only used with --start/--end. Default: 2"
    )

    # Contrast settings (repeatable, always horizontal)
    contrast_parser.add_argument(
        "--contrast",
        action="append",
        help="Contrast settings (repeatable, x-axis): CT preset (ct-lung, ct-bone, ct-brain, ct-abdomen, ct-liver, ct-mediastinum, ct-soft-tissue), "
             "MR preset (mr-t1, mr-t2, mr-proton), legacy alias (lung, bone, brain, etc.), "
             "shortcut (soft, media, t1, t2, proton), 'auto', 'embedded', or custom window/level. "
             "Formats: '1500/500' (slash, medical standard) or '1500,500' (comma). "
             "Negative values supported (e.g., '1500/-500'). At least one --contrast is required."
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
        # Determine output directory (either cache-dir or output directory)
        if hasattr(args, 'output') and args.output:
            # Single series mode: -o specifies the output directory
            output_dir = Path(args.output)
            if len(args.series) != 1:
                logger.error("When using -o, specify exactly one series")
                return 1
            series_list = args.series
        elif hasattr(args, 'cache_dir') and args.cache_dir:
            # Multiple series mode: use cache directory
            output_dir = Path(args.cache_dir)
            series_list = args.series
        else:
            # Default: use default cache directory
            output_dir = get_cache_directory()
            series_list = args.series

        if args.verbose:
            logger.info(f"Building indices for {len(series_list)} series")
            logger.info(f"Output directory: {output_dir}")

        success_count = 0
        for series_spec in series_list:
            try:
                # Parse and normalize series specification
                result = parse_and_normalize_series(series_spec, args.root, logger)
                if result is None:
                    logger.error(f"Failed to parse series: {series_spec}")
                    continue

                root_path, series_uid = result

                if args.verbose:
                    logger.info(f"Building index for series {series_uid}...")

                # Determine index path
                index_path = output_dir / "indices" / f"{series_uid}_index.parquet"

                # Fetch headers using retriever
                retriever = DICOMRetriever(root_path)
                instance_uids = retriever.list_instances(series_uid)
                if not instance_uids:
                    logger.warning(f"No instances found for series {series_uid}")
                    continue

                # Apply limit if specified
                if hasattr(args, 'limit') and args.limit:
                    instance_uids = instance_uids[:args.limit]

                # Fetch headers in parallel using progressive range requests
                urls = [f"{series_uid}/{uid}" for uid in instance_uids]
                datasets_list = retriever.get_instances(urls, headers_only=True)

                # Build dict mapping uid to dataset
                datasets_by_uid = {}
                for uid, dataset in zip(instance_uids, datasets_list):
                    if dataset is not None:
                        datasets_by_uid[uid] = dataset

                if not datasets_by_uid:
                    logger.warning(f"Failed to fetch any headers from {len(instance_uids)} instances")
                    continue

                # Construct storage root with series UID
                storage_root = f"{root_path}/{series_uid}/"

                # Generate parquet table and write to file
                if args.verbose:
                    logger.info("Generating index parquet table...")
                df = _generate_parquet_table(datasets_by_uid, series_uid, storage_root)
                index_path.parent.mkdir(parents=True, exist_ok=True)
                df.write_parquet(str(index_path))

                if args.verbose:
                    logger.info(
                        f"Index saved: {index_path} "
                        f"({len(df)} rows, {len(df.columns)} columns)"
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
    1. Multiple series with --cache-dir: dicom-series-preview build-index SERIES1 SERIES2 ... --cache-dir /path
    2. Single series with -o: dicom-series-preview build-index SERIES -o /output/dir

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
        nargs="+",
        metavar="SERIES",
        help="DICOM Series UID(s) or path(s). Can be: series UID (e.g., 38902e14-b11f-4548-910e-771ee757dc82), "
             "partial UID prefix (e.g., 38902e14*, 389*), or full path (e.g., s3://idc-open-data/38902e14-b11f-4548-910e-771ee757dc82). "
             "Full paths override --root parameter."
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

    # Optional arguments
    index_parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of instances to process (useful for large series)"
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

    # Utility arguments
    get_index_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable detailed logging"
    )

    get_index_parser.set_defaults(func=get_index_command)


def _setup_parser():
    """
    Setup and configure the main argument parser with all subcommands.

    Returns:
        Configured ArgumentParser with mosaic, image, and contrast-mosaic subcommands
    """
    parser = argparse.ArgumentParser(
        description="Preview DICOM series stored on S3, HTTP, or local files with intelligent sampling and visualization.",
        prog="dicom-series-preview"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    subparsers.required = True

    # Setup subcommands
    _setup_mosaic_subcommand(subparsers)
    _setup_image_subcommand(subparsers)
    _setup_contrast_mosaic_subcommand(subparsers)
    _setup_build_index_subcommand(subparsers)
    _setup_get_index_subcommand(subparsers)

    return parser


def main():
    parser = _setup_parser()
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    return args.func(args, logger)


if __name__ == "__main__":
    sys.exit(main())
