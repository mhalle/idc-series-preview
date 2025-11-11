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
from typing import Optional

import polars as pl

from .retriever import DICOMRetriever
from .mosaic import MosaicGenerator
from .contrast import ContrastPresets
from .index_cache import _generate_parquet_table, load_or_generate_index, get_cache_directory


def parse_series_specification(
    series_spec: str, default_root: str
) -> tuple[str, str]:
    """
    Parse series specification, which can be either a series UID or a full path.

    Handles multiple formats:
    - Series UID only: "38902e14-b11f-4548-910e-771ee757dc82"
    - Series UID with prefix: "38902e14*"
    - Full path: "s3://idc-open-data/38902e14-b11f-4548-910e-771ee757dc82"
    - Full path with slash: "s3://idc-open-data/38902e14-b11f-4548-910e-771ee757dc82/"
    - Full path with wildcard: "s3://idc-open-data/38902e14-b11f-4548-910e-771ee757dc82/*"
    - Local path: "file:///path/to/series/38902e14-b11f-4548-910e-771ee757dc82"
    - HTTP URL: "http://example.com/dicom/38902e14-b11f-4548-910e-771ee757dc82"

    Args:
        series_spec: Series specification (UID or full path)
        default_root: Default root path to use if only UID is provided

    Returns:
        Tuple of (root_path, series_uid_or_prefix)

    Raises:
        ValueError: If the specification format is invalid
    """
    # Check if this is a full path (starts with a storage scheme)
    if any(series_spec.startswith(scheme) for scheme in ("s3://", "http://", "https://", "file://")):
        # This is a full path - extract root and series UID
        # Remove trailing wildcards and slashes
        clean_spec = series_spec.rstrip("/*")

        # Find the last slash to separate root from series UID
        last_slash = clean_spec.rfind("/")
        if last_slash == -1:
            raise ValueError(f"Invalid full path format: {series_spec}")

        root = clean_spec[:last_slash]
        series_uid = clean_spec[last_slash + 1 :]

        if not series_uid:
            raise ValueError(f"No series UID found in path: {series_spec}")

        return root, series_uid
    else:
        # This is a series UID or prefix - use default root
        return default_root, series_spec


def normalize_series_uid(series_uid: str) -> str:
    """
    Normalize a series UID by adding hyphens if not present.

    Converts formats like:
    - 38902e14b11f4548910e771ee757dc82
    - 38902e14-b11f-4548-910e-771ee757dc82
    - 38902e14* (prefix with wildcard)
    - 389* (partial prefix)

    To standard UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
    Or returns prefix with wildcard for partial matches.

    Args:
        series_uid: Series UID with or without hyphens, or prefix with wildcard

    Returns:
        Normalized series UID with hyphens, or prefix pattern for searching
    """
    # Check if this is a prefix search (contains wildcard)
    if '*' in series_uid:
        # This is a prefix search - clean it and return for searching
        prefix = series_uid.replace('*', '').replace('-', '').lower()
        if not prefix:
            raise ValueError("Prefix cannot be empty")
        return f"{prefix}*"  # Return as prefix pattern

    # Remove any existing hyphens for full UUID
    cleaned = series_uid.replace('-', '').lower()

    # UUID format: 8-4-4-4-12 characters
    if len(cleaned) != 32:
        raise ValueError(f"Series UID must be 32 hex characters (got {len(cleaned)}): {series_uid}")

    # Re-insert hyphens at correct positions
    formatted = f"{cleaned[0:8]}-{cleaned[8:12]}-{cleaned[12:16]}-{cleaned[16:20]}-{cleaned[20:32]}"

    return formatted


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
        "--image-width",
        type=int,
        default=128,
        help="Width of each image tile in pixels. Height will be proportionally scaled. Default: 128"
    )

    # Contrast parameters (shared)
    parser.add_argument(
        "--contrast",
        help="Contrast settings: CT preset (ct-lung, ct-bone, ct-brain, ct-abdomen, ct-liver, ct-mediastinum, ct-soft-tissue), "
             "legacy alias (lung, bone, brain, etc.), shortcut (soft for ct-soft-tissue, media for ct-mediastinum), "
             "'auto' for auto-detection, 'embedded' for DICOM file window/level, "
             "or custom window/level values (e.g., '1500/500' or '1500,-500'). "
             "Supports both slash (medical standard) and comma separators."
    )

    # Output format arguments
    parser.add_argument(
        "-q", "--quality",
        type=int,
        default=25,
        help="Output image quality 0-100. Default: 25 for WebP, 70+ recommended for JPEG"
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
        "--start",
        type=float,
        default=0.0,
        help="Start of normalized z-position range (0.0-1.0). Default: 0.0 (beginning of series)"
    )
    parser.add_argument(
        "--end",
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


def _load_or_generate_index(
    index_dir: Optional[str], series_uid: str, root_path: str, verbose: bool, logger
) -> Optional[pl.DataFrame]:
    """
    Load or generate a DICOM series index.

    Loads from cache directory or auto-generates if needed.
    Index files are named {series_uid}_index.parquet in the cache directory.

    Args:
        index_dir: Optional cache directory from --index-directory
        series_uid: Normalized series UID
        root_path: Root storage path
        verbose: Verbose logging enabled
        logger: Logger instance

    Returns:
        Polars DataFrame with index data, or None on error
    """
    return load_or_generate_index(
        series_uid=series_uid,
        root_path=root_path,
        index_dir=index_dir,
        logger_instance=logger,
    )


def _initialize_retriever_with_cache(
    root_path: str, series_uid: str, args, logger
) -> DICOMRetriever:
    """
    Initialize DICOMRetriever with optional cache support.

    Checks if caching is enabled (default unless --no-cache is set),
    loads or generates index if enabled, and passes to retriever.

    Args:
        root_path: Root storage path
        series_uid: Normalized series UID
        args: Parsed command arguments (must have cache_dir and no_cache attributes)
        logger: Logger instance

    Returns:
        DICOMRetriever instance (with optional index_df)
    """
    index_df = None
    use_cache = not getattr(args, 'no_cache', False)
    if use_cache:
        cache_dir = getattr(args, 'cache_dir', None)
        index_df = _load_or_generate_index(
            cache_dir, series_uid, root_path, args.verbose, logger
        )

    return DICOMRetriever(root_path, index_df=index_df)


def _parse_and_normalize_series(series_spec, root, logger):
    """
    Parse, normalize, and resolve series specification.

    Handles full paths, prefixes, and series UIDs, performing prefix search if needed.

    Args:
        series_spec: Series specification (UID, prefix, or full path)
        root: Default root path
        logger: Logger instance

    Returns:
        Tuple of (root_path, series_uid) on success
        None on error (error already logged)
    """
    # Parse series specification (can be UID or full path)
    try:
        root_path, parsed_spec = parse_series_specification(series_spec, root)

        # If a full path was provided and --root was also specified, note the override
        if root_path != root:
            logger.debug(f"Full path specified, overriding --root with: {root_path}")
    except ValueError as e:
        logger.error(f"Invalid series specification: {e}")
        return None

    # Normalize series UID (add hyphens if not present, or prepare for prefix search)
    try:
        series_uid = normalize_series_uid(parsed_spec)
    except ValueError as e:
        logger.error(f"Invalid series UID: {e}")
        return None

    # Handle prefix search (ends with *)
    if series_uid.endswith('*'):
        logger.info(f"Searching for series matching prefix: {parsed_spec}...")

        retriever_temp = DICOMRetriever(root_path)
        prefix = series_uid.rstrip('*')
        matches = retriever_temp.find_series_by_prefix(prefix)

        if not matches:
            logger.error(f"No series found matching prefix: {parsed_spec}")
            return None
        elif len(matches) > 1:
            logger.error(f"Prefix '{parsed_spec}' matches {len(matches)} series:")
            for match in matches[:10]:  # Show first 10
                logger.error(f"  - {match}")
            if len(matches) > 10:
                logger.error(f"  ... and {len(matches) - 10} more")
            logger.error("Please provide a more specific prefix")
            return None
        else:
            series_uid = matches[0]
            logger.info(f"Found matching series: {series_uid}")

    return root_path, series_uid


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
        # Parse and normalize series specification
        result = _parse_and_normalize_series(args.seriesuid, args.root, logger)
        if result is None:
            return 1
        root_path, series_uid = result

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
            logger.info(f"Series UID: {series_uid}")
            logger.info(f"Root: {root_path}")
            logger.info(f"Tile grid: {args.tile_width}x{tile_height}")
            logger.info(f"Tile width: {args.image_width}px")
            if args.start > 0.0 or args.end < 1.0:
                logger.info(f"Range: {args.start:.1%} to {args.end:.1%} of series")

        # Initialize retriever with optional cache support
        retriever = _initialize_retriever_with_cache(root_path, series_uid, args, logger)

        # Retrieve DICOM instances
        if args.verbose:
            logger.info("Retrieving DICOM instances...")
        instances = retriever.get_instances_distributed(
            series_uid,
            args.tile_width * tile_height,
            start=args.start,
            end=args.end
        )

        if not instances:
            logger.error(f"No DICOM instances found for series {series_uid}")
            return 1

        if args.verbose:
            logger.info(f"Retrieved {len(instances)} instances")

        # Generate mosaic
        if args.verbose:
            logger.info("Generating mosaic...")
        generator = MosaicGenerator(
            tile_width=args.tile_width,
            tile_height=tile_height,
            image_width=args.image_width,
            window_settings=window_settings
        )

        output_image = generator.create_mosaic(instances, retriever, series_uid)

        if not output_image:
            logger.error("Failed to generate mosaic")
            return 1

        # Save output
        if args.verbose:
            logger.info(f"Saving mosaic to {args.output}...")
        generator.save_image(
            output_image,
            args.output,
            quality=args.quality
        )

        if args.verbose:
            logger.info("Done!")
        return 0

    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1


def image_command(args, logger):
    """Extract a single image from a DICOM series at a specific position."""
    try:
        # Parse and normalize series specification
        result = _parse_and_normalize_series(args.seriesuid, args.root, logger)
        if result is None:
            return 1
        root_path, series_uid = result

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
            logger.info(f"Series UID: {series_uid}")
            logger.info(f"Root: {root_path}")
            logger.info(f"Position: {args.position:.1%}")

        # Initialize retriever with optional cache support
        retriever = _initialize_retriever_with_cache(root_path, series_uid, args, logger)

        # Retrieve single instance at position
        if args.verbose:
            logger.info("Retrieving DICOM instance...")
        instance = retriever.get_instance_at_position(
            series_uid, args.position, slice_offset=args.slice_offset
        )

        if not instance:
            if args.slice_offset != 0:
                logger.error(f"Slice offset {args.slice_offset} is out of bounds for this series (check error messages above for details)")
            else:
                logger.error(f"No DICOM instance found at position {args.position}")
            return 1

        if args.verbose:
            instance_uid, _ = instance
            logger.info(f"Retrieved instance {instance_uid}")

        # Generate single image (no mosaic)
        if args.verbose:
            logger.info("Generating image...")
        generator = MosaicGenerator(
            image_width=args.image_width,
            window_settings=window_settings
        )

        output_image = generator.create_single_image(instance, retriever, series_uid)

        if not output_image:
            logger.error("Failed to generate image")
            return 1

        # Save output
        if args.verbose:
            logger.info(f"Saving image to {args.output}...")
        generator.save_image(
            output_image,
            args.output,
            quality=args.quality
        )

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
        # Parse and normalize series specification
        result = _parse_and_normalize_series(args.seriesuid, args.root, logger)
        if result is None:
            return 1
        root_path, series_uid = result

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

        # Determine instance selection mode
        instances_list = []
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
                logger.info(f"Series UID: {series_uid}")
                logger.info(f"Root: {root_path}")
                logger.info(f"Position: {args.position:.1%}")
                logger.info(f"Contrast variations: {len(parsed_contrasts)}")

            # Initialize retriever with optional cache support
            retriever = _initialize_retriever_with_cache(root_path, series_uid, args, logger)

            # Retrieve single instance at position
            if args.verbose:
                logger.info("Retrieving DICOM instance...")
            instance = retriever.get_instance_at_position(
                series_uid, args.position, slice_offset=args.slice_offset
            )

            if not instance:
                if args.slice_offset != 0:
                    logger.error(f"Slice offset {args.slice_offset} is out of bounds for this series")
                else:
                    logger.error(f"No DICOM instance found at position {args.position}")
                return 1

            if args.verbose:
                instance_uid, _ = instance
                logger.info(f"Retrieved instance {instance_uid}")

            instances_list = [instance]

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
                logger.info(f"Series UID: {series_uid}")
                logger.info(f"Root: {root_path}")
                logger.info(f"Range: {args.start:.1%} to {args.end:.1%}")
                logger.info(f"Tile height (instances): {tile_height}")
                logger.info(f"Contrast variations: {len(parsed_contrasts)}")

            # Initialize retriever with optional cache support
            retriever = _initialize_retriever_with_cache(root_path, series_uid, args, logger)

            # Retrieve distributed instances across range
            if args.verbose:
                logger.info("Retrieving DICOM instances...")
            instances_list = retriever.get_instances_distributed(
                series_uid,
                tile_height,
                start=args.start,
                end=args.end
            )

            if not instances_list:
                logger.error(f"No DICOM instances found in range {args.start:.1%}-{args.end:.1%}")
                return 1

            if args.verbose:
                logger.info(f"Retrieved {len(instances_list)} instances")

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
        for inst_idx, instance in enumerate(instances_list):
            for contrast_idx, contrast_settings in enumerate(parsed_contrasts):
                contrast_str = args.contrast[contrast_idx]
                if args.verbose:
                    logger.info(
                        f"Instance {inst_idx+1}/{len(instances_list)}, "
                        f"contrast {contrast_idx+1}/{num_contrasts}: {contrast_str}"
                    )

                # Create generator with this contrast setting
                gen = MosaicGenerator(
                    image_width=args.image_width,
                    window_settings=contrast_settings
                )

                img = gen.create_single_image(instance, retriever, series_uid)
                if not img:
                    logger.error(f"Failed to generate image for instance {inst_idx+1}, contrast {contrast_str}")
                    return 1
                all_images.append(img)

        # Tile the images into grid
        output_image = generator.tile_images(all_images)

        if not output_image:
            logger.error("Failed to tile images into grid")
            return 1

        # Save output
        if args.verbose:
            logger.info(f"Saving contrast grid to {args.output}...")
        generator.save_image(
            output_image,
            args.output,
            quality=args.quality
        )

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
        result = _parse_and_normalize_series(args.series, args.root, logger)
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
            datasets_list = retriever.get_instances(urls, headers_only=True, max_workers=8)

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
        default=6,
        help="Number of images per row in mosaic. Default: 6"
    )
    mosaic_parser.add_argument(
        "--tile-height",
        type=int,
        help="Number of images per column in mosaic. Default: same as --tile-width"
    )
    mosaic_parser.set_defaults(func=mosaic_command)


def _setup_get_image_subcommand(subparsers):
    """
    Setup get-image subcommand with all its arguments.

    Args:
        subparsers: The subparsers object from ArgumentParser
    """
    image_parser = subparsers.add_parser(
        "get-image",
        help="Extract a single image from a DICOM series at a specific position"
    )
    add_common_arguments(image_parser)
    add_cache_arguments(image_parser)
    image_parser.add_argument(
        "--position",
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
        "--image-width",
        type=int,
        default=128,
        help="Width of each image in pixels. Height will be proportionally scaled. Default: 128"
    )

    # Instance selection: position mode (single instance)
    contrast_parser.add_argument(
        "--position",
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
        "--start",
        type=float,
        help="Start of normalized z-position range (0.0-1.0) for selecting multiple instances. "
             "Cannot be used with --position."
    )
    contrast_parser.add_argument(
        "--end",
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
             "legacy alias (lung, bone, brain, etc.), shortcut (soft, media), 'auto', 'embedded', or custom window/level. "
             "Formats: '1500/500' (slash, medical standard) or '1500,500' (comma). "
             "Negative values supported (e.g., '1500/-500'). At least one --contrast is required."
    )

    # Output format arguments
    contrast_parser.add_argument(
        "-q", "--quality",
        type=int,
        default=25,
        help="Output image quality 0-100. Default: 25 for WebP, 70+ recommended for JPEG"
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
                result = _parse_and_normalize_series(series_spec, args.root, logger)
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
                datasets_list = retriever.get_instances(urls, headers_only=True, max_workers=8)

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
        Configured ArgumentParser with mosaic, get-image, and contrast-mosaic subcommands
    """
    parser = argparse.ArgumentParser(
        description="Preview DICOM series stored on S3, HTTP, or local files with intelligent sampling and visualization.",
        prog="dicom-series-preview"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    subparsers.required = True

    # Setup subcommands
    _setup_mosaic_subcommand(subparsers)
    _setup_get_image_subcommand(subparsers)
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
