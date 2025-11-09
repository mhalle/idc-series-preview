#!/usr/bin/env python3
"""
DICOM Mosaic Generator

Retrieves DICOM instances from S3/HTTP/local storage, creates a mosaic, and exports to image format.
"""

import argparse
import sys
import logging
from pathlib import Path

from .retriever import DICOMRetriever
from .mosaic import MosaicGenerator
from .contrast import ContrastPresets


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


def main():
    parser = argparse.ArgumentParser(
        description="Generate a DICOM mosaic from a series stored on S3, HTTP, or local files"
    )

    # Required arguments
    parser.add_argument(
        "seriesuid",
        help="DICOM Series UID (full UID, partial with hyphens, or prefix with wildcard). "
             "Examples: 38902e14-b11f-4548-910e-771ee757dc82, "
             "38902e14b11f4548910e771ee757dc82, 38902e14*, 389*"
    )
    parser.add_argument(
        "output",
        help="Output image path (webp or jpg)"
    )

    # Storage arguments
    parser.add_argument(
        "--root",
        default="s3://idc-open-data",
        help="Root path for DICOM files (S3, HTTP, or local path). Default: s3://idc-open-data"
    )

    # Tiling arguments
    parser.add_argument(
        "--tile-width",
        type=int,
        default=6,
        help="Number of images per row in mosaic. Default: 6"
    )
    parser.add_argument(
        "--tile-height",
        type=int,
        help="Number of images per column in mosaic. Default: same as --tile-width"
    )

    # Image scaling
    parser.add_argument(
        "--image-width",
        type=int,
        default=128,
        help="Width of each image tile in pixels. Height will be proportionally scaled. Default: 128"
    )

    # Contrast parameters
    parser.add_argument(
        "--window-width",
        type=float,
        help="Window width for contrast adjustment"
    )
    parser.add_argument(
        "--window-center",
        type=float,
        help="Window center for contrast adjustment"
    )
    parser.add_argument(
        "--contrast-preset",
        choices=list(ContrastPresets.PRESETS.keys()) + ["auto"],
        help="Preset contrast settings (auto will auto-detect from image)"
    )

    # Range selection arguments
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
    parser.add_argument(
        "--position",
        type=float,
        help="Extract single image at normalized z-position (0.0-1.0). When specified, no tiling is performed."
    )
    parser.add_argument(
        "--slice-offset",
        type=int,
        default=0,
        help="Offset from --position by number of slices (e.g., 1 for next slice, -1 for previous). Only valid with --position."
    )

    # Output format arguments
    parser.add_argument(
        "-q", "--quality",
        type=int,
        default=25,
        help="Output image quality (0-100). Default: 25"
    )

    # Utility arguments
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Normalize series UID (add hyphens if not present, or prepare for prefix search)
        try:
            series_uid = normalize_series_uid(args.seriesuid)
        except ValueError as e:
            logger.error(f"Invalid series UID: {e}")
            return 1

        # Handle prefix search (ends with *)
        if series_uid.endswith('*'):
            if args.verbose:
                logger.info(f"Searching for series matching prefix: {args.seriesuid}...")

            retriever_temp = DICOMRetriever(args.root)
            prefix = series_uid.rstrip('*')
            matches = retriever_temp.find_series_by_prefix(prefix)

            if not matches:
                logger.error(f"No series found matching prefix: {args.seriesuid}")
                return 1
            elif len(matches) > 1:
                logger.error(f"Prefix '{args.seriesuid}' matches {len(matches)} series:")
                for match in matches[:10]:  # Show first 10
                    logger.error(f"  - {match}")
                if len(matches) > 10:
                    logger.error(f"  ... and {len(matches) - 10} more")
                logger.error("Please provide a more specific prefix")
                return 1
            else:
                series_uid = matches[0]
                if args.verbose:
                    logger.info(f"Found matching series: {series_uid}")

        # Validate output format
        output_path = Path(args.output)
        if output_path.suffix.lower() not in ['.webp', '.jpg', '.jpeg']:
            logger.error("Output file must be .webp or .jpg/.jpeg")
            return 1

        # Validate position parameter
        if args.position is not None:
            if not (0.0 <= args.position <= 1.0):
                logger.error("--position must be between 0.0 and 1.0")
                return 1
            if args.start != 0.0 or args.end != 1.0:
                logger.error("--position cannot be used with --start or --end")
                return 1
        else:
            # --slice-offset only valid with --position
            if args.slice_offset != 0:
                logger.error("--slice-offset can only be used with --position")
                return 1

        # Validate range parameters (only if not using position)
        if args.position is None:
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
        window_settings = None
        if args.contrast_preset:
            if args.contrast_preset == "auto":
                window_settings = "auto"
            else:
                window_settings = ContrastPresets.get_preset(args.contrast_preset)
        elif args.window_width and args.window_center:
            window_settings = {
                'window_width': args.window_width,
                'window_center': args.window_center
            }

        if args.verbose:
            logger.info(f"Starting DICOM mosaic generation")
            logger.info(f"Series UID: {series_uid}")
            logger.info(f"Root: {args.root}")

        # Initialize retriever
        retriever = DICOMRetriever(args.root)

        # Handle position mode (single image) vs range/mosaic mode
        if args.position is not None:
            if args.verbose:
                logger.info(f"Extracting single image at position {args.position:.1%}")

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
        else:
            # Range/mosaic mode
            if args.verbose:
                logger.info(f"Tile grid: {args.tile_width}x{tile_height}")
                logger.info(f"Tile width: {args.image_width}px")
                if args.start > 0.0 or args.end < 1.0:
                    logger.info(f"Range: {args.start:.1%} to {args.end:.1%} of series")

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
            mode = "image" if args.position is not None else "mosaic"
            logger.info(f"Saving {mode} to {args.output}...")
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


if __name__ == "__main__":
    sys.exit(main())
