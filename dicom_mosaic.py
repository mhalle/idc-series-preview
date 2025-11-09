#!/usr/bin/env python3
"""
DICOM Mosaic Generator

Retrieves DICOM instances from S3/HTTP/local storage, creates a mosaic, and exports to image format.
"""

import argparse
import sys
import logging
from pathlib import Path

from src.retriever import DICOMRetriever
from src.mosaic import MosaicGenerator
from src.contrast import ContrastPresets


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
        help="DICOM Series UID"
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

    # Output format arguments
    parser.add_argument(
        "-q", "--quality",
        type=int,
        default=85,
        help="Output image quality (0-100). Default: 85"
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
        # Validate output format
        output_path = Path(args.output)
        if output_path.suffix.lower() not in ['.webp', '.jpg', '.jpeg']:
            logger.error("Output file must be .webp or .jpg/.jpeg")
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
            logger.info(f"Series UID: {args.seriesuid}")
            logger.info(f"Root: {args.root}")
            logger.info(f"Tile grid: {args.tile_width}x{tile_height}")
            logger.info(f"Tile width: {args.image_width}px")

        # Retrieve DICOM instances
        if args.verbose:
            logger.info("Retrieving DICOM instances...")
        retriever = DICOMRetriever(args.root)
        instances = retriever.get_instances_distributed(
            args.seriesuid,
            args.tile_width * tile_height
        )

        if not instances:
            logger.error(f"No DICOM instances found for series {args.seriesuid}")
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

        mosaic_image = generator.create_mosaic(instances, retriever, args.seriesuid)

        if not mosaic_image:
            logger.error("Failed to generate mosaic")
            return 1

        # Save output
        if args.verbose:
            logger.info(f"Saving mosaic to {args.output}...")
        generator.save_image(
            mosaic_image,
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
