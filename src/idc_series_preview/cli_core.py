#!/usr/bin/env python3
"""
DICOM Series Preview

Preview DICOM series stored on S3, HTTP, or local files with intelligent sampling
and visualization. Supports both tiled mosaics and individual image extraction.
"""

import logging
import math
import shutil
import json
from pathlib import Path
from datetime import date, datetime
from decimal import Decimal
from fnmatch import fnmatchcase

from PIL import Image

from .image_utils import MosaicRenderer, save_image
from .contrast import ContrastPresets
from .index_cache import get_cache_directory
from .constants import DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_QUALITY
from .workers import optimal_workers


SUPPORTED_INDEX_FORMATS = {"parquet", "json", "jsonl"}
VIDEO_FETCH_CHUNK_SIZE = 32
VIDEO_CRF_BEST = 10
VIDEO_CRF_WORST = 40
HEADER_DEFAULT_INDENT = 2


def _chunked(sequence, chunk_size):
    """Yield slices of `sequence` with at most chunk_size entries."""
    for idx in range(0, len(sequence), chunk_size):
        yield sequence[idx : idx + chunk_size]


def _map_quality_to_crf(quality: int) -> int:
    """Map user-friendly 0-100 quality into libx264 CRF (10 best, 40 worst)."""
    clamped = max(0, min(100, quality))
    span = VIDEO_CRF_WORST - VIDEO_CRF_BEST
    # Higher quality -> lower CRF
    crf = VIDEO_CRF_WORST - (clamped / 100.0) * span
    return int(round(crf))


def _normalize_json_value(value):
    """Convert Polars/pydicom/native values into JSON-serializable equivalents."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {k: _normalize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_json_value(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return _normalize_json_value(value.tolist())
        except Exception:
            return str(value)
    return str(value)


def _convert_header_value(value):
    """Convert pydicom values into JSON-serializable structures."""
    from pydicom.dataset import Dataset
    from pydicom.sequence import Sequence
    from pydicom.multival import MultiValue

    if value is None:
        return None
    if isinstance(value, Dataset):
        return _dataset_to_header_dict(value)
    if isinstance(value, Sequence):
        return [_dataset_to_header_dict(item) for item in value]
    if isinstance(value, MultiValue) or isinstance(value, (list, tuple)):
        return [_convert_header_value(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            return str(value)
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _dataset_to_header_dict(dataset):
    """Serialize a pydicom Dataset into a JSON-friendly dictionary."""
    header = {}
    for element in dataset:
        keyword = element.keyword or element.name or str(element.tag)
        if keyword == "PixelData":
            continue
        header[keyword] = _convert_header_value(element.value)
    return header


def _start_ffmpeg_process(width: int, height: int, fps: float, output_path: str, *, crf: int):
    """Start ffmpeg process that accepts rgb24 frames over stdin."""
    try:
        import ffmpeg
    except ModuleNotFoundError as exc:
        raise RuntimeError("ffmpeg-python is required for video export") from exc

    try:
        stream = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                s=f"{width}x{height}",
                framerate=fps,
            )
            .output(
                output_path,
                vcodec="libx264",
                pix_fmt="yuv420p",
                r=fps,
                movflags="faststart",
                crf=crf,
            )
            .overwrite_output()
            .global_args("-hide_banner", "-loglevel", "error")
        )
        return stream.run_async(
            pipe_stdin=True,
            pipe_stdout=True,
            pipe_stderr=True,
        )
    except Exception as exc:  # pragma: no cover - delegated to ffmpeg
        raise RuntimeError(f"Failed to launch ffmpeg: {exc}") from exc


def _shutdown_ffmpeg_process(process, logger, *, report_errors=True):
    """Close stdin and wait for ffmpeg to exit."""
    if process is None:
        return 0

    stdin = getattr(process, "stdin", None)
    if stdin is not None:
        try:
            stdin.close()
        except BrokenPipeError:
            pass
        except ValueError:
            pass
        finally:
            try:
                process.stdin = None
            except Exception:  # pragma: no cover - defensive
                pass

    stdout = b""
    stderr = b""
    try:
        stdout, stderr = process.communicate()
    except Exception:
        process.kill()
        stdout, stderr = process.communicate()

    if report_errors and getattr(process, "returncode", 0) != 0:
        stderr_text = stderr.decode("utf-8", errors="ignore").strip()
        details = f": {stderr_text}" if stderr_text else ""
        logger.error(f"ffmpeg exited with code {process.returncode}{details}")
        return 1

    return 0


def _generate_positions(series_index, samples: int, start: float, end: float):
    """Return evenly spaced normalized positions within [start, end]."""
    from .api import PositionInterpolator

    if samples < 1:
        raise ValueError("--samples must be >= 1")

    interp = PositionInterpolator(series_index.instance_count)
    positions, _ = interp.interpolate_unique(
        samples,
        start=start,
        end=end,
    )
    if not positions:
        raise ValueError("Requested range did not produce any slices")
    return positions


def _resolve_grid_dimensions(slice_count: int, columns: int | None, rows: int | None):
    """Determine column/row counts for mosaics."""
    if slice_count < 1:
        raise ValueError("No slices available to layout")

    if columns is not None and columns < 1:
        raise ValueError("--columns must be >= 1")
    if rows is not None and rows < 1:
        raise ValueError("--rows must be >= 1")

    if columns is None and rows is None:
        columns = max(1, math.ceil(math.sqrt(slice_count)))
        rows = max(1, math.ceil(slice_count / columns))
    elif columns is None:
        rows = max(1, rows)
        columns = max(1, math.ceil(slice_count / rows))
    elif rows is None:
        columns = max(1, columns)
        rows = max(1, math.ceil(slice_count / columns))
    else:
        columns = max(1, columns)
        rows = max(1, rows)

    return columns, rows


def _resize_canvas_if_needed(
    image,
    target_width: int | None,
    target_height: int | None,
    *,
    shrink_only: bool = False,
):
    """Resize the given PIL image if target dimensions are provided."""
    if target_width is None and target_height is None:
        return image

    width, height = image.size
    resized = image

    if target_width is not None and target_height is not None:
        scale = min(target_width / width, target_height / height)
        scale = max(scale, 0)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        if new_width != width or new_height != height:
            resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        if shrink_only or (new_width == target_width and new_height == target_height):
            return resized
        canvas = Image.new(image.mode, (target_width, target_height), color=0)
        offset = ((target_width - new_width) // 2, (target_height - new_height) // 2)
        canvas.paste(resized, offset)
        return canvas

    if target_width is not None:
        new_height = max(1, int(round(height * (target_width / width))))
        new_size = (max(1, target_width), new_height)
    else:
        new_width = max(1, int(round(width * (target_height / height))))
        new_size = (new_width, max(1, target_height))

    if new_size == image.size:
        return image
    return image.resize(new_size, Image.Resampling.LANCZOS)


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
        from .api import SeriesIndex

        output_path = Path(args.output)
        if not _validate_output_format(output_path):
            logger.error("Output file must be .webp or .jpg/.jpeg")
            return 1

        if not (0.0 <= args.start <= 1.0):
            logger.error("--start must be between 0.0 and 1.0")
            return 1
        if not (0.0 <= args.end <= 1.0):
            logger.error("--end must be between 0.0 and 1.0")
            return 1
        if args.start > args.end:
            logger.error("--start must be less than or equal to --end")
            return 1
        if args.samples < 1:
            logger.error("--samples must be >= 1")
            return 1

        window_settings = _get_window_settings_from_args(args)

        if args.verbose:
            logger.info("Generating DICOM series mosaic")
            logger.info(f"Series UID: {args.seriesuid}")
            logger.info(f"Root: {args.root}")
            logger.info(f"Requested samples: {args.samples}")
            if args.start > 0.0 or args.end < 1.0:
                logger.info(f"Range: {args.start:.1%} to {args.end:.1%} of series")

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

        try:
            positions = _generate_positions(
                series_index,
                args.samples,
                args.start,
                args.end,
            )
        except ValueError as e:
            logger.error(str(e))
            return 1

        actual_slices = len(positions)
        try:
            columns, rows = _resolve_grid_dimensions(actual_slices, args.columns, args.rows)
        except ValueError as e:
            logger.error(str(e))
            return 1

        max_tiles = columns * rows
        if actual_slices > max_tiles:
            positions = positions[:max_tiles]
            actual_slices = len(positions)
            if args.verbose:
                logger.info(
                    "Limiting mosaic to %d slices to fit %d x %d grid",
                    actual_slices,
                    columns,
                    rows,
                )

        tile_pixel_width = args.width
        if tile_pixel_width:
            tile_pixel_width = max(1, tile_pixel_width // columns)
        else:
            tile_pixel_width = max(1, 768 // columns)

        if args.verbose:
            logger.info(f"Grid layout: {columns}x{rows} (columns x rows)")
            if args.width:
                logger.info(f"Target canvas width: {args.width}px")
            if args.height:
                logger.info(f"Target canvas height: {args.height}px")

        mosaic_workers = optimal_workers(actual_slices, max_workers=10, min_workers=5)
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

        renderer = InstanceRenderer(image_width=tile_pixel_width, window_settings=window_settings)
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
            tile_width=columns,
            tile_height=rows,
            image_width=tile_pixel_width,
        )

        output_image = generator.tile_images(images)
        if not output_image:
            logger.error("Failed to tile images")
            return 1

        output_image = _resize_canvas_if_needed(
            output_image,
            args.width,
            args.height,
            shrink_only=getattr(args, "shrink_to_fit", False),
        )

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

            renderer = InstanceRenderer(image_width=args.width, window_settings=window_settings)
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


def header_command(args, logger):
    """Export cached header information for a single instance."""
    try:
        from .api import SeriesIndex
    except ImportError as e:  # pragma: no cover
        logger.error(f"Missing dependency: {e}")
        return 1

    if not (0.0 <= args.position <= 1.0):
        logger.error("--position must be between 0.0 and 1.0")
        return 1

    if args.verbose:
        logger.info("Exporting instance header from index")
        logger.info(f"Series UID: {args.seriesuid}")
        logger.info(f"Root: {args.root}")
        logger.info(f"Position: {args.position:.1%}")
        if args.slice_offset:
            logger.info(f"Slice offset: {args.slice_offset}")

    try:
        use_cache = not getattr(args, "no_cache", False)
        cache_dir = getattr(args, "cache_dir", None)
        series_index = SeriesIndex(
            args.seriesuid,
            root=args.root,
            cache_dir=cache_dir,
            use_cache=use_cache,
        )
    except ValueError as e:
        logger.error(f"Failed to initialize series index: {e}")
        return 1

    index_df = series_index.index_dataframe
    if "IndexNormalized" not in index_df.columns:
        logger.error("Index does not contain normalized positions")
        return 1

    normalized = index_df["IndexNormalized"].to_list()
    if not normalized:
        logger.error("Series index is empty")
        return 1

    target = args.position
    closest_idx = min(range(len(normalized)), key=lambda i: abs(normalized[i] - target))
    slice_idx = closest_idx + args.slice_offset
    if not (0 <= slice_idx < len(normalized)):
        logger.error("Slice selection is outside the series bounds")
        return 1

    row_dict = index_df[slice_idx : slice_idx + 1].to_dicts()[0]
    normalized_row = {key: _normalize_json_value(value) for key, value in row_dict.items()}

    requested_tags = [tag for tag in getattr(args, "tags", []) if tag]
    if requested_tags:
        lookup = {key.lower(): key for key in normalized_row.keys()}
        ordered_keys = list(normalized_row.keys())
        filtered = {}
        missing = []
        for tag in requested_tags:
            lower_tag = tag.lower()
            is_glob = any(ch in tag for ch in "*?[")
            if not is_glob:
                key = lookup.get(lower_tag)
                if key is not None:
                    filtered[key] = normalized_row[key]
                else:
                    missing.append(tag)
                continue

            matched = False
            for candidate in ordered_keys:
                if fnmatchcase(candidate.lower(), lower_tag):
                    filtered[candidate] = normalized_row[candidate]
                    matched = True
            if not matched:
                # Glob patterns are implicitly quiet; no warning
                continue

        normalized_row = filtered
        exact_missing = [tag for tag in missing if not any(ch in tag for ch in "*?[")]
        if exact_missing and not getattr(args, "quiet", False):
            logger.warning("Tags not found in index: %s", ", ".join(exact_missing))

    indent = HEADER_DEFAULT_INDENT if args.indent is None else args.indent
    json_text = json.dumps(normalized_row, indent=indent if indent > 0 else None)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_text)
        if args.verbose:
            logger.info(f"Header written to {output_path}")
    else:
        print(json_text)

    return 0


def video_command(args, logger):
    """Render every slice (or selected slices) into an MP4 video."""
    try:
        from .api import SeriesIndex, PositionInterpolator
        from .image_utils import InstanceRenderer
    except ImportError as e:  # pragma: no cover - import guard
        logger.error(f"Missing dependency: {e}")
        return 1

    output_path = Path(args.output)
    if output_path.suffix.lower() != ".mp4":
        logger.error("Video output path must end with .mp4")
        return 1

    if args.fps <= 0:
        logger.error("--fps must be greater than 0")
        return 1

    if args.samples < 1:
        logger.error("--samples must be >= 1")
        return 1

    video_quality = getattr(args, "quality", DEFAULT_IMAGE_QUALITY)
    if video_quality is None:
        video_quality = DEFAULT_IMAGE_QUALITY
    if not (0 <= video_quality <= 100):
        logger.error("--quality must be between 0 and 100")
        return 1
    video_crf = _map_quality_to_crf(video_quality)

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

    window_settings = _get_window_settings_from_args(args)

    if args.verbose:
        logger.info("Encoding DICOM series to video")
        logger.info(f"Series UID: {args.seriesuid}")
        logger.info(f"Root: {args.root}")
        logger.info(f"Output: {args.output}")
        logger.info(f"FPS: {args.fps}")
        logger.info(f"Quality: {video_quality} -> CRF {video_crf}")
        logger.info(f"Target frames: {args.samples}")
        if args.start > 0.0 or args.end < 1.0:
            logger.info(f"Range: {args.start:.1%} to {args.end:.1%}")

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

    interpolator = PositionInterpolator(series_index.instance_count)
    positions, _ = interpolator.interpolate_unique(
        args.samples,
        start=args.start,
        end=args.end,
    )
    if not positions:
        logger.error("Unable to interpolate frames for requested range")
        return 1
    selection = positions
    selection_mode = "positions"

    ffmpeg_process = None
    frame_size = None
    frame_count = 0
    from .image_utils import InstanceRenderer
    renderer = InstanceRenderer(
        image_width=args.width if args.width is not None else DEFAULT_IMAGE_WIDTH,
        window_settings=window_settings,
    )

    try:
        for chunk in _chunked(selection, VIDEO_FETCH_CHUNK_SIZE):
            fetch_kwargs = {
                "max_workers": optimal_workers(len(chunk), max_workers=10, min_workers=4),
                "headers_only": False,
                "positions": chunk,
            }

            try:
                instances = series_index.get_instances(**fetch_kwargs)
            except ValueError as e:
                logger.error(f"Failed to retrieve instances: {e}")
                _shutdown_ffmpeg_process(ffmpeg_process, logger, report_errors=False)
                return 1

            for instance in instances:
                image = renderer.render_instance(instance.dataset)
                if image is None:
                    logger.error(f"Failed to render image for instance {instance.instance_uid}")
                    _shutdown_ffmpeg_process(ffmpeg_process, logger, report_errors=False)
                    return 1

                frame = image.convert("RGB")
                if frame_size is None:
                    frame_size = frame.size
                    try:
                        ffmpeg_process = _start_ffmpeg_process(
                            frame_size[0],
                            frame_size[1],
                            args.fps,
                            args.output,
                            crf=video_crf,
                        )
                    except RuntimeError as exc:
                        logger.error(str(exc))
                        return 1
                    if args.verbose:
                        logger.info(f"Video resolution: {frame_size[0]}x{frame_size[1]}")
                elif frame.size != frame_size:
                    frame = frame.resize(frame_size, Image.Resampling.LANCZOS)

                try:
                    ffmpeg_process.stdin.write(frame.tobytes())
                except BrokenPipeError as e:
                    logger.error(f"ffmpeg process terminated unexpectedly: {e}")
                    _shutdown_ffmpeg_process(ffmpeg_process, logger, report_errors=False)
                    return 1

                frame_count += 1
    except Exception as e:  # pragma: no cover - defensive future proofing
        logger.exception(f"Unexpected error while encoding video: {e}")
        _shutdown_ffmpeg_process(ffmpeg_process, logger, report_errors=False)
        return 1

    if frame_count == 0 or ffmpeg_process is None:
        logger.error("No frames were generated for the requested range")
        _shutdown_ffmpeg_process(ffmpeg_process, logger, report_errors=False)
        return 1

    rc = _shutdown_ffmpeg_process(ffmpeg_process, logger, report_errors=True)
    if rc != 0:
        return 1

    if args.verbose:
        logger.info(f"Wrote {frame_count} frames to {args.output}")
    return 0


def contrast_mosaic_command(args, logger):
    """Generate a grid comparing a DICOM instance(s) under multiple contrast settings.

    Grid layout: contrasts on x-axis (columns), instances on y-axis (rows).
    """
    try:
        from .api import SeriesIndex

        # Validate output format
        output_path = Path(args.output)
        if not _validate_output_format(output_path):
            logger.error("Output file must be .webp or .jpg/.jpeg")
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

        instances: list[Any] = []
        if args.position is not None:
            if not (0.0 <= args.position <= 1.0):
                logger.error("--position must be between 0.0 and 1.0")
                return 1
            if args.verbose:
                logger.info("Generating contrast grid from single position %.1f%%", args.position * 100)
            try:
                instance = series_index.get_instance(
                    position=args.position,
                    slice_offset=args.slice_offset,
                )
                instances = [instance]
            except ValueError as e:
                logger.error(f"Failed to retrieve instance: {e}")
                return 1
        else:
            start = args.start if args.start is not None else 0.0
            end = args.end if args.end is not None else 1.0
            if not (0.0 <= start <= 1.0) or not (0.0 <= end <= 1.0) or start > end:
                logger.error("Provide a valid --start/--end range within 0.0-1.0")
                return 1
            if args.samples < 1:
                logger.error("--samples must be >= 1")
                return 1
            if args.slice_offset != 0:
                logger.error("--slice-offset can only be used with --position")
                return 1

            if args.verbose:
                logger.info(
                    "Sampling %d slices between %.1f%% and %.1f%%",
                    args.samples,
                    start * 100,
                    end * 100,
                )
            try:
                positions = _generate_positions(series_index, args.samples, start, end)
            except ValueError as e:
                logger.error(str(e))
                return 1

            try:
                instances = series_index.get_instances(positions=positions, headers_only=False)
            except ValueError as e:
                logger.error(f"Failed to retrieve instances: {e}")
                return 1

            if not instances:
                logger.error("No DICOM instances found in requested range")
                return 1

        from .image_utils import InstanceRenderer, MosaicRenderer

        tile_rows = len(instances)
        tile_columns = num_contrasts
        if tile_rows == 0:
            logger.error("No slices available to render")
            return 1

        tile_pixel_width = args.width
        if tile_pixel_width:
            tile_pixel_width = max(1, tile_pixel_width // tile_columns)
        else:
            tile_pixel_width = max(1, 768 // tile_columns)

        if args.verbose:
            logger.info("Grid layout: %d x %d (contrasts x slices)", tile_columns, tile_rows)
            if args.width:
                logger.info(f"Target canvas width: {args.width}px")
            if args.height:
                logger.info(f"Target canvas height: {args.height}px")

        renderer_cache: dict[str, InstanceRenderer] = {}
        image_grid = []
        for inst_idx, instance in enumerate(instances):
            for contrast_idx, contrast_settings in enumerate(parsed_contrasts):
                contrast_str = args.contrast[contrast_idx]
                if args.verbose:
                    logger.info(
                        f"Instance {inst_idx+1}/{len(instances)}, "
                        f"contrast {contrast_idx+1}/{num_contrasts}: {contrast_str}"
                    )

                key = args.contrast[contrast_idx]
                renderer = renderer_cache.get(key)
                if renderer is None:
                    renderer = InstanceRenderer(
                        image_width=tile_pixel_width,
                        window_settings=contrast_settings,
                    )
                    renderer_cache[key] = renderer
                img = renderer.render_instance(instance.dataset)
                if img is None:
                    logger.error(
                        f"Failed to render instance {inst_idx+1} contrast {contrast_str}"
                    )
                    return 1
                image_grid.append(img)

        generator = MosaicRenderer(
            tile_width=tile_columns,
            tile_height=tile_rows,
            image_width=tile_pixel_width,
        )

        output_image = generator.tile_images(image_grid)

        if not output_image:
            logger.error("Failed to tile images into grid")
            return 1

        output_image = _resize_canvas_if_needed(
            output_image,
            args.width,
            args.height,
            shrink_only=getattr(args, "shrink_to_fit", False),
        )

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
        help="DICOM Series UID(s) or path(s). Provide complete UIDs (e.g., 38902e14-b11f-4548-910e-771ee757dc82) "
             "or explicit paths (e.g., s3://idc-open-data/38902e14-b11f-4548-910e-771ee757dc82). "
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
        help="DICOM Series UID or path. Provide the complete UID (e.g., 38902e14-b11f-4548-910e-771ee757dc82) "
             "or an explicit path (e.g., s3://idc-open-data/38902e14-b11f-4548-910e-771ee757dc82). "
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
        help="Optional complete series UIDs or explicit paths to remove from the cache"
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
