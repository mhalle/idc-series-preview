"""Utility functions for image handling and output."""

import io
import logging
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pydicom
try:
    # pydicom >=3.0
    from pydicom.pixels import apply_modality_lut, apply_windowing
except ImportError:  # pragma: no cover - fallback for older pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_windowing

from .contrast import ContrastPresets


logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def get_label_font(size: int = 10) -> ImageFont.FreeTypeFont:
    """
    Load the bundled Oswald Bold font at the specified size.

    Uses importlib.resources for reliable package resource access.
    Results are cached to avoid repeated disk/memory access.

    Parameters
    ----------
    size : int, default 10
        Font size in points

    Returns
    -------
    PIL.ImageFont.FreeTypeFont
        The loaded font object
    """
    font_bytes = files("idc_series_preview.fonts").joinpath("Oswald-Bold.ttf").read_bytes()
    return ImageFont.truetype(io.BytesIO(font_bytes), size=size)


def add_label(
    image: Image.Image,
    text: str,
    position: str = "tr",
    font_size: int = 14,
    padding: int = 1,
    margin: int = 0,
) -> Image.Image:
    """
    Add a text label overlay to an image.

    Parameters
    ----------
    image : PIL.Image.Image
        Image to add label to (modified in place if RGB, converted if grayscale)
    text : str
        Label text
    position : str, default "tr"
        Corner position: "tl" (top-left), "tr" (top-right),
        "bl" (bottom-left), "br" (bottom-right)
    font_size : int, default 14
        Font size in points
    padding : int, default 1
        Padding inside the label background
    margin : int, default 0
        Margin from image edge

    Returns
    -------
    PIL.Image.Image
        Image with label added (RGB mode)
    """
    # Convert to RGB if grayscale
    if image.mode == "L":
        image = image.convert("RGB")

    draw = ImageDraw.Draw(image)
    font = get_label_font(font_size)

    # Get text bounding box using anchor="lt" for consistent positioning
    bbox = draw.textbbox((0, 0), text, font=font, anchor="lt")
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Calculate background box position based on corner
    img_w, img_h = image.size
    if position == "tl":
        bg_x, bg_y = margin, margin
    elif position == "tr":
        bg_x = img_w - margin - text_w - padding * 2
        bg_y = margin
    elif position == "bl":
        bg_x = margin
        bg_y = img_h - margin - text_h - padding * 2
    elif position == "br":
        bg_x = img_w - margin - text_w - padding * 2
        bg_y = img_h - margin - text_h - padding * 2
    else:
        raise ValueError(f"Invalid position: {position}. Use 'tl', 'tr', 'bl', or 'br'.")

    # Draw background and text
    bg_box = (bg_x, bg_y, bg_x + text_w + padding * 2, bg_y + text_h + padding * 2)
    draw.rectangle(bg_box, fill="white")

    # Draw text with anchor="lt" for proper alignment
    draw.text((bg_x + padding, bg_y + padding), text, fill="black", font=font, anchor="lt")

    return image


# Minimum image width for labels (skip if smaller)
MIN_LABEL_WIDTH = 64


def format_position_label(position: float, axis: str = "") -> str:
    """
    Format normalized position as a label string.

    Parameters
    ----------
    position : float
        Normalized position (0.0 to 1.0)
    axis : str, default ""
        Axis prefix (e.g., "Z", "X", "Y", "I") or empty string for no prefix

    Returns
    -------
    str
        Formatted label like "Z:0.5000"
    """
    if axis:
        return f"{axis}:{position:.4f}"
    return f"{position:.4f}"


def format_contrast_label(window_width: float, window_center: float) -> str:
    """
    Format window/level contrast settings as a label string.

    Parameters
    ----------
    window_width : float
        Window width value
    window_center : float
        Window center/level value

    Returns
    -------
    str
        Formatted label like "W1500/L-500"
    """
    return f"W{int(window_width)}/L{int(window_center)}"


def format_primary_position_label(position: float, axis: str = "") -> str:
    """
    Format primary position (actual DICOM coordinate) as a label string.

    Parameters
    ----------
    position : float
        Primary position value (e.g., Z coordinate from ImagePositionPatient)
    axis : str, default ""
        Axis prefix (e.g., "Z", "X", "Y") or empty string for no prefix

    Returns
    -------
    str
        Formatted label like "Z:-156.5"
    """
    if axis:
        return f"{axis}:{position:.1f}"
    return f"{position:.1f}"


def add_image_labels(
    image: Image.Image,
    labels: Dict[str, str],
    font_size: int = 14,
    margin: int = 2,
) -> Image.Image:
    """
    Add text labels to an image at specified corner positions.

    Labels are skipped if image width is less than MIN_LABEL_WIDTH (64px).

    Parameters
    ----------
    image : PIL.Image.Image
        Image to add labels to
    labels : Dict[str, str]
        Mapping of position to label text. Valid positions are:
        "tl" (top-left), "tr" (top-right), "bl" (bottom-left), "br" (bottom-right)
    font_size : int, default 14
        Font size in points
    margin : int, default 2
        Margin from image edge in pixels

    Returns
    -------
    PIL.Image.Image
        Image with labels added (RGB mode), or original if too small

    Examples
    --------
    >>> labels = {
    ...     "bl": format_position_label(0.5),
    ...     "br": format_contrast_label(1500, -500),
    ... }
    >>> labeled_image = add_image_labels(image, labels)
    """
    # Skip labels on small images
    if image.width < MIN_LABEL_WIDTH:
        return image

    for position, text in labels.items():
        image = add_label(image, text, position=position, font_size=font_size, margin=margin)

    return image


def save_image(
    image: Image.Image,
    output_path: str,
    quality: int = 85,
) -> bool:
    """
    Save PIL Image to file with quality settings.

    Supports WebP and JPEG output formats.

    Parameters
    ----------
    image : PIL.Image.Image
        Image to save
    output_path : str
        Output file path (.webp or .jpg/.jpeg)
    quality : int, default 85
        Quality (0-100)

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        output_path = Path(output_path)
        suffix = output_path.suffix.lower()

        if suffix == '.webp':
            image.save(output_path, 'WEBP', quality=quality)
        elif suffix in ['.jpg', '.jpeg']:
            image.save(output_path, 'JPEG', quality=quality)
        else:
            logger.error(f"Unsupported format: {suffix}")
            return False

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Saved image to {output_path}")

        return True

    except Exception as e:
        logger.error(f"Failed to save image: {e}")
        return False


class InstanceRenderer:
    """Render individual DICOM instances into PIL images."""

    def __init__(
        self,
        image_width: Optional[int] = 128,
        window_settings: Optional[Union[str, Dict[str, float]]] = None,
    ):
        self.image_width = image_width
        self.window_settings = window_settings

    def render_instance(
        self,
        dataset: pydicom.Dataset,
        *,
        instance_uid: Optional[str] = None,
        retriever=None,
        series_uid: Optional[str] = None,
    ) -> Optional[Image.Image]:
        pixel_array = self._extract_pixel_array(dataset)
        ds_for_windowing = dataset

        if pixel_array is None and retriever and series_uid and instance_uid:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Fetching full instance data for {instance_uid}")
            ds_full = retriever.get_instance_data(series_uid, instance_uid)
            if ds_full is not None:
                pixel_array = self._extract_pixel_array(ds_full)
                ds_for_windowing = ds_full

        if pixel_array is None:
            logger.error(
                "No pixel data available for rendering"
                + (f" (instance {instance_uid})" if instance_uid else "")
            )
            return None

        img = self._pixel_array_to_image(pixel_array, ds_for_windowing)
        return self._resize_image(img)


    def _extract_pixel_array(self, ds: pydicom.Dataset) -> Optional[np.ndarray]:
        """
        Extract pixel array from DICOM dataset.

        Args:
            ds: pydicom.Dataset

        Returns:
            NumPy array of pixel values or None
        """
        try:
            # Use pydicom's modality LUT handling (slope/intercept, LUTs, per-frame)
            pixel_array = apply_modality_lut(ds.pixel_array, ds)

            return pixel_array

        except Exception as e:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Failed to extract pixel array: {e}")
            return None

    def _get_window_settings(
        self, pixel_array: np.ndarray, ds: Optional[pydicom.Dataset] = None
    ) -> Dict[str, float]:
        """
        Determine window/center settings based on priority:
        1. If user specifies 'embedded': use DICOM file values, fall back to auto
        2. If user specifies 'auto': auto-detect from pixel statistics
        3. If user specifies preset or values: use those
        4. Otherwise: try file values, then fall back to auto

        Args:
            pixel_array: Pixel data
            ds: Optional DICOM dataset to read stored window/center values

        Returns:
            Dict with window_width and window_center
        """
        # Remove padding from stats when available
        if ds is not None and hasattr(ds, "PixelPaddingValue"):
            padding = ds.PixelPaddingValue
            if hasattr(ds, "PixelPaddingRangeLimit"):
                lo, hi = sorted([padding, ds.PixelPaddingRangeLimit])
                mask = (pixel_array >= lo) & (pixel_array <= hi)
            else:
                mask = pixel_array == padding
            pixel_array = np.where(mask, np.nan, pixel_array)

        # If user explicitly requests embedded window/level
        if self.window_settings == "embedded":
            # Try to get from file first
            if ds is not None:
                file_settings = self._get_file_window_settings(ds)
                if file_settings:
                    return file_settings
            # Fall back to auto if not in file
            return ContrastPresets.auto_detect(pixel_array)

        # If user requests auto-detection
        if self.window_settings == "auto":
            return ContrastPresets.auto_detect(pixel_array)

        # If user provides dict or preset
        if isinstance(self.window_settings, dict):
            return self.window_settings
        elif isinstance(self.window_settings, str):
            # Treat as preset name
            preset = ContrastPresets.get_preset(self.window_settings)
            if preset:
                return preset

        # Default behavior: try file, then auto
        if ds is not None:
            file_settings = self._get_file_window_settings(ds)
            if file_settings:
                return file_settings

        return ContrastPresets.auto_detect(pixel_array)

    def _get_file_window_settings(self, ds: pydicom.Dataset) -> Optional[Dict[str, float]]:
        """
        Extract window/center settings from DICOM file metadata.

        Args:
            ds: pydicom.Dataset

        Returns:
            Dict with window_width and window_center, or None if not found
        """
        try:
            if hasattr(ds, 'WindowWidth') and hasattr(ds, 'WindowCenter'):
                # WindowWidth/WindowCenter can be single value or list/MultiValue
                ww = ds.WindowWidth
                wc = ds.WindowCenter

                # Handle lists/sequences/MultiValue - use first one
                if hasattr(ww, '__getitem__'):  # Sequence-like (list, tuple, MultiValue)
                    ww = ww[0]
                if hasattr(wc, '__getitem__'):  # Sequence-like
                    wc = wc[0]

                window_settings = {
                    "window_width": float(ww),
                    "window_center": float(wc)
                }
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Using file window settings: WW={ww}, WC={wc}")
                return window_settings
        except Exception as e:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Could not read window settings from file: {e}")
        return None

    def _pixel_array_to_image(self, pixel_array: np.ndarray, ds: Optional[pydicom.Dataset] = None) -> Image.Image:
        """
        Convert pixel array to PIL Image.

        Args:
            pixel_array: NumPy array of pixel values
            ds: Optional DICOM dataset for reading stored window/center values

        Returns:
            PIL Image object
        """
        # Prefer DICOM VOI/LUT when using embedded/default behavior and dataset is available
        has_voi = False
        if ds is not None:
            has_voi = hasattr(ds, "VOILUTSequence") or hasattr(ds, "WindowCenter")

        if ds is not None and (self.window_settings is None or self.window_settings == "embedded") and has_voi:
            try:
                windowed = apply_windowing(
                    pixel_array,
                    ds,
                    out_dtype=np.uint8,
                    out_min_max=(0, 255),
                )
            except Exception as e:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"apply_windowing failed, falling back to custom windowing: {e}")
                windowed = None
        else:
            windowed = None

        # Fall back to user presets/auto/custom handling
        if windowed is None:
            window_settings = self._get_window_settings(pixel_array, ds)
            windowed = ContrastPresets.apply_windowing(
                pixel_array,
                window_settings['window_width'],
                window_settings['window_center'],
            )

        # Ensure uint8 dtype
        if windowed.dtype != np.uint8:
            windowed = windowed.astype(np.uint8)

        # Handle multi-frame or RGB data
        if windowed.ndim == 3:
            # Multi-frame or RGB - use first frame
            windowed = windowed[0] if windowed.shape[0] == 1 else windowed

        # Convert to PIL Image with explicit 8-bit mode
        if windowed.ndim == 2:
            img = Image.fromarray(windowed, mode='L')
        else:
            # Assume RGB or similar
            img = Image.fromarray(windowed)

        return img

    def _resize_image(self, img: Image.Image) -> Image.Image:
        """
        Resize image to tile width while maintaining aspect ratio.

        Args:
            img: PIL Image

        Returns:
            Resized PIL Image
        """
        if self.image_width is None or self.image_width <= 0:
            return img

        width, height = img.size
        aspect_ratio = height / width
        new_height = int(self.image_width * aspect_ratio)
        return img.resize((self.image_width, new_height), Image.Resampling.LANCZOS)

class MosaicRenderer:
    """Generate mosaics from a collection of pre-rendered images."""

    def __init__(
        self,
        tile_width: int = 6,
        tile_height: int = 6,
        image_width: int = 128,
    ):
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.image_width = image_width

    def tile_images(self, images: List[Image.Image]) -> Optional[Image.Image]:
        """
        Tile a list of pre-processed PIL Images into a mosaic.

        Args:
            images: List of PIL Image objects (all should be same size or will be standardized)

        Returns:
            PIL Image of the tiled mosaic, or None if tiling failed
        """
        if not images:
            logger.error("No images provided for tiling")
            return None

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Tiling {len(images)} pre-processed images")

        # Pad images list to fill grid
        total_tiles = self.tile_width * self.tile_height
        while len(images) < total_tiles:
            # Create blank image with same dimensions as last image
            blank = Image.new('L', images[-1].size, color=0)
            images.append(blank)

        # Trim to exact tile count
        images = images[:total_tiles]

        # Calculate mosaic dimensions
        # Use the maximum width and height from all images
        max_width = max(img.width for img in images)
        max_height = max(img.height for img in images)

        # Ensure all tiles are same size
        standardized_images = []
        for img in images:
            if img.size != (max_width, max_height):
                # Pad image to max size
                padded = Image.new('L', (max_width, max_height), color=0)
                offset = ((max_width - img.width) // 2, (max_height - img.height) // 2)
                padded.paste(img, offset)
                standardized_images.append(padded)
            else:
                standardized_images.append(img)

        # Create mosaic
        mosaic_width = max_width * self.tile_width
        mosaic_height = max_height * self.tile_height
        mosaic = Image.new('L', (mosaic_width, mosaic_height), color=0)

        for idx, img in enumerate(standardized_images):
            row = idx // self.tile_width
            col = idx % self.tile_width
            x = col * max_width
            y = row * max_height
            mosaic.paste(img, (x, y))

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Created {self.tile_width}x{self.tile_height} mosaic ({mosaic_width}x{mosaic_height}px)")
        return mosaic
