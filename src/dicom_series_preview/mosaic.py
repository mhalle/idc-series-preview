"""DICOM mosaic creation and image output."""

import logging
from typing import List, Tuple, Optional, Union, Dict, Any
import numpy as np
from PIL import Image
import pydicom
from pathlib import Path

from .contrast import ContrastPresets


logger = logging.getLogger(__name__)


class MosaicGenerator:
    """Generate DICOM mosaics from a collection of instances."""

    def __init__(
        self,
        tile_width: int = 6,
        tile_height: int = 6,
        image_width: int = 128,
        window_settings: Optional[Union[str, Dict[str, float]]] = None,
    ):
        """
        Initialize mosaic generator.

        Args:
            tile_width: Number of images per row
            tile_height: Number of images per column
            image_width: Width of each tile in pixels (height scaled proportionally)
            window_settings: One of:
                - 'auto': auto-detect from pixel statistics
                - 'embedded': use window/level from DICOM file (fall back to auto if not present)
                - preset name (lung, bone, brain, etc.)
                - dict with window_width/window_center keys
        """
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.image_width = image_width
        self.window_settings = window_settings

    def _extract_pixel_array(self, ds: pydicom.Dataset) -> Optional[np.ndarray]:
        """
        Extract pixel array from DICOM dataset.

        Args:
            ds: pydicom.Dataset

        Returns:
            NumPy array of pixel values or None
        """
        try:
            pixel_array = ds.pixel_array

            # Handle DICOM rescale/slope/intercept
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                slope = float(ds.RescaleSlope)
                intercept = float(ds.RescaleIntercept)
                pixel_array = pixel_array * slope + intercept

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
        # Get window settings for this image
        window_settings = self._get_window_settings(pixel_array, ds)

        # Apply windowing
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
        width, height = img.size
        aspect_ratio = height / width
        new_height = int(self.image_width * aspect_ratio)
        return img.resize((self.image_width, new_height), Image.Resampling.LANCZOS)

    def create_single_image(
        self, instance: Tuple[str, pydicom.Dataset], retriever=None, series_uid=None
    ) -> Optional[Image.Image]:
        """
        Create a single image from a DICOM instance (no tiling).

        Args:
            instance: Tuple of (instance_uid, pydicom.Dataset)
            retriever: Optional DICOMRetriever for fetching full instance data when headers only are present
            series_uid: Series UID for fetching full instance data

        Returns:
            PIL Image, or None if creation failed
        """
        instance_uid, ds = instance

        try:
            pixel_array = self._extract_pixel_array(ds)
            ds_for_windowing = ds

            # If no pixel data and we have a retriever, fetch full instance
            if pixel_array is None and retriever and series_uid:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Fetching full instance data for {instance_uid}")
                ds_full = retriever.get_instance_data(series_uid, instance_uid)
                if ds_full:
                    pixel_array = self._extract_pixel_array(ds_full)
                    ds_for_windowing = ds_full

            if pixel_array is None:
                logger.error(f"No pixel data for instance {instance_uid}")
                return None

            img = self._pixel_array_to_image(pixel_array, ds_for_windowing)
            img = self._resize_image(img)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Created single image {img.width}x{img.height}px from {instance_uid}")

            return img

        except Exception as e:
            logger.error(f"Failed to process instance {instance_uid}: {e}")
            return None

    def create_mosaic(
        self, instances: List[Tuple[str, pydicom.Dataset]], retriever=None, series_uid=None
    ) -> Optional[Image.Image]:
        """
        Create a mosaic from DICOM instances.

        Args:
            instances: List of (instance_uid, pydicom.Dataset) tuples
            retriever: Optional DICOMRetriever for fetching full instance data when headers only are present
            series_uid: Series UID for fetching full instance data

        Returns:
            PIL Image of the mosaic, or None if creation failed
        """
        if not instances:
            logger.error("No instances provided")
            return None

        # Sort instances by z-position (slice order)
        def get_z_position(item):
            instance_uid, ds = item
            # Try ImagePositionPatient (z coordinate)
            if hasattr(ds, 'ImagePositionPatient') and len(ds.ImagePositionPatient) >= 3:
                return float(ds.ImagePositionPatient[2])
            # Fall back to InstanceNumber
            elif hasattr(ds, 'InstanceNumber'):
                return float(ds.InstanceNumber)
            # Fall back to SliceLocation
            elif hasattr(ds, 'SliceLocation'):
                return float(ds.SliceLocation)
            # Default: return 0 to maintain order
            return 0

        instances = sorted(instances, key=get_z_position)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Sorted {len(instances)} instances by z-position")

        # Convert DICOM data to images
        images = []
        for instance_uid, ds in instances:
            try:
                pixel_array = self._extract_pixel_array(ds)
                ds_for_windowing = ds

                # If no pixel data and we have a retriever, fetch full instance
                if pixel_array is None and retriever and series_uid:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Fetching full instance data for {instance_uid}")
                    ds_full = retriever.get_instance_data(series_uid, instance_uid)
                    if ds_full:
                        pixel_array = self._extract_pixel_array(ds_full)
                        ds_for_windowing = ds_full

                if pixel_array is None:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Skipping {instance_uid} - no pixel data")
                    continue

                img = self._pixel_array_to_image(pixel_array, ds_for_windowing)
                img = self._resize_image(img)
                images.append(img)

            except Exception as e:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Failed to process instance {instance_uid}: {e}")
                continue

        if not images:
            logger.error("No valid images could be processed. This may occur if DICOM files use unsupported compression codecs (e.g., JPEG Extended with 12-bit precision).")
            return None

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Created {len(images)} processed images for mosaic")

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

    def save_image(
        self,
        image: Image.Image,
        output_path: str,
        quality: int = 85,
    ) -> bool:
        """
        Save mosaic image to file.

        Args:
            image: PIL Image
            output_path: Output file path (.webp or .jpg)
            quality: Quality (0-100)

        Returns:
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
                logger.debug(f"Saved mosaic to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            return False
