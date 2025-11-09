"""DICOM window/level (contrast) presets."""

import logging
from typing import Dict, Optional, Union
import numpy as np


logger = logging.getLogger(__name__)


class ContrastPresets:
    """Standard DICOM window/level presets for different anatomies."""

    PRESETS: Dict[str, Dict[str, float]] = {
        # Lung
        "lung": {
            "window_width": 1500,
            "window_center": -500,
        },
        # Bone
        "bone": {
            "window_width": 2000,
            "window_center": 300,
        },
        # Abdomen
        "abdomen": {
            "window_width": 350,
            "window_center": 50,
        },
        # Brain
        "brain": {
            "window_width": 80,
            "window_center": 40,
        },
        # Mediastinum
        "mediastinum": {
            "window_width": 350,
            "window_center": 50,
        },
        # Liver
        "liver": {
            "window_width": 150,
            "window_center": 30,
        },
        # Soft tissue
        "soft_tissue": {
            "window_width": 400,
            "window_center": 50,
        },
    }

    @classmethod
    def get_preset(cls, name: str) -> Optional[Dict[str, float]]:
        """
        Get a preset by name.

        Args:
            name: Preset name (e.g., 'lung', 'bone')

        Returns:
            Dict with 'window_width' and 'window_center' keys, or None if not found
        """
        preset = cls.PRESETS.get(name.lower())
        if preset and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Using '{name}' contrast preset: WW={preset['window_width']}, WC={preset['window_center']}")
        return preset

    @classmethod
    def auto_detect(cls, pixel_array: np.ndarray) -> Dict[str, float]:
        """
        Auto-detect window and center from pixel array statistics.

        Args:
            pixel_array: NumPy array of pixel values

        Returns:
            Dict with 'window_width' and 'window_center'
        """
        # Use percentile-based approach for auto-detection
        p2 = np.percentile(pixel_array, 2)
        p98 = np.percentile(pixel_array, 98)

        window_center = (p2 + p98) / 2
        window_width = p98 - p2

        # Ensure minimum window width
        if window_width < 10:
            window_width = 10

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Auto-detected contrast: WW={window_width:.1f}, WC={window_center:.1f}")
        return {
            "window_width": window_width,
            "window_center": window_center,
        }

    @staticmethod
    def apply_windowing(
        pixel_array: np.ndarray,
        window_width: float,
        window_center: float,
    ) -> np.ndarray:
        """
        Apply window/level adjustment to pixel array.

        Args:
            pixel_array: Input pixel array
            window_width: Window width in HU
            window_center: Window center in HU

        Returns:
            Adjusted pixel array (uint8, 0-255)
        """
        # Calculate lower and upper bounds
        c = window_center
        w = window_width

        below = c - w / 2
        above = c + w / 2

        # Apply windowing
        windowed = np.clip(pixel_array, below, above)

        # Scale to 0-255
        if w > 0:
            windowed = ((windowed - below) / w) * 255
        else:
            windowed = np.zeros_like(windowed)

        return windowed.astype(np.uint8)
