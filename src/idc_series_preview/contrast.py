"""DICOM window/level (contrast) presets."""

import logging
from typing import Dict, List, Optional, Union
import numpy as np


logger = logging.getLogger(__name__)


class ContrastPresets:
    """Standard DICOM window/level presets for different anatomies."""

    # Pre-compute tanh lookup table once
    _TANH_LUT = None
    _TANH_LUT_SIZE = 1024

    @classmethod
    def _get_tanh_lut(cls):
        """Get or create the pre-computed tanh lookup table."""
        if cls._TANH_LUT is None:
            lut = np.zeros(cls._TANH_LUT_SIZE, dtype=np.uint8)
            for i in range(cls._TANH_LUT_SIZE):
                # Normalized position: -3 to +3
                normalized = (i / cls._TANH_LUT_SIZE) * 6 - 3
                # Apply tanh: maps (-inf, inf) to (-1, 1)
                tanh_val = np.tanh(normalized)
                # Scale to 0-255
                lut[i] = int((tanh_val + 1) / 2 * 255)
            cls._TANH_LUT = lut
        return cls._TANH_LUT

    PRESET_DEFINITIONS: List[Dict[str, object]] = [
        {"names": ["ct-lung", "lung"], "window_width": 1500, "window_center": -500},
        {"names": ["ct-bone", "bone"], "window_width": 2000, "window_center": 300},
        {"names": ["ct-abdomen", "abdomen"], "window_width": 350, "window_center": 50},
        {"names": ["ct-brain", "brain"], "window_width": 80, "window_center": 40},
        {"names": ["ct-mediastinum", "mediastinum", "media"], "window_width": 350, "window_center": 50},
        {"names": ["ct-vascular", "vascular"], "window_width": 700, "window_center": 200},
        {"names": ["ct-liver", "liver"], "window_width": 150, "window_center": 30},
        {"names": ["ct-soft-tissue", "soft-tissue", "soft"], "window_width": 400, "window_center": 50},
        {"names": ["mr-t1", "t1"], "window_width": 700, "window_center": 300},
        {"names": ["mr-t2", "t2"], "window_width": 475, "window_center": 155},
        {"names": ["mr-proton", "proton"], "window_width": 920, "window_center": 420},
    ]

    PRESETS: Dict[str, Dict[str, float]] = {}
    SHORTCUTS: Dict[str, str] = {}

    for definition in PRESET_DEFINITIONS:
        names = definition.get("names", [])
        if not names:
            continue
        canonical = names[0].lower()
        preset_values = {
            "window_width": definition["window_width"],
            "window_center": definition["window_center"],
        }
        PRESETS[canonical] = preset_values
        for alias in names[1:]:
            SHORTCUTS[alias.lower()] = canonical

    @classmethod
    def get_preset(cls, name: str) -> Optional[Dict[str, float]]:
        """
        Get a preset by name or shortcut.

        Args:
            name: Preset name (e.g., 'lung', 'bone', 'soft-tissue') or shortcut (e.g., 'soft', 'media')

        Returns:
            Dict with 'window_width' and 'window_center' keys, or None if not found
        """
        name_lower = name.lower()

        # Check for shortcuts first
        if name_lower in cls.SHORTCUTS:
            name_lower = cls.SHORTCUTS[name_lower]

        preset = cls.PRESETS.get(name_lower)
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
        p2 = np.nanpercentile(pixel_array, 2)
        p98 = np.nanpercentile(pixel_array, 98)

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

    @classmethod
    def parse_wl_string(cls, spec: str) -> Optional[Dict[str, float]]:
        """
        Parse a window/level string like '1500/500' or '1500,-500'.

        Args:
            spec: String in format 'WW/WL', 'WW,WL', or 'WW/L-NNN'

        Returns:
            Dict with 'window_width' and 'window_center', or None if not parseable
        """
        if "/" in spec:
            parts = spec.split("/")
        elif "," in spec:
            parts = spec.split(",")
        else:
            return None

        if len(parts) != 2:
            return None

        try:
            ww = float(parts[0].strip())
            wc = float(parts[1].strip())
            return {"window_width": ww, "window_center": wc}
        except ValueError:
            return None

    @classmethod
    def resolve_contrast(
        cls,
        contrast_spec: Optional[str],
        dataset: Optional["pydicom.Dataset"] = None,
        pixel_array: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Resolve a contrast specification to actual W/L values.

        Args:
            contrast_spec: Contrast specification (preset name, 'auto', 'embedded', 'WW/WL', or None)
            dataset: DICOM dataset (needed for 'embedded' mode)
            pixel_array: Pixel data (needed for 'auto' mode)

        Returns:
            Dict with 'window_width' and 'window_center', or None if cannot resolve
        """
        if contrast_spec is None:
            # Try embedded, then auto
            if dataset is not None:
                ww = getattr(dataset, "WindowWidth", None)
                wc = getattr(dataset, "WindowCenter", None)
                if ww is not None and wc is not None:
                    # Handle MultiValue
                    if hasattr(ww, "__getitem__"):
                        ww = ww[0]
                    if hasattr(wc, "__getitem__"):
                        wc = wc[0]
                    return {"window_width": float(ww), "window_center": float(wc)}
            if pixel_array is not None:
                return cls.auto_detect(pixel_array)
            return None

        if contrast_spec == "auto":
            if pixel_array is not None:
                return cls.auto_detect(pixel_array)
            return None

        if contrast_spec == "embedded":
            if dataset is not None:
                ww = getattr(dataset, "WindowWidth", None)
                wc = getattr(dataset, "WindowCenter", None)
                if ww is not None and wc is not None:
                    if hasattr(ww, "__getitem__"):
                        ww = ww[0]
                    if hasattr(wc, "__getitem__"):
                        wc = wc[0]
                    return {"window_width": float(ww), "window_center": float(wc)}
            # Fall back to auto
            if pixel_array is not None:
                return cls.auto_detect(pixel_array)
            return None

        # Check if it's a preset
        preset = cls.get_preset(contrast_spec)
        if preset:
            return preset

        # Try to parse as WW/WL string
        parsed = cls.parse_wl_string(contrast_spec)
        if parsed:
            return parsed

        return None

    @classmethod
    def apply_windowing(
        cls,
        pixel_array: np.ndarray,
        window_width: float,
        window_center: float,
    ) -> np.ndarray:
        """
        Apply window/level adjustment to pixel array with linear windowing.

        Args:
            pixel_array: Input pixel array
            window_width: Window width in HU
            window_center: Window center in HU

        Returns:
            Adjusted pixel array (uint8, 0-255)
        """
        if window_width <= 0:
            return np.zeros_like(pixel_array, dtype=np.uint8)

        # Calculate lower and upper bounds
        c = window_center
        w = window_width

        below = c - w / 2
        above = c + w / 2

        # Apply windowing with hard clipping
        windowed = np.clip(pixel_array, below, above)

        # Scale to 0-255
        windowed = ((windowed - below) / w) * 255

        return windowed.astype(np.uint8)
