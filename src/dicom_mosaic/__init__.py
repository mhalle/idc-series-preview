"""DICOM Mosaic Generator - Generate tiled mosaic images from DICOM series."""

from .__main__ import main, normalize_series_uid, parse_series_specification, setup_logging
from .retriever import DICOMRetriever
from .mosaic import MosaicGenerator
from .contrast import ContrastPresets

__version__ = "0.1.0"
__all__ = [
    "main",
    "normalize_series_uid",
    "parse_series_specification",
    "setup_logging",
    "DICOMRetriever",
    "MosaicGenerator",
    "ContrastPresets",
]
