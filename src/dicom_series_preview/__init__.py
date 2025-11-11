"""DICOM Series Preview - Preview DICOM series stored on S3, HTTP, or local files."""

from .__main__ import main, normalize_series_uid, parse_series_specification, setup_logging
from .retriever import DICOMRetriever
from .image_utils import MosaicGenerator
from .contrast import ContrastPresets
from .slice_sorting import sort_slices, SliceParameterization
from .api import SeriesIndex, Contrast, Instance, PositionInterpolator

__version__ = "0.1.0"
__all__ = [
    "main",
    "normalize_series_uid",
    "parse_series_specification",
    "setup_logging",
    "DICOMRetriever",
    "MosaicGenerator",
    "ContrastPresets",
    "sort_slices",
    "SliceParameterization",
    "SeriesIndex",
    "Contrast",
    "Instance",
    "PositionInterpolator",
]
