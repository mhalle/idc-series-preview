"""High-level API for dicom-series-preview."""

import logging
from typing import Optional

import polars as pl

from .__main__ import _parse_and_normalize_series, setup_logging
from .index_cache import IndexCache

logger = logging.getLogger(__name__)


class SeriesIndex:
    """
    Index of a DICOM series with query and image generation methods.

    Provides access to series metadata, index querying, and image generation.

    Examples
    --------
    >>> index = SeriesIndex("38902e14-b11f-4548-910e-771ee757dc82")
    >>> print(index.instance_count)
    181
    >>> print(index.primary_axis)
    'Z'
    """

    def __init__(
        self,
        series: str,
        root: str = "s3://idc-open-data",
        cache_dir: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize a series index.

        Parameters
        ----------
        series : str
            Series UID, prefix (with *), or full path.
            Examples:
            - "38902e14-b11f-4548-910e-771ee757dc82"
            - "38902e14*"
            - "s3://idc-open-data/38902e14-b11f-4548-910e-771ee757dc82"
            - "/local/dicom/series-uid"

        root : str, default "s3://idc-open-data"
            Root storage path. Overridden if series contains full path.

        cache_dir : str or None, default None
            Cache directory for indices. If None, uses platform default.

        verbose : bool, default False
            Enable logging output.

        Raises
        ------
        ValueError
            If series cannot be resolved or index generation fails.
        """
        setup_logging(verbose)
        self._verbose = verbose
        self._logger = logging.getLogger(__name__)

        # Parse and normalize series specification
        result = _parse_and_normalize_series(
            series, root, verbose, self._logger
        )
        if result is None:
            raise ValueError(f"Could not resolve series: {series}")

        self._root_path, self._series_uid = result

        # Load or generate index
        index_df = IndexCache.load_or_generate_index(
            series_uid=self._series_uid,
            root_path=self._root_path,
            index_dir=cache_dir,
            verbose=verbose,
            logger_instance=self._logger,
        )

        if index_df is None:
            raise ValueError(
                f"Failed to generate index for series {self._series_uid}"
            )

        self._index_df = index_df
        self._cache_dir = cache_dir

    @property
    def series_uid(self) -> str:
        """Normalized series UID."""
        return self._series_uid

    @property
    def root_path(self) -> str:
        """Storage root path."""
        return self._root_path

    @property
    def instance_count(self) -> int:
        """Number of instances in series."""
        return len(self._index_df)

    @property
    def primary_axis(self) -> str:
        """
        Dominant axis used for sorting.

        Returns
        -------
        str
            One of 'X' (sagittal), 'Y' (coronal), 'Z' (axial), or 'I' (instance number)
        """
        if "PrimaryAxis" not in self._index_df.columns:
            return "I"  # Default to instance number

        axes = self._index_df["PrimaryAxis"].unique().to_list()
        if len(axes) == 1:
            return axes[0]

        # If mixed (shouldn't happen), return most common
        axis_counts = (
            self._index_df.select("PrimaryAxis")
            .group_by("PrimaryAxis")
            .count()
            .sort("count", descending=True)
        )
        return axis_counts["PrimaryAxis"][0]

    @property
    def position_range(self) -> tuple[float, float]:
        """
        Min and max position on primary axis.

        Returns
        -------
        tuple[float, float]
            (min_position, max_position)
        """
        if "PrimaryPosition" not in self._index_df.columns:
            return (0.0, float(self.instance_count - 1))

        min_pos = self._index_df["PrimaryPosition"].min()
        max_pos = self._index_df["PrimaryPosition"].max()

        return (float(min_pos), float(max_pos))

    def __len__(self) -> int:
        """Instance count."""
        return self.instance_count

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SeriesIndex("
            f"uid='{self._series_uid}', "
            f"instances={self.instance_count}, "
            f"axis='{self.primary_axis}'"
            f")"
        )
