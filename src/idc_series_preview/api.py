"""High-level API for idc-series-preview."""

import logging
from typing import Optional, Union

import numpy as np
import polars as pl
import pydicom
from PIL import Image

from .series_spec import parse_and_normalize_series
from .index_cache import load_or_generate_index
from .retriever import DICOMRetriever
from .image_utils import InstanceRenderer

logger = logging.getLogger(__name__)


class PositionInterpolator:
    """Generate evenly-spaced positions across a normalized range.

    Useful for sampling DICOM series at regular intervals or creating
    grid layouts (mosaic, contrast grids, etc.).

    Examples
    --------
    >>> interp = PositionInterpolator(instance_count=100)
    >>> positions = interp.interpolate(num_positions=6)  # 6 evenly-spaced positions
    >>> positions = interp.interpolate(num_positions=6, start=0.3, end=0.7)  # middle 40%
    """

    def __init__(self, instance_count: int):
        """
        Initialize interpolator for a series with given instance count.

        Parameters
        ----------
        instance_count : int
            Number of instances in the series
        """
        if instance_count < 1:
            raise ValueError(f"instance_count must be >= 1, got {instance_count}")
        self.instance_count = instance_count

    def interpolate(
        self,
        num_positions: int,
        start: float = 0.0,
        end: float = 1.0,
    ) -> list[float]:
        """
        Generate evenly-spaced positions within a normalized range.

        Parameters
        ----------
        num_positions : int
            Number of positions to generate
        start : float, default 0.0
            Start of normalized range (0.0-1.0)
        end : float, default 1.0
            End of normalized range (0.0-1.0)

        Returns
        -------
        list[float]
            Evenly-spaced positions in [start, end]

        Raises
        ------
        ValueError
            If num_positions < 1 or range invalid
        """
        if num_positions < 1:
            raise ValueError(f"num_positions must be >= 1, got {num_positions}")
        if not (0.0 <= start <= 1.0):
            raise ValueError(f"start must be in [0.0, 1.0], got {start}")
        if not (0.0 <= end <= 1.0):
            raise ValueError(f"end must be in [0.0, 1.0], got {end}")
        if start > end:
            raise ValueError(f"start must be <= end, got start={start}, end={end}")

        # Single position: return midpoint of range
        if num_positions == 1:
            return [(start + end) / 2.0]

        # Multiple positions: evenly-spaced across range
        positions = []
        for i in range(num_positions):
            pos = start + (end - start) * i / (num_positions - 1)
            positions.append(pos)

        return positions

    def interpolate_unique(
        self,
        num_positions: int,
        start: float = 0.0,
        end: float = 1.0,
    ) -> tuple[list[float], list[int]]:
        """Return evenly spaced positions deduplicated by slice index.

        Useful when requesting many samples over a tiny range: instead of
        repeating the same slice multiple times, this method collapses the
        selection to the unique slice indices available between ``start`` and
        ``end``.

        Returns
        -------
        (positions, indices)
            positions: list of normalized positions (subset of interpolate())
            indices: corresponding zero-indexed slice numbers
        """
        positions = self.interpolate(num_positions, start=start, end=end)

        start_idx = self.position_to_index(start)
        end_idx = self.position_to_index(end)
        max_unique = max(1, end_idx - start_idx + 1)

        unique_positions: list[float] = []
        unique_indices: list[int] = []
        seen_indices: set[int] = set()

        for pos in positions:
            idx = self.position_to_index(pos)
            if idx in seen_indices:
                continue
            seen_indices.add(idx)
            unique_positions.append(pos)
            unique_indices.append(idx)

            if len(unique_positions) >= max_unique:
                break

        return unique_positions, unique_indices

    def position_to_index(
        self, position: float, slice_offset: int = 0
    ) -> int:
        """
        Convert normalized position to slice index with optional offset.

        Parameters
        ----------
        position : float
            Normalized position (0.0-1.0)
        slice_offset : int, optional
            Offset from the calculated index. Default: 0.
            Examples:
            - 1: next slice
            - -1: previous slice
            - 0: no offset (default)

        Returns
        -------
        int
            Zero-indexed slice number

        Raises
        ------
        ValueError
            If position is out of range or offset results in out-of-bounds index
        """
        if not (0.0 <= position <= 1.0):
            raise ValueError(f"position must be in [0.0, 1.0], got {position}")

        # Map position to slice index
        base_index = int(position * (self.instance_count - 1))

        # Apply offset
        target_index = base_index + slice_offset

        # Validate bounds
        if target_index < 0 or target_index >= self.instance_count:
            raise ValueError(
                f"Slice offset {slice_offset} out of bounds: "
                f"position {position:.1%} → index {base_index}, "
                f"+ offset {slice_offset} = {target_index}, "
                f"but valid range is 0-{self.instance_count - 1}"
            )

        return target_index


class Contrast:
    """Contrast specification for image rendering.

    Supports multiple ways to specify contrast:
    - Preset names: 'lung', 'bone', 'brain', etc.
    - Auto-detection: 'auto' or 'embedded'
    - Window/level values: '1500/500' or '1500,-500' or '1500,500'
    - Direct parameters: window_width=1500, window_center=500
    """

    def __init__(
        self,
        spec: Optional[str] = None,
        window_width: Optional[float] = None,
        window_center: Optional[float] = None,
    ):
        """
        Initialize contrast specification.

        Parameters
        ----------
        spec : str, optional
            Contrast specification. Can be:
            - Preset name: 'lung', 'bone', 'brain', etc.
            - Auto: 'auto' or 'embedded'
            - Window/level: '1500/500' or '1500,-500' or '1500,500'

        window_width : float, optional
            Window width for custom window/level

        window_center : float, optional
            Window center (level) for custom window/level
        """
        self.spec = spec
        self.window_width = window_width
        self.window_center = window_center


class Instance:
    """A single DICOM instance with rendering methods.

    Wraps a pydicom.Dataset and provides image generation methods.
    The dataset is cached in memory for efficient multi-contrast rendering.
    """

    def __init__(
        self,
        instance_uid: str,
        dataset: "pydicom.Dataset",
        series_index: "SeriesIndex",
    ):
        """
        Initialize a DICOM instance.

        Parameters
        ----------
        instance_uid : str
            The DICOM instance UID
        dataset : pydicom.Dataset
            The DICOM dataset object
        series_index : SeriesIndex
            Reference to parent SeriesIndex for retriever access
        """
        self.instance_uid = instance_uid
        self.dataset = dataset
        self._series_index = series_index

    def get_image(
        self,
        contrast: Optional[Union[str, Contrast]] = None,
        image_width: int = 128,
    ) -> Image.Image:
        """
        Render this instance as an image.

        Parameters
        ----------
        contrast : str, Contrast, or None
            Contrast specification. Can be a string, Contrast object, or None.
            If string, will be converted to Contrast.

        image_width : int, default 128
            Width of output image in pixels. Height is scaled proportionally.

        Returns
        -------
        PIL.Image.Image
            Rendered image

        Raises
        ------
        ValueError
            If image rendering fails
        """
        # Normalize contrast to dict format that the renderer expects
        window_settings = self._normalize_contrast(contrast)

        # Create generator with these settings
        renderer = InstanceRenderer(
            image_width=image_width,
            window_settings=window_settings,
        )

        img = renderer.render_instance(
            self.dataset,
            instance_uid=self.instance_uid,
            retriever=self._series_index._get_or_create_retriever(),
            series_uid=self._series_index.series_uid,
        )

        if img is None:
            raise ValueError(f"Failed to render image for instance {self.instance_uid}")

        return img

    def get_pixel_array(self) -> np.ndarray:
        """
        Get the raw pixel array from this instance.

        Includes DICOM RescaleSlope and RescaleIntercept if present.

        Returns
        -------
        np.ndarray
            Pixel data with rescaling applied

        Raises
        ------
        ValueError
            If pixel array cannot be extracted
        """
        try:
            pixel_array = self.dataset.pixel_array

            # Handle DICOM rescale/slope/intercept
            if hasattr(self.dataset, "RescaleSlope") and hasattr(
                self.dataset, "RescaleIntercept"
            ):
                slope = float(self.dataset.RescaleSlope)
                intercept = float(self.dataset.RescaleIntercept)
                pixel_array = pixel_array * slope + intercept

            return pixel_array

        except Exception as e:
            raise ValueError(f"Failed to extract pixel array: {e}")

    def get_contrast_grid(
        self,
        contrasts: list[str],
        image_width: int = 128,
    ) -> Image.Image:
        """
        Render this instance with multiple contrasts in a grid.

        Parameters
        ----------
        contrasts : list of str
            List of contrast specifications (preset names, 'auto', 'embedded', etc.)

        image_width : int, default 128
            Width of each output image in pixels

        Returns
        -------
        PIL.Image.Image
            Grid of images with different contrasts

        Raises
        ------
        ValueError
            If grid rendering fails
        """
        if not contrasts:
            raise ValueError("Must provide at least one contrast")

        # Render each contrast
        images = []
        for contrast_spec in contrasts:
            try:
                img = self.get_image(contrast=contrast_spec, image_width=image_width)
                images.append(img)
            except Exception as e:
                logger.warning(f"Failed to render contrast {contrast_spec}: {e}")
                continue

        if not images:
            raise ValueError("Failed to render any contrast images")

        # Create grid (1 row, N columns for now)
        # Simple layout: arrange horizontally
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)

        grid = Image.new("L", (total_width, max_height))
        x_offset = 0
        for img in images:
            grid.paste(img, (x_offset, 0))
            x_offset += img.width

        return grid

    def _normalize_contrast(self, contrast: Optional[Union[str, dict, Contrast]]) -> Union[str, dict, None]:
        """
        Normalize contrast input to format expected by MosaicGenerator.

        Parameters
        ----------
        contrast : str, Contrast, dict, or None
            Contrast specification. Can be:
            - None: use default
            - str: preset name, 'auto', 'embedded'
            - Contrast: contrast object
            - dict: with 'window_width' and 'window_center' keys

        Returns
        -------
        Union[str, dict, None]
            Normalized contrast (string like 'auto', or dict with window settings)
        """
        if contrast is None:
            return None

        if isinstance(contrast, str):
            return contrast

        if isinstance(contrast, dict):
            # Dict with window_width and window_center
            if 'window_width' in contrast and 'window_center' in contrast:
                return {
                    'window_width': contrast['window_width'],
                    'window_center': contrast['window_center'],
                }
            return None

        if isinstance(contrast, Contrast):
            # If spec is provided, use that (handles presets, auto, embedded, WW/WL)
            if contrast.spec is not None:
                return contrast.spec

            # If window_width/window_center are provided, return as dict
            if contrast.window_width is not None and contrast.window_center is not None:
                return {
                    "window_width": contrast.window_width,
                    "window_center": contrast.window_center,
                }

            # Default to None
            return None

        raise TypeError(f"Invalid contrast type: {type(contrast)}")


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
        use_cache: bool = True,
        force_rebuild: bool = False,
    ):
        """
        Initialize a series index.

        Parameters
        ----------
        series : str
            Series UID or full path.
            Examples:
            - "38902e14-b11f-4548-910e-771ee757dc82"
            - "s3://idc-open-data/38902e14-b11f-4548-910e-771ee757dc82"
            - "/local/dicom/series-uid"

        root : str, default "s3://idc-open-data"
            Root storage path. Overridden if series contains full path.

        cache_dir : str or None, default None
            Cache directory for indices. If None, uses platform default.
            Ignored if use_cache=False.

        use_cache : bool, default True
            If True, loads/generates and caches DICOM series index.
            If False, fetches instances on-demand without caching.
            Caching enables O(1) position lookup and faster repeated access.
            Disable caching for one-off queries or to avoid index generation overhead.

        Raises
        ------
        ValueError
            If series cannot be resolved or index generation fails.
        """
        self._logger = logging.getLogger(__name__)
        self._logger.debug(f"Initializing SeriesIndex for {series}")

        # Parse and normalize series specification
        result = parse_and_normalize_series(series, root, self._logger)
        if result is None:
            raise ValueError(f"Could not resolve series: {series}")

        self._root_path, self._series_uid = result
        self._logger.debug(f"Resolved to UID: {self._series_uid}, Root: {self._root_path}")

        self._use_cache = use_cache
        self._cache_dir = cache_dir
        self._retriever = None  # Lazy-initialized retriever

        # Load or generate index (either from cache or fresh)
        # If use_cache=False, still generates but doesn't save to disk
        index_df = load_or_generate_index(
            series_uid=self._series_uid,
            root_path=self._root_path,
            index_dir=cache_dir,
            logger_instance=self._logger,
            save_to_cache=use_cache,
            force_rebuild=force_rebuild,
        )

        if index_df is None:
            raise ValueError(
                f"Failed to generate index for series {self._series_uid}"
            )

        self._index_df = index_df

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
    def index_dataframe(self) -> pl.DataFrame:
        """Return the underlying index DataFrame."""
        return self._index_df

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

    def _get_or_create_retriever(self) -> DICOMRetriever:
        """
        Lazily create and cache the DICOMRetriever.

        Returns
        -------
        DICOMRetriever
            Initialized retriever for this series
        """
        if self._retriever is None:
            self._retriever = DICOMRetriever(self._root_path, index_df=self._index_df)
        return self._retriever

    def get_instances(
        self,
        positions: Optional[list[float]] = None,
        slice_numbers: Optional[list[int]] = None,
        max_workers: Optional[int] = None,
        remove_duplicates: bool = False,
        headers_only: bool = False,
    ) -> list[Instance]:
        """
        Fetch multiple DICOM instances in parallel.

        Core method for instance retrieval. Handles parallel fetching,
        deduplication, and optional headers-only mode.

        Parameters
        ----------
        positions : list of float, optional
            Normalized positions (0.0-1.0) along primary axis.
            Mutually exclusive with slice_numbers.

        slice_numbers : list of int, optional
            Zero-indexed slice numbers in the series.
            Mutually exclusive with positions.

        max_workers : int, default 8
            Maximum number of parallel fetch threads

        remove_duplicates : bool, default False
            If True, removes duplicate positions/slice_numbers, keeping only
            the first occurrence. Useful when you don't know the series size.

        headers_only : bool, default False
            If True, fetches only DICOM headers (fast, lightweight metadata).
            Instance.get_image() will still work, fetching full data when needed.

        Returns
        -------
        list of Instance
            Instance objects in original order, with datasets cached in memory.

        Raises
        ------
        ValueError
            If both/neither positions and slice_numbers specified, invalid values,
            or if instance retrieval fails.
        """
        if (positions is None and slice_numbers is None) or (
            positions is not None and slice_numbers is not None
        ):
            raise ValueError("Specify either positions or slice_numbers, not both")

        retriever = self._get_or_create_retriever()

        # Handle positions case
        if positions is not None:
            if not positions:
                raise ValueError("positions list cannot be empty")

            # Validate and deduplicate
            if remove_duplicates:
                seen = set()
                unique_positions = []
                for pos in positions:
                    if pos not in seen:
                        unique_positions.append(pos)
                        seen.add(pos)
                if len(unique_positions) < len(positions):
                    logger.debug(
                        f"Removed {len(positions) - len(unique_positions)} duplicate positions"
                    )
                positions = unique_positions
            else:
                unique_positions = positions

            # Validate all positions
            for pos in positions:
                if not (0.0 <= pos <= 1.0):
                    raise ValueError(f"Position must be between 0.0 and 1.0, got {pos}")

            # Map positions to instance UIDs, look up filenames, and build URLs
            urls_with_pos = []  # List of (url, pos, instance_uid, dataset) tuples
            for pos in positions:
                result = retriever.get_instance_at_position(self._series_uid, pos)
                if result is None:
                    logger.warning(f"Failed to locate instance at position {pos}")
                    continue
                instance_uid, dataset = result

                # Look up filename in index
                matching = self._index_df.filter(pl.col("SOPInstanceUID") == instance_uid)
                if matching.height == 0:
                    logger.warning(f"Instance {instance_uid} not found in index")
                    continue
                data_url = matching[0, "DataURL"]

                urls_with_pos.append((data_url, pos, instance_uid, dataset))

            if not urls_with_pos:
                raise ValueError(f"Failed to locate any instances at {len(positions)} positions")

            # Fetch remaining datasets in parallel when get_instance_at_position returned None headers (cache miss)
            urls_needing_fetch = []
            for (url, _, instance_uid, dataset) in urls_with_pos:
                if dataset is None or headers_only:
                    urls_needing_fetch.append((url, instance_uid))

            fetched_map = {}
            if urls_needing_fetch:
                fetch_urls = [url for url, _ in urls_needing_fetch]
                datasets = retriever.get_instances(
                    fetch_urls,
                    headers_only=headers_only,
                    max_workers=max_workers,
                )
                for (url, instance_uid), dataset in zip(urls_needing_fetch, datasets):
                    fetched_map[(url, instance_uid)] = dataset

            # Build result mapping
            results = {}
            for (url, pos, instance_uid, dataset) in urls_with_pos:
                resolved = dataset
                if resolved is None or headers_only:
                    resolved = fetched_map.get((url, instance_uid))
                if resolved is not None:
                    results[pos] = Instance(instance_uid, resolved, self)

            # Return in position order, filtering out failed fetches
            instances = [results[pos] for pos in unique_positions if pos in results]

            if not instances:
                raise ValueError(f"Failed to retrieve any instances from {len(positions)} positions")

            return instances

        else:  # slice_numbers is not None
            if not slice_numbers:
                raise ValueError("slice_numbers list cannot be empty")

            # Validate and deduplicate
            if remove_duplicates:
                seen = set()
                unique_slices = []
                for slice_num in slice_numbers:
                    if slice_num not in seen:
                        unique_slices.append(slice_num)
                        seen.add(slice_num)
                if len(unique_slices) < len(slice_numbers):
                    logger.debug(
                        f"Removed {len(slice_numbers) - len(unique_slices)} duplicate slice numbers"
                    )
                slice_numbers = unique_slices
            else:
                unique_slices = slice_numbers

            # Validate all slice numbers
            for slice_num in slice_numbers:
                if not (0 <= slice_num < self.instance_count):
                    raise ValueError(
                        f"Slice number {slice_num} out of bounds "
                        f"(series has {self.instance_count} instances)"
                    )

            # Get metadata for all slices from index and build URLs
            sorted_df = self._index_df.sort("Index")
            urls_with_slice = []  # List of (url, slice_num, instance_uid) tuples
            for slice_num in slice_numbers:
                try:
                    row = sorted_df.row(slice_num, named=True)
                except IndexError as exc:
                    raise ValueError(
                        f"Slice number {slice_num} out of bounds "
                        f"(series has {self.instance_count} instances)"
                    ) from exc

                data_url = row["DataURL"]
                instance_uid = row["SOPInstanceUID"]
                urls_with_slice.append((data_url, slice_num, instance_uid))

            # Fetch all in parallel via retriever
            urls = [url for url, _, _ in urls_with_slice]
            datasets = retriever.get_instances(urls, headers_only=headers_only, max_workers=max_workers)

            # Build result mapping
            results = {}
            for (url, slice_num, instance_uid), dataset in zip(urls_with_slice, datasets):
                if dataset is not None:
                    results[slice_num] = Instance(instance_uid, dataset, self)

            # Return in slice order, filtering out failed fetches
            instances = [results[s] for s in unique_slices if s in results]

            if not instances:
                raise ValueError(f"Failed to retrieve any instances from {len(slice_numbers)} slices")

            return instances

    def get_instance(
        self,
        position: Optional[float] = None,
        slice_number: Optional[int] = None,
        slice_offset: int = 0,
    ) -> Instance:
        """
        Get a single DICOM instance at the specified position or slice number.

        Convenience wrapper around get_instances() for single instance retrieval.
        Supports optional slice offset for relative navigation.

        Parameters
        ----------
        position : float, optional
            Normalized position (0.0-1.0) along the primary axis.
            Mutually exclusive with slice_number.

        slice_number : int, optional
            Zero-indexed slice number in the series.
            Mutually exclusive with position.

        slice_offset : int, optional
            Offset from the initial position/slice_number by number of slices.
            Applied after position/slice_number selection. Default: 0.
            Examples:
            - 1: next slice
            - -1: previous slice
            - 0: no offset (default)

        Returns
        -------
        Instance
            The DICOM instance with cached dataset

        Raises
        ------
        ValueError
            If both or neither position and slice_number are specified,
            if slice_offset is out of bounds,
            or if the instance cannot be retrieved.
        """
        if (position is None and slice_number is None) or (
            position is not None and slice_number is not None
        ):
            raise ValueError("Specify either position or slice_number, not both")

        # Convert position to slice_number using PositionInterpolator
        if position is not None:
            interp = PositionInterpolator(self.instance_count)
            target_slice = interp.position_to_index(position, slice_offset=slice_offset)
            instances = self.get_instances(slice_numbers=[target_slice])
        else:
            # Handle slice_number with offset
            target_slice = slice_number + slice_offset
            if target_slice < 0 or target_slice >= self.instance_count:
                raise ValueError(
                    f"Slice offset {slice_offset} out of bounds: "
                    f"slice {slice_number} + offset {slice_offset} = {target_slice}, "
                    f"but series has {self.instance_count} instances (0-{self.instance_count - 1})"
                )
            instances = self.get_instances(slice_numbers=[target_slice])

        return instances[0]

    def get_image(
        self,
        position: Optional[float] = None,
        slice_number: Optional[int] = None,
        contrast: Optional[Union[str, Contrast]] = None,
        image_width: int = 128,
    ) -> Image.Image:
        """
        Render an image from this series.

        Convenience method that fetches an instance and renders it.
        Equivalent to: instance = index.get_instance(...); instance.get_image(...)

        Parameters
        ----------
        position : float, optional
            Normalized position (0.0-1.0). Mutually exclusive with slice_number.

        slice_number : int, optional
            Zero-indexed slice number. Mutually exclusive with position.

        contrast : str, Contrast, or None
            Contrast specification (preset, 'auto', 'embedded', or WW/WL)

        image_width : int, default 128
            Width of output image in pixels

        Returns
        -------
        PIL.Image.Image
            Rendered image

        Raises
        ------
        ValueError
            If position and slice_number are both/neither specified, or if
            rendering fails.
        """
        instance = self.get_instance(position=position, slice_number=slice_number)
        from .image_utils import InstanceRenderer

        renderer = InstanceRenderer(image_width=image_width, window_settings=contrast)
        img = renderer.render_instance(instance.dataset)
        if img is None:
            raise ValueError("Failed to render image for selected instance")
        return img

    def get_images(
        self,
        positions: Optional[list[float]] = None,
        slice_numbers: Optional[list[int]] = None,
        contrast: Optional[Union[str, Contrast]] = None,
        image_width: int = 128,
        max_workers: Optional[int] = None,
        remove_duplicates: bool = False,
    ) -> list[Image.Image]:
        """
        Render multiple images at specified positions or slice numbers.

        Convenience wrapper: fetches instances in parallel, then renders them sequentially.
        Equivalent to: instances = index.get_instances(...); [inst.get_image(...) for inst in instances]

        Parameters
        ----------
        positions : list of float, optional
            Normalized positions (0.0-1.0) for images to render.
            Mutually exclusive with slice_numbers.

        slice_numbers : list of int, optional
            Zero-indexed slice numbers for images to render.
            Mutually exclusive with positions.

        contrast : str, Contrast, or None
            Contrast specification (preset, 'auto', 'embedded', or WW/WL)

        image_width : int, default 128
            Width of each output image in pixels

        max_workers : int, default 8
            Maximum number of parallel fetch threads (rendering is sequential)

        remove_duplicates : bool, default False
            If True, removes duplicate positions/slice_numbers, keeping only
            the first occurrence. Useful when you don't know the series size
            and want to avoid rendering the same image multiple times.

        Returns
        -------
        list of PIL.Image.Image
            Rendered images in the order specified

        Raises
        ------
        ValueError
            If both/neither positions and slice_numbers specified,
            or if any position/slice is invalid.
        """
        # Fetch instances in parallel (handles validation and deduplication)
        instances = self.get_instances(
            positions=positions,
            slice_numbers=slice_numbers,
            max_workers=max_workers,
            remove_duplicates=remove_duplicates,
            headers_only=False,  # Need full data for rendering
        )

        from .image_utils import InstanceRenderer

        renderer = InstanceRenderer(image_width=image_width, window_settings=contrast)
        images = []
        for instance in instances:
            img = renderer.render_instance(instance.dataset)
            if img is None:
                raise ValueError(
                    f"Failed to render image for {instance.instance_uid}"
                )
            images.append(img)
        return images

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SeriesIndex("
            f"uid='{self._series_uid}', "
            f"instances={self.instance_count}, "
            f"axis='{self.primary_axis}'"
            f")"
        )
