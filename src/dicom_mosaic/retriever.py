"""DICOM instance retrieval from various storage backends."""

import logging
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from obstore.store import from_url
import pydicom
from io import BytesIO

logger = logging.getLogger(__name__)


def _configure_pixel_data_handlers():
    """Configure pydicom to use gdcm for pixel data if available.

    Note: gdcm support is disabled until we can test it (not available on macOS ARM64).
    """
    # TODO: Enable gdcm support when available on test platform
    # try:
    #     # Try to import gdcm to check if it's available
    #     import gdcm  # noqa: F401
    #     # If gdcm is available, set it as the preferred handler
    #     # This ensures JPEG Extended and other advanced codecs can be decoded
    #     pydicom.config.settings.pixel_data_handlers = ['gdcm', 'pillow']
    #     logger.debug("Configured pydicom to use gdcm for pixel data decoding")
    # except ImportError:
    #     # gdcm not available, use default handlers (pillow only)
    #     logger.debug("gdcm not available, using default pixel data handlers")
    pass


def _get_sort_key(item: Tuple[str, pydicom.Dataset]) -> Tuple[float, float]:
    """
    Create a sort key for DICOM instances that handles both spatial and temporal ordering.

    For temporal sequences where multiple instances share the same or similar z-position,
    this function creates a two-level sort key:
    1. Primary: z-position (spatial location)
    2. Secondary: instance number (temporal ordering at same location)

    Args:
        item: Tuple of (instance_uid, pydicom.Dataset)

    Returns:
        Tuple of (z_position, instance_number) for sorting
    """
    instance_uid, ds = item

    # Get z-position (spatial ordering)
    if hasattr(ds, 'ImagePositionPatient') and len(ds.ImagePositionPatient) >= 3:
        z_position = float(ds.ImagePositionPatient[2])
    elif hasattr(ds, 'SliceLocation'):
        z_position = float(ds.SliceLocation)
    else:
        z_position = 0.0

    # Get instance number (temporal ordering within same spatial location)
    if hasattr(ds, 'InstanceNumber'):
        instance_number = float(ds.InstanceNumber)
    else:
        instance_number = 0.0

    return (z_position, instance_number)


class DICOMRetriever:
    """Retrieve DICOM instances from S3, HTTP, or local storage."""

    _handlers_configured = False

    def __init__(self, root_path: str):
        """
        Initialize the retriever.

        Args:
            root_path: Root path for DICOM files (S3, HTTP, or local path)
        """
        # Configure pixel data handlers once at first initialization
        if not DICOMRetriever._handlers_configured:
            _configure_pixel_data_handlers()
            DICOMRetriever._handlers_configured = True

        self.root_path = root_path
        self.store = self._init_store(root_path)

    @staticmethod
    def _init_store(root_path: str):
        """Initialize the appropriate object store based on path."""
        if not root_path.startswith(("s3://", "http://", "https://", "file://")):
            # Local filesystem - add file:// prefix
            root_path = f"file://{root_path}"

        # For S3, use anonymous access (skip signature)
        if root_path.startswith("s3://"):
            from obstore.store import S3Store
            return S3Store.from_url(
                root_path,
                config={"aws_skip_signature": "true"}
            )

        return from_url(root_path)

    def find_series_by_prefix(self, prefix: str) -> List[str]:
        """
        Find series UIDs matching a prefix.

        Args:
            prefix: Series UID prefix (without hyphens, e.g., "38902e14")

        Returns:
            List of matching series UIDs with hyphens
        """
        matches = []

        try:
            # Normalize prefix to hex without hyphens
            prefix_clean = prefix.replace('-', '').lower()

            found_series = set()

            # Build hyphenated version to use as S3 list prefix
            # UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
            if len(prefix_clean) <= 8:
                s3_prefix = prefix_clean
            elif len(prefix_clean) <= 12:
                s3_prefix = f"{prefix_clean[0:8]}-{prefix_clean[8:]}"
            elif len(prefix_clean) <= 16:
                s3_prefix = f"{prefix_clean[0:8]}-{prefix_clean[8:12]}-{prefix_clean[12:]}"
            elif len(prefix_clean) <= 20:
                s3_prefix = f"{prefix_clean[0:8]}-{prefix_clean[8:12]}-{prefix_clean[12:16]}-{prefix_clean[16:]}"
            else:
                s3_prefix = f"{prefix_clean[0:8]}-{prefix_clean[8:12]}-{prefix_clean[12:16]}-{prefix_clean[16:20]}-{prefix_clean[20:]}"

            # Try listing with the hyphenated prefix
            try:
                results = self.store.list(prefix=s3_prefix)

                for batch in results:
                    if isinstance(batch, list):
                        for obj in batch:
                            path = obj.get('path') if isinstance(obj, dict) else str(obj)
                            # Extract series UID (first part of path before /)
                            series_uid = path.split('/')[0]

                            # Validate UUID format and that it matches prefix
                            if (len(series_uid) == 36 and
                                series_uid.count('-') == 4 and
                                series_uid.replace('-', '').lower().startswith(prefix_clean)):
                                found_series.add(series_uid)

            except Exception as e:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Error listing with prefix {s3_prefix}: {e}")

            matches = sorted(found_series)

        except Exception as e:
            logger.error(f"Error searching for series prefix {prefix}: {e}")

        return matches

    def _get_instance_path(self, series_uid: str, instance_uid: str) -> str:
        """Get the full path to a DICOM instance."""
        return f"{series_uid}/{instance_uid}.dcm"

    def _get_instance_headers(
        self, series_uid: str, instance_uid: str, max_bytes: int = 5000
    ) -> Tuple[Optional[pydicom.Dataset], int]:
        """
        Retrieve DICOM headers via range request.

        Args:
            series_uid: The DICOM series UID
            instance_uid: The DICOM instance UID
            max_bytes: Maximum bytes to retrieve (default 5KB for minimal header data)

        Returns:
            Tuple of (pydicom.Dataset or None, total_file_size)
        """
        try:
            path = self._get_instance_path(series_uid, instance_uid)

            # Try initial range request (5KB for minimal header data)
            try:
                range_result = self.store.get_range(path, start=0, length=max_bytes)
                # get_range returns obstore.Bytes directly
                data = bytes(range_result) if hasattr(range_result, '__len__') else bytes(range_result.bytes())
            except Exception as e:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Range request failed for {path}, falling back to full file: {e}")
                # Fall back to full file
                result = self.store.get(path)
                data = bytes(result.bytes())

            # Parse DICOM headers
            try:
                ds = pydicom.dcmread(BytesIO(data), stop_before_pixels=True, force=True)
                # Get file size from metadata if available
                meta_data = self.store.head(path)
                size = meta_data.get('size') if isinstance(meta_data, dict) else meta_data.size
                return ds, size
            except NotImplementedError as e:
                # Unsupported compression, skip this file
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Cannot read DICOM with unsupported compression {path}: {e}")
                return None, 0
            except Exception as e:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Failed to parse DICOM headers for {path}: {e}")
                # Try retrieving full file as fallback
                try:
                    result = self.store.get(path)
                    # GetResult.bytes() returns obstore.Bytes which can be converted to bytes
                    full_data = bytes(result.bytes())
                    ds = pydicom.dcmread(BytesIO(full_data), stop_before_pixels=True, force=True)
                    return ds, len(full_data)
                except NotImplementedError as e2:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Cannot read DICOM with unsupported compression {path}: {e2}")
                    return None, 0
                except Exception as e2:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Failed to parse full DICOM file for {path}: {e2}")
                    return None, 0

        except Exception as e:
            logger.error(f"Error retrieving instance headers {series_uid}/{instance_uid}: {e}")
            return None, 0

    def list_instances(self, series_uid: str) -> List[str]:
        """
        List all DICOM instances in a series.

        Args:
            series_uid: The DICOM series UID

        Returns:
            List of instance UIDs
        """
        try:
            # List objects in the series directory
            # Use prefix parameter for cleaner API
            prefix = f"{series_uid}/" if not series_uid.endswith("/") else series_uid
            results = self.store.list(prefix=prefix)
            instances = []

            # obstore.list returns an iterator that yields batches
            for batch in results:
                # Each batch is a list of objects
                if isinstance(batch, list):
                    for obj in batch:
                        # Each object is a dict with 'path' key
                        path = obj.get('path') if isinstance(obj, dict) else str(obj)
                        if path.endswith('.dcm'):
                            # Extract just the filename without the series UID prefix
                            instance_uid = path.split('/')[-1].replace('.dcm', '')
                            instances.append(instance_uid)
                else:
                    # Single object (shouldn't happen with list)
                    path = batch.get('path') if isinstance(batch, dict) else str(batch)
                    if path.endswith('.dcm'):
                        instance_uid = path.split('/')[-1].replace('.dcm', '')
                        instances.append(instance_uid)

            instances.sort()
            return instances

        except Exception as e:
            logger.error(f"Error listing instances for series {series_uid}: {e}")
            return []

    def get_instances_distributed(
        self, series_uid: str, count: int, start: float = 0.0, end: float = 1.0
    ) -> List[Tuple[str, pydicom.Dataset]]:
        """
        Get a distributed subset of DICOM instances from a series.

        Selects instances evenly distributed across a specified z-position range
        to represent the full set of images in that range.

        Args:
            series_uid: The DICOM series UID
            count: Number of instances to retrieve
            start: Start of normalized z-position range (0.0-1.0). Default: 0.0
            end: End of normalized z-position range (0.0-1.0). Default: 1.0

        Returns:
            List of tuples (instance_uid, pydicom.Dataset).
            If fewer instances are found in the range than requested, returns all
            instances in the range (no duplicates).
        """
        all_instances = self.list_instances(series_uid)

        if not all_instances:
            logger.error(f"No instances found for series {series_uid}")
            return []

        # First pass: Get headers for all instances in parallel
        all_headers = []
        headers_bytes = 0

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(self._get_instance_headers, series_uid, uid): uid
                for uid in all_instances
            }

            for future in as_completed(futures):
                instance_uid = futures[future]
                try:
                    ds, size = future.result()
                    if ds is not None:
                        all_headers.append((instance_uid, ds))
                        headers_bytes += min(5000, size)
                except Exception as e:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Failed to retrieve header for {instance_uid}: {e}")

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Retrieved headers for {len(all_headers)} instances ({headers_bytes / 1024 / 1024:.2f}MB)")

        # Sort all instances by z-position, then by instance number for temporal sequences
        sorted_headers = sorted(all_headers, key=_get_sort_key)

        # Apply range filtering if not using full range
        if start > 0.0 or end < 1.0:
            if len(sorted_headers) > 0:
                # Extract z-positions for range calculation
                z_positions = [_get_sort_key(item)[0] for item in sorted_headers]
                min_z = min(z_positions)
                max_z = max(z_positions)
                z_range = max_z - min_z

                # Map normalized range [0, 1] to actual z-values
                start_z = min_z + (z_range * start)
                end_z = min_z + (z_range * end)

                # Filter instances within the range (inclusive on both ends)
                filtered_headers = [
                    item for item in sorted_headers
                    if start_z <= _get_sort_key(item)[0] <= end_z
                ]

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Range filter: {start:.1%} to {end:.1%} ({start_z:.2f} to {end_z:.2f}) selected {len(filtered_headers)} of {len(sorted_headers)} instances")
            else:
                filtered_headers = sorted_headers
        else:
            filtered_headers = sorted_headers

        # Select distributed instances from filtered list
        if len(filtered_headers) <= count:
            # Return all instances if we have fewer than requested (no duplicates)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Filtered range has {len(filtered_headers)} instances, less than requested {count}")
            selected = filtered_headers
        else:
            # Select evenly distributed instances (includes first and last to avoid fencepost errors)
            indices = [int(i * (len(filtered_headers) - 1) / (count - 1)) for i in range(count)]
            selected = [filtered_headers[i] for i in indices]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Selected {len(selected)} instances from {len(all_headers)} total")

        # Second pass: Get full pixel data only for selected instances in parallel
        results = []
        pixel_bytes = 0

        # Extract UIDs from selected (which are now tuples of (uid, ds))
        selected_uids = [uid for uid, ds in selected]

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(self.get_instance_data, series_uid, uid): uid
                for uid in selected_uids
            }

            for future in as_completed(futures):
                instance_uid = futures[future]
                try:
                    ds = future.result()
                    if ds is not None:
                        results.append((instance_uid, ds))
                        # Count actual file size
                        try:
                            path = self._get_instance_path(series_uid, instance_uid)
                            meta = self.store.head(path)
                            file_size = meta.get('size') if isinstance(meta, dict) else meta.size
                            pixel_bytes += file_size
                        except:
                            pass
                except Exception as e:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Failed to retrieve pixel data for {instance_uid}: {e}")

        if logger.isEnabledFor(logging.DEBUG):
            total_bytes = headers_bytes + pixel_bytes
            logger.debug(f"Successfully retrieved {len(results)} instances with pixel data ({pixel_bytes / 1024 / 1024:.2f}MB)")
            logger.debug(f"Total S3 data transferred: {total_bytes / 1024 / 1024:.2f}MB ({headers_bytes / 1024 / 1024:.2f}MB headers + {pixel_bytes / 1024 / 1024:.2f}MB pixel data)")
        return results

    def get_instance_at_position(
        self, series_uid: str, position: float, slice_offset: int = 0
    ) -> Optional[Tuple[str, pydicom.Dataset]]:
        """
        Get a single DICOM instance at a specific position using priority-based selection.

        Selection strategy (in priority order):
        1. If z-position varies: Map position to z-position range (select by spatial location)
        2. If temporal data exists: Map position to temporal range (select by time)
        3. Otherwise: Map position to index in sorted sequence (select by order)

        Then applies slice_offset to move up/down in the sequence.

        Args:
            series_uid: The DICOM series UID
            position: Normalized position (0.0-1.0)
            slice_offset: Number of slices to offset from selected position
                          (e.g., 1 for next slice, -1 for previous)

        Returns:
            Tuple of (instance_uid, pydicom.Dataset) or None if retrieval failed.
        """
        all_instances = self.list_instances(series_uid)

        if not all_instances:
            logger.error(f"No instances found for series {series_uid}")
            return None

        # Get headers for all instances
        all_headers = []

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(self._get_instance_headers, series_uid, uid): uid
                for uid in all_instances
            }

            for future in as_completed(futures):
                instance_uid = futures[future]
                try:
                    ds, size = future.result()
                    if ds is not None:
                        all_headers.append((instance_uid, ds))
                except Exception as e:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Failed to retrieve header for {instance_uid}: {e}")

        if not all_headers:
            logger.error(f"No instances could be retrieved for series {series_uid}")
            return None

        # Sort by z-position, then by instance number for temporal sequences
        sorted_headers = sorted(all_headers, key=_get_sort_key)

        # Extract z-positions to check for spatial variation
        z_positions = [_get_sort_key(item)[0] for item in sorted_headers]
        has_z_variation = len(set(z_positions)) > 1  # Check if z-positions vary

        selected_item = None
        selection_method = None

        # Strategy 1: If z-position varies, select by spatial location
        if has_z_variation:
            min_z = min(z_positions)
            max_z = max(z_positions)
            z_range = max_z - min_z
            target_z = min_z + (z_range * position)

            # Find closest instance by z-position
            closest_idx = min(
                range(len(sorted_headers)),
                key=lambda i: abs(_get_sort_key(sorted_headers[i])[0] - target_z)
            )
            selected_item = sorted_headers[closest_idx]
            selection_method = f"spatial (z-position {target_z:.2f}, closest at {z_positions[closest_idx]:.2f})"

        # Strategy 2: Check for temporal data (multiple instances at same z-position)
        if selected_item is None:
            z_to_instances = {}
            for item in sorted_headers:
                z = _get_sort_key(item)[0]
                if z not in z_to_instances:
                    z_to_instances[z] = []
                z_to_instances[z].append(item)

            # Check if any z-position has multiple instances (potential temporal sequence)
            has_temporal = any(len(instances) > 1 for instances in z_to_instances.values())

            if has_temporal:
                # Collect all instances with time information
                instances_with_time = []
                for item in sorted_headers:
                    instance_uid, ds = item
                    # Try to extract time from various DICOM tags
                    time_value = None
                    if hasattr(ds, 'InstanceCreationTime'):
                        time_value = ds.InstanceCreationTime
                    elif hasattr(ds, 'ContentTime'):
                        time_value = ds.ContentTime
                    elif hasattr(ds, 'AcquisitionTime'):
                        time_value = ds.AcquisitionTime

                    # Use instance number as fallback temporal ordering
                    temporal_order = float(ds.InstanceNumber) if hasattr(ds, 'InstanceNumber') else 0.0
                    instances_with_time.append((item, temporal_order, time_value))

                # Check if we actually have time data (not just instance numbers)
                has_time_data = any(t[2] is not None for t in instances_with_time)

                if has_time_data:
                    # Sort by temporal information and select by position in temporal range
                    instances_with_time.sort(key=lambda x: x[1])
                    target_index = int(position * (len(instances_with_time) - 1))
                    selected_item = instances_with_time[target_index][0]
                    selection_method = f"temporal (by time/instance number)"
                else:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("Series has multiple instances at same z-position but no temporal data found")

        # Strategy 3: Select by sequence index (fallback)
        if selected_item is None:
            target_index = int(position * (len(sorted_headers) - 1))
            selected_item = sorted_headers[target_index]
            selection_method = f"sequence index ({target_index} of {len(sorted_headers)})"

        # Apply slice offset if specified
        if slice_offset != 0:
            # Find current selected item index in sorted_headers
            current_index = sorted_headers.index(selected_item)
            target_index = current_index + slice_offset

            # Validate that offset stays within bounds
            if target_index < 0:
                logger.error(f"Slice offset {slice_offset} goes before first instance (would be index {target_index}, but valid range is 0-{len(sorted_headers)-1})")
                return None
            elif target_index >= len(sorted_headers):
                logger.error(f"Slice offset {slice_offset} goes past last instance (would be index {target_index}, but valid range is 0-{len(sorted_headers)-1})")
                return None

            selected_item = sorted_headers[target_index]
            selection_method = f"{selection_method} + offset {slice_offset} → index {target_index}"

        instance_uid, ds = selected_item
        sort_key = _get_sort_key(selected_item)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Position {position:.1%} selected by {selection_method} (z={sort_key[0]:.2f}, instance_number={sort_key[1]:.0f})")

        # Get full pixel data for this instance
        full_ds = self.get_instance_data(series_uid, instance_uid)
        if full_ds is None:
            logger.error(f"Could not retrieve pixel data for instance {instance_uid}")
            return None

        return (instance_uid, full_ds)

    def get_instance_data(
        self, series_uid: str, instance_uid: str
    ) -> Optional[pydicom.Dataset]:
        """
        Retrieve complete DICOM instance.

        Args:
            series_uid: The DICOM series UID
            instance_uid: The DICOM instance UID

        Returns:
            pydicom.Dataset or None if retrieval failed
        """
        try:
            path = self._get_instance_path(series_uid, instance_uid)
            result = self.store.get(path)
            # GetResult.bytes() returns obstore.Bytes which can be converted to bytes
            data = bytes(result.bytes())
            ds = pydicom.dcmread(BytesIO(data), force=True)
            return ds

        except NotImplementedError as e:
            # Unsupported compression (e.g., JPEG Extended with 12-bit precision)
            # Skip this instance, mosaic will be generated from decodable instances
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Skipping {instance_uid}: unsupported compression - {e}")
            return None
        except Exception as e:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Error retrieving instance {series_uid}/{instance_uid}: {e}")
            return None
