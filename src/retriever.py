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
        self, series_uid: str, count: int
    ) -> List[Tuple[str, pydicom.Dataset]]:
        """
        Get a distributed subset of DICOM instances from a series.

        Selects instances evenly distributed across the entire series range
        to represent the full set of images.

        Args:
            series_uid: The DICOM series UID
            count: Number of instances to retrieve

        Returns:
            List of tuples (instance_uid, pydicom.Dataset)
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

        if len(all_headers) <= count:
            # Return all instances if we have fewer than requested
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Series has {len(all_headers)} instances, less than requested {count}")
            selected = all_headers
        else:
            # Sort all instances by z-position
            def get_z_position(item):
                instance_uid, ds = item
                if hasattr(ds, 'ImagePositionPatient') and len(ds.ImagePositionPatient) >= 3:
                    return float(ds.ImagePositionPatient[2])
                elif hasattr(ds, 'InstanceNumber'):
                    return float(ds.InstanceNumber)
                elif hasattr(ds, 'SliceLocation'):
                    return float(ds.SliceLocation)
                return 0

            sorted_headers = sorted(all_headers, key=get_z_position)

            # Select evenly distributed instances from sorted list (includes first and last)
            indices = [int(i * (len(sorted_headers) - 1) / (count - 1)) for i in range(count)]
            selected = [sorted_headers[i] for i in indices]

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
