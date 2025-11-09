"""DICOM instance retrieval from various storage backends."""

import logging
from typing import List, Tuple, Optional

from obstore.store import from_url
import pydicom
from io import BytesIO


logger = logging.getLogger(__name__)


class DICOMRetriever:
    """Retrieve DICOM instances from S3, HTTP, or local storage."""

    def __init__(self, root_path: str):
        """
        Initialize the retriever.

        Args:
            root_path: Root path for DICOM files (S3, HTTP, or local path)
        """
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
            max_bytes: Maximum bytes to retrieve (default 5KB, falls back to 20KB)

        Returns:
            Tuple of (pydicom.Dataset or None, total_file_size)
        """
        try:
            path = self._get_instance_path(series_uid, instance_uid)

            # Try initial range request
            try:
                range_result = self.store.get_range(path, start=0, length=max_bytes)
                # get_range returns obstore.Bytes directly
                data = bytes(range_result) if hasattr(range_result, '__len__') else bytes(range_result.bytes())
            except Exception as e:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Initial range request failed for {path}, trying larger range: {e}")
                # Fall back to larger range
                range_result = self.store.get_range(path, start=0, length=20000)
                data = bytes(range_result) if hasattr(range_result, '__len__') else bytes(range_result.bytes())

            # Parse DICOM headers
            try:
                ds = pydicom.dcmread(BytesIO(data), stop_before_pixels=True, force=True)
                # Get file size from metadata if available
                meta_data = self.store.head(path)
                size = meta_data.get('size') if isinstance(meta_data, dict) else meta_data.size
                return ds, size
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

        if len(all_instances) <= count:
            # Return all instances if we have fewer than requested
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Series has {len(all_instances)} instances, less than requested {count}")
            selected = all_instances
        else:
            # Select evenly distributed instances
            indices = [int(i * len(all_instances) / count) for i in range(count)]
            selected = [all_instances[i] for i in indices]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Selected {len(selected)} instances from {len(all_instances)} total")

        # Retrieve headers for selected instances
        results = []
        for instance_uid in selected:
            ds, size = self._get_instance_headers(series_uid, instance_uid)
            if ds is not None:
                results.append((instance_uid, ds))
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Failed to retrieve {instance_uid}")

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Successfully retrieved {len(results)} instance headers")
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

        except Exception as e:
            logger.error(f"Error retrieving instance {series_uid}/{instance_uid}: {e}")
            return None
