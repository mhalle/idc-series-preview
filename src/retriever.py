"""DICOM instance retrieval from various storage backends."""

import logging
from typing import List, Tuple, Optional
import asyncio

import obstore as obs
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
        self._store = self._init_store(root_path)

    @staticmethod
    def _init_store(root_path: str):
        """Initialize the appropriate object store based on path."""
        if root_path.startswith("s3://"):
            # S3 storage
            return obs.parse_url(root_path)
        elif root_path.startswith("http://") or root_path.startswith("https://"):
            # HTTP storage
            return obs.parse_url(root_path)
        else:
            # Local filesystem
            return obs.parse_url(f"file://{root_path}")

    def _get_instance_path(self, series_uid: str, instance_uid: str) -> str:
        """Get the full path to a DICOM instance."""
        return f"{series_uid}/{instance_uid}.dcm"

    async def _get_instance_headers(
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
                data = obs.get_range(self._store[0], path, 0, max_bytes)
            except Exception as e:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Initial range request failed for {path}, trying larger range: {e}")
                # Fall back to larger range
                data = obs.get_range(self._store[0], path, 0, 20000)

            # Parse DICOM headers
            try:
                ds = pydicom.dcmread(BytesIO(data), stop_before_pixels=True)
                # Get file size from metadata if available
                meta_data = obs.head(self._store[0], path)
                size = meta_data.size
                return ds, size
            except Exception as e:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Failed to parse DICOM headers for {path}: {e}")
                return None, 0

        except Exception as e:
            logger.error(f"Error retrieving instance headers {series_uid}/{instance_uid}: {e}")
            return None, 0

    def _get_instance_path_sync(
        self, series_uid: str, instance_uid: str
    ) -> Tuple[Optional[pydicom.Dataset], int]:
        """Synchronous wrapper for getting instance headers."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self._get_instance_headers(series_uid, instance_uid)
        )

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
            results = obs.list(self._store[0], series_uid)
            instances = []

            for obj in results:
                if obj.name.endswith('.dcm'):
                    instance_uid = obj.name.replace('.dcm', '')
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
            ds, size = self._get_instance_path_sync(series_uid, instance_uid)
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
            data = obs.get(self._store[0], path)
            ds = pydicom.dcmread(BytesIO(data))
            return ds

        except Exception as e:
            logger.error(f"Error retrieving instance {series_uid}/{instance_uid}: {e}")
            return None
