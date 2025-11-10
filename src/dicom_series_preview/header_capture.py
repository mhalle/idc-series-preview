"""
Experimental header capture functionality for DICOM series analysis.

This module provides utilities to extract and capture DICOM headers from entire series
for analysis purposes. Headers are returned as raw JSON for flexibility in analysis.

Note: This is experimental functionality and subject to change/removal.
"""

import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pydicom

from .retriever import DICOMRetriever


class HeaderCapture:
    """Capture DICOM headers from a series for analysis."""

    def __init__(self, root_path: str, max_workers: int = 8):
        """
        Initialize HeaderCapture.

        Args:
            root_path: Root path for DICOM files (S3, HTTP, or local)
            max_workers: Number of parallel workers for header retrieval
        """
        self.retriever = DICOMRetriever(root_path)
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)

    def _get_instance_header(self, series_uid: str, instance_uid: str) -> dict[str, Any] | None:
        """
        Retrieve and parse headers from a single DICOM instance.

        Args:
            series_uid: Series UID
            instance_uid: Instance UID

        Returns:
            Dictionary of header data, or None if retrieval fails
        """
        try:
            # Get DICOM dataset (get_instance_data already parses and returns a pydicom.Dataset)
            ds = self.retriever.get_instance_data(series_uid, instance_uid)
            if not ds:
                self.logger.warning(f"No data retrieved for instance {instance_uid}")
                return None

            # Convert to dictionary format
            header_dict = self._dicom_to_dict(ds)
            header_dict["_instance_uid"] = instance_uid

            return header_dict

        except Exception as e:
            self.logger.error(f"Error retrieving headers for instance {instance_uid}: {e}")
            return None

    @staticmethod
    def _dicom_to_dict(ds: pydicom.Dataset) -> dict[str, Any]:
        """
        Convert pydicom Dataset to a JSON-serializable dictionary.

        Args:
            ds: pydicom Dataset

        Returns:
            Dictionary representation of DICOM headers
        """
        result = {}

        for elem in ds:
            tag_key = f"{elem.tag.group:04X}{elem.tag.elem:04X}"
            tag_name = elem.keyword if elem.keyword else elem.name

            try:
                # Try to get the value
                value = elem.value

                # Handle different value types
                if isinstance(value, pydicom.Sequence):
                    # Sequence: convert to list of dicts
                    value = [HeaderCapture._dicom_to_dict(item) for item in value]
                elif isinstance(value, pydicom.Dataset):
                    # Nested dataset
                    value = HeaderCapture._dicom_to_dict(value)
                elif isinstance(value, bytes):
                    # Binary data: represent as hex string with size info
                    value = {
                        "_type": "binary",
                        "size": len(value),
                        "hex_preview": value[:32].hex(),  # First 32 bytes
                    }
                elif hasattr(value, '__iter__') and not isinstance(value, str):
                    # Array-like values
                    try:
                        value = list(value)
                    except Exception:
                        value = str(value)
                else:
                    # Scalar values (int, float, str, etc.)
                    value = value

                result[tag_key] = {
                    "name": tag_name,
                    "vr": elem.VR,
                    "value": value,
                }

            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.debug(f"Could not convert element {tag_key} ({tag_name}): {e}")
                result[tag_key] = {
                    "name": tag_name,
                    "vr": elem.VR,
                    "error": str(e),
                }

        return result

    def capture_series_headers(
        self, series_uid: str, limit: int | None = None
    ) -> dict[str, Any]:
        """
        Capture headers from all instances in a series.

        Args:
            series_uid: Series UID
            limit: Maximum number of instances to process (None for all)

        Returns:
            Dictionary containing series metadata and instance headers
        """
        self.logger.info(f"Listing instances in series {series_uid}...")

        # Get list of instances
        instances = self.retriever.list_instances(series_uid)
        if not instances:
            raise ValueError(f"No instances found in series {series_uid}")

        self.logger.info(f"Found {len(instances)} instances")

        # Apply limit if specified
        if limit:
            instances = instances[:limit]
            self.logger.info(f"Processing first {limit} instances")

        # Capture headers in parallel
        self.logger.info("Retrieving headers...")
        headers = {}
        errors = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._get_instance_header, series_uid, instance_uid): instance_uid
                for instance_uid in instances
            }

            completed = 0
            for future in futures:
                instance_uid = futures[future]
                try:
                    header = future.result()
                    if header:
                        headers[instance_uid] = header
                    else:
                        errors.append(f"No data for {instance_uid}")

                    completed += 1
                    if completed % 10 == 0:
                        self.logger.info(f"  {completed}/{len(instances)} instances...")

                except Exception as e:
                    errors.append(f"Error processing {instance_uid}: {e}")

        self.logger.info(f"Completed: {len(headers)} successful, {len(errors)} errors")

        # Compile result
        result = {
            "series_uid": series_uid,
            "total_instances": len(instances),
            "instances_processed": len(headers),
            "errors_count": len(errors),
            "headers": headers,
        }

        if errors:
            result["errors"] = errors

        return result

    def save_headers_json(self, headers_data: dict[str, Any], output_path: Path) -> None:
        """
        Save headers data to JSON file.

        Args:
            headers_data: Headers data from capture_series_headers()
            output_path: Path to write JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(headers_data, f, indent=2, default=str)

        self.logger.info(f"Headers saved to {output_path}")

    def generate_compact_schema(
        self, headers_data: dict[str, Any], storage_root: str
    ) -> dict[str, Any]:
        """
        Generate a compact schema representation with constant and varying headers.

        This uses a run-length encoding approach where constant headers appear once
        and varying headers are arrays. File paths are referenced via FileName field.

        Instances are sorted using the same method as the main program:
        1. Primary: z-position (ImagePositionPatient[2] or SliceLocation)
        2. Secondary: InstanceNumber (temporal ordering at same location)

        Args:
            headers_data: Headers data from capture_series_headers()
            storage_root: Root path for storage (e.g., s3://bucket/series-uid/)

        Returns:
            Compact schema dictionary
        """
        instances = headers_data.get("headers", {})

        if not instances:
            raise ValueError("No instances in headers data")

        # Sort instances using same method as main program: z-position, then instance number
        def get_sort_key(uid: str) -> tuple[float, float]:
            """Create sort key (z_position, instance_number) for an instance."""
            instance_data = instances[uid]

            # Get z-position (spatial ordering)
            z_position = 0.0
            if "ImagePositionPatient" in instance_data:
                tag_info = instance_data["ImagePositionPatient"]
                if isinstance(tag_info, dict) and "value" in tag_info:
                    val = tag_info["value"]
                    if isinstance(val, list) and len(val) >= 3:
                        z_position = float(val[2])
            elif "SliceLocation" in instance_data:
                tag_info = instance_data["SliceLocation"]
                if isinstance(tag_info, dict) and "value" in tag_info:
                    z_position = float(tag_info["value"])

            # Get instance number (temporal ordering within same spatial location)
            instance_number = 0.0
            if "InstanceNumber" in instance_data:
                tag_info = instance_data["InstanceNumber"]
                if isinstance(tag_info, dict) and "value" in tag_info:
                    instance_number = float(tag_info["value"])

            return (z_position, instance_number)

        # Sort instance UIDs using the sort key
        instance_uids = sorted(instances.keys(), key=get_sort_key)

        # Collect all tags
        all_tags = set()
        for uid in instance_uids:
            all_tags.update(instances[uid].keys())

        # Analyze each tag
        varying_items = {}
        constant_items = {}

        for tag in sorted(all_tags):
            values = []
            tag_name = None

            for uid in instance_uids:
                if tag in instances[uid]:
                    tag_info = instances[uid][tag]
                    if isinstance(tag_info, dict):
                        val = tag_info.get("value")
                        if tag_name is None:
                            tag_name = tag_info.get("name", "?")
                    else:
                        val = tag_info
                    values.append(val)
                else:
                    values.append("[MISSING]")

            # Skip internal/unknown tags
            if not tag_name or tag_name.startswith("_") or tag_name == "?":
                continue

            # Check if all values are the same
            values_str = [str(v) for v in values]
            if len(set(values_str)) == 1:
                constant_items[tag_name] = values[0]
            else:
                varying_items[tag_name] = values

        # Add FileName to varying headers (using sorted order)
        filenames = [f"{uid}.dcm" for uid in instance_uids]
        varying_items["FileName"] = filenames

        # Sort varying items for consistent output
        varying_items = dict(sorted(varying_items.items()))

        # Create schema
        schema = {
            "series_uid": headers_data.get("series_uid"),
            "storage_root": storage_root,
            "instance_count": len(instance_uids),
            "constant_headers": constant_items,
            "varying_headers": varying_items,
        }

        return schema
