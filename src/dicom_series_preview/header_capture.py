"""
Experimental header capture functionality for DICOM series analysis.

This module provides utilities to extract and capture DICOM headers from entire series
for analysis purposes. Headers are returned as raw JSON for flexibility in analysis.

Note: This is experimental functionality and subject to change/removal.
"""

import json
import logging
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pydicom
import pydicom.datadict as dd
import polars as pl

from .retriever import DICOMRetriever
from .slice_sorting import sort_slices


# Map DICOM VR (Value Representation) codes to Polars types
VR_TO_POLARS_TYPE = {
    # Text strings
    "AE": pl.Utf8,  # Application Entity
    "AS": pl.Utf8,  # Age String
    "AT": pl.Utf8,  # Attribute Tag
    "CS": pl.Utf8,  # Code String
    "DA": pl.Utf8,  # Date
    "DT": pl.Utf8,  # Date Time
    "LO": pl.Utf8,  # Long String
    "LT": pl.Utf8,  # Long Text
    "PN": pl.Utf8,  # Person Name
    "SH": pl.Utf8,  # Short String
    "ST": pl.Utf8,  # Short Text
    "TM": pl.Utf8,  # Time
    "UC": pl.Utf8,  # Unlimited Characters
    "UI": pl.Utf8,  # Unique Identifier
    "UR": pl.Utf8,  # URI/URL
    # Numeric
    "DS": pl.Float32,  # Decimal String
    "FD": pl.Float64,  # Floating Point Double
    "FL": pl.Float32,  # Floating Point Single
    "IS": pl.Int32,  # Integer String
    "SL": pl.Int32,  # Signed Long
    "SS": pl.Int16,  # Signed Short
    "UL": pl.UInt32,  # Unsigned Long
    "US": pl.UInt32,  # Unsigned Short
    # Binary/Other (handled specially)
    "OB": None,  # Other Byte
    "OD": None,  # Other Double
    "OF": None,  # Other Float
    "OL": None,  # Other Long
    "OW": None,  # Other Word
    "UN": None,  # Unknown
    # Sequences (handled specially)
    "SQ": None,  # Sequence
}


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

    def generate_parquet_table(
        self, headers_data: dict[str, Any], storage_root: str
    ) -> pl.DataFrame:
        """
        Generate a Polars DataFrame from headers data for parquet export.

        Dynamically determines column types from DICOM VR (Value Representation) codes.
        Handles special fields like ImagePositionPatient by expanding into separate columns.

        Instances are sorted by z-position then instance number (same as main program).

        Args:
            headers_data: Headers data from capture_series_headers()
            storage_root: Root path for storage (e.g., s3://bucket/series-uid/)

        Returns:
            Polars DataFrame with strongly-typed columns ready for parquet export
        """
        instances = headers_data.get("headers", {})

        if not instances:
            raise ValueError("No instances in headers data")

        # Convert instances to slice dict format for centralized sorting
        slice_dicts_with_uids = []
        for uid, instance_data in instances.items():
            slice_dict = {"_uid": uid}  # Track UID for later

            # Extract ImagePositionPatient and InstanceNumber
            for tag_hex, tag_info in instance_data.items():
                if not isinstance(tag_info, dict):
                    continue

                tag_name = tag_info.get("name", "")
                val = tag_info.get("value")

                if tag_name == "ImagePositionPatient" and isinstance(val, list) and len(val) >= 3:
                    slice_dict["ImagePositionPatient"] = val
                elif tag_name == "InstanceNumber":
                    try:
                        slice_dict["InstanceNumber"] = int(val)
                    except (ValueError, TypeError):
                        pass

            slice_dicts_with_uids.append(slice_dict)

        # Use centralized sorting logic (detects axis automatically)
        sorted_slices = sort_slices(slice_dicts_with_uids)
        instance_uids = [s["_uid"] for s in sorted_slices]

        # Create mapping from uid to sorted slice info (for PrimaryPosition/Axis extraction)
        uid_to_sorted_slice = {s["_uid"]: s for s in sorted_slices}

        # Collect all tags and identify which vary
        all_tags = set()
        for uid in instance_uids:
            all_tags.update(instances[uid].keys())

        # Determine varying fields with their VR codes
        varying_fields = {}  # tag_hex -> (keyword, vr, values)

        for tag_hex in sorted(all_tags):
            values = []
            keyword = None
            vr = None

            for uid in instance_uids:
                if tag_hex in instances[uid]:
                    tag_info = instances[uid][tag_hex]
                    if isinstance(tag_info, dict):
                        val = tag_info.get("value")
                        if keyword is None:
                            keyword = tag_info.get("name")
                            vr = tag_info.get("vr")
                    else:
                        val = tag_info
                    values.append(val)
                else:
                    values.append(None)

            # Skip internal/unknown tags
            if not keyword or keyword.startswith("_") or keyword == "?":
                continue

            # Check if values vary
            values_str = [str(v) for v in values]
            if len(set(values_str)) > 1:  # Has variation
                varying_fields[tag_hex] = (keyword, vr, values)

        # Build column data with dynamic typing
        column_data = {}
        column_types = {}

        # Always include Index (sort order), FileName, and primary position/axis metadata
        column_data["Index"] = list(range(len(instance_uids)))
        column_types["Index"] = pl.UInt32

        column_data["FileName"] = [f"{uid}.dcm" for uid in instance_uids]
        column_types["FileName"] = pl.Utf8

        # Extract PrimaryPosition and PrimaryAxis from sorting metadata
        primary_positions = []
        primary_axes = []
        for uid in instance_uids:
            sorted_slice = uid_to_sorted_slice[uid]
            axis = sorted_slice.get("axis")
            axis_label = sorted_slice.get("axis_label", "I")
            primary_axes.append(axis_label)

            # Extract actual position value (not negated sort_value)
            if axis_label != "I":  # Spatial position
                if "ImagePositionPatient" in sorted_slice and axis is not None:
                    try:
                        position = float(sorted_slice["ImagePositionPatient"][axis])
                        primary_positions.append(position)
                    except (ValueError, TypeError, IndexError):
                        primary_positions.append(0.0)
                elif "SliceLocation" in sorted_slice:
                    try:
                        position = float(sorted_slice["SliceLocation"])
                        primary_positions.append(position)
                    except (ValueError, TypeError):
                        primary_positions.append(0.0)
                else:
                    primary_positions.append(0.0)
            else:  # Instance number
                primary_positions.append(float(sorted_slice.get("InstanceNumber", 0)))

        column_data["PrimaryPosition"] = primary_positions
        column_types["PrimaryPosition"] = pl.Float32
        column_data["PrimaryAxis"] = primary_axes
        column_types["PrimaryAxis"] = pl.Utf8

        # Helper function to get Polars type from VR code
        def get_polars_type(vr: str) -> pl.DataType:
            """Map DICOM VR code to Polars type."""
            if vr in VR_TO_POLARS_TYPE:
                polars_type = VR_TO_POLARS_TYPE[vr]
                return polars_type if polars_type is not None else pl.Utf8
            return pl.Utf8  # Default to string for unknown VR

        # Process varying fields
        for tag_hex, (keyword, vr, values) in varying_fields.items():
            # Skip ImagePositionPatient (already stored as PrimaryPosition/PrimaryAxis)
            if keyword == "ImagePositionPatient":
                continue

            # Special handling for ImageOrientationPatient (expand to 6 columns)
            elif keyword == "ImageOrientationPatient":
                col_vals = [[] for _ in range(6)]
                for val in values:
                    if isinstance(val, list) and len(val) >= 6:
                        for i in range(6):
                            col_vals[i].append(float(val[i]))
                    else:
                        for i in range(6):
                            col_vals[i].append(None)

                for i in range(6):
                    col_name = f"ImageOrientationPatient_{i}"
                    column_data[col_name] = col_vals[i]
                    column_types[col_name] = pl.Float32

            # Special handling for binary data (store size and hash)
            elif vr in ["OB", "OW", "OD"]:
                sizes, hashes = [], []
                for val in values:
                    if isinstance(val, dict) and "_type" in val:
                        sizes.append(val.get("size", 0))
                        hex_preview = val.get("hex_preview", "")
                        hash_val = (
                            hashlib.sha256(hex_preview.encode()).hexdigest()
                            if hex_preview
                            else ""
                        )
                        hashes.append(hash_val)
                    else:
                        sizes.append(0)
                        hashes.append("")

                column_data[f"{keyword}_Size"] = sizes
                column_data[f"{keyword}_Hash"] = hashes
                column_types[f"{keyword}_Size"] = pl.Int32
                column_types[f"{keyword}_Hash"] = pl.Utf8

            # Skip sequences
            elif vr == "SQ":
                self.logger.debug(f"Skipping sequence field: {keyword}")
                continue

            # Regular scalar/string fields
            else:
                polars_type = get_polars_type(vr)
                typed_values = []

                for val in values:
                    if val is None:
                        typed_values.append(None)
                    elif polars_type in [pl.Int16, pl.Int32, pl.UInt32]:
                        try:
                            typed_values.append(int(val))
                        except (ValueError, TypeError):
                            typed_values.append(None)
                    elif polars_type in [pl.Float32, pl.Float64]:
                        try:
                            typed_values.append(float(val))
                        except (ValueError, TypeError):
                            typed_values.append(None)
                    else:
                        typed_values.append(str(val))

                column_data[keyword] = typed_values
                column_types[keyword] = polars_type

        # Create DataFrame with explicit types
        df_dict = {}
        for col_name, values in column_data.items():
            col_type = column_types.get(col_name, pl.Utf8)
            df_dict[col_name] = pl.Series(col_name, values, dtype=col_type)

        df = pl.DataFrame(df_dict)

        # Add metadata columns
        df = df.with_columns(
            pl.lit(headers_data.get("series_uid")).alias("SeriesUID"),
            pl.lit(storage_root).alias("StorageRoot"),
        )

        return df
