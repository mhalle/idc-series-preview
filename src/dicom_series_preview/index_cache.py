"""
DICOM series index caching and management.

Provides tools to load, validate, and use cached DICOM header indices
for avoiding redundant header retrieval from storage.

Index directory resolution (in order of precedence):
1. --index-directory CLI argument
2. DICOM_SERIES_PREVIEW_INDEX_DIR environment variable
3. Default: {platformdirs.user_cache_dir("dicom-series-preview")}/indices/

Index generation uses progressive range requests to fetch headers efficiently,
then extracts necessary fields to build a Polars DataFrame.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional, Any

import polars as pl
import pydicom
from platformdirs import user_cache_dir

from .retriever import DICOMRetriever
from .slice_sorting import sort_slices

logger = logging.getLogger(__name__)


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


def dicom_header_to_dict(dataset: pydicom.Dataset) -> dict[str, Any]:
    """
    Convert pydicom Dataset to a dictionary with image metadata.

    Extracts key fields needed for index generation:
    - ImagePositionPatient
    - InstanceNumber
    - SliceLocation
    - And other DICOM elements

    Args:
        dataset: pydicom.Dataset object

    Returns:
        Dictionary with key DICOM fields
    """
    result = {}

    # Extract specific fields we need for sorting
    if hasattr(dataset, "ImagePositionPatient"):
        try:
            result["ImagePositionPatient"] = list(dataset.ImagePositionPatient)
        except Exception:
            pass

    if hasattr(dataset, "InstanceNumber"):
        try:
            result["InstanceNumber"] = int(dataset.InstanceNumber)
        except Exception:
            pass

    if hasattr(dataset, "SliceLocation"):
        try:
            result["SliceLocation"] = float(dataset.SliceLocation)
        except Exception:
            pass

    # Extract all DICOM elements for varying field detection
    for elem in dataset:
        tag_key = f"{elem.tag.group:04X}{elem.tag.elem:04X}"
        tag_name = elem.keyword if elem.keyword else elem.name

        try:
            value = elem.value

            # Handle different value types
            if isinstance(value, pydicom.Sequence):
                # Skip sequences for now
                continue
            elif isinstance(value, pydicom.Dataset):
                # Skip nested datasets
                continue
            elif isinstance(value, bytes):
                # Binary data: store as dict with size info
                value = {
                    "_type": "binary",
                    "size": len(value),
                    "hex_preview": value[:32].hex(),
                }
            elif hasattr(value, '__iter__') and not isinstance(value, str):
                # Array-like values
                try:
                    value = list(value)
                except Exception:
                    value = str(value)

            result[tag_key] = {
                "name": tag_name,
                "vr": elem.VR,
                "value": value,
            }

        except Exception:
            # Skip elements that can't be converted
            pass

    return result


# Required columns that must exist in a valid index
REQUIRED_COLUMNS = {"Index", "PrimaryPosition", "PrimaryAxis", "SOPInstanceUID"}


def get_cache_directory(cli_arg: Optional[str] = None) -> Path:
    """
    Resolve cache directory with fallback chain.

    Resolution order:
    1. CLI argument (--cache-dir)
    2. Environment variable (DICOM_SERIES_PREVIEW_CACHE_DIR)
    3. Default: platformdirs.user_cache_dir("dicom-series-preview")

    Indices are stored in: {cache_dir}/indices/

    Args:
        cli_arg: Optional directory from --cache-dir CLI argument

    Returns:
        Resolved Path to cache directory (parent, not the indices subdirectory)
    """
    # 1. CLI argument takes precedence
    if cli_arg:
        return Path(cli_arg)

    # 2. Environment variable
    env_var = os.environ.get("DICOM_SERIES_PREVIEW_CACHE_DIR")
    if env_var:
        return Path(env_var)

    # 3. Default platformdirs location
    return Path(user_cache_dir("dicom-series-preview"))


def get_index_path(series_uid: str, cache_dir: Optional[Path] = None) -> Path:
    """
    Get full path to index file for a series.

    Args:
        series_uid: Series UID (normalized format with hyphens)
        cache_dir: Cache directory (None to use defaults)

    Returns:
        Path to index file ({cache_dir}/indices/{series_uid}_index.parquet)
    """
    if cache_dir is None:
        cache_dir = get_cache_directory()

    indices_dir = cache_dir / "indices"
    return indices_dir / f"{series_uid}_index.parquet"


def _load_index(index_path: Path) -> pl.DataFrame:
    """
    Load and validate index file.

    Args:
        index_path: Path to Parquet index file

    Returns:
        Polars DataFrame with validated index data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not valid Parquet or missing required columns
    """
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    try:
        df = pl.read_parquet(str(index_path))
    except Exception as e:
        raise ValueError(f"Failed to read Parquet file {index_path}: {e}")

    return df


def _validate_index(df: pl.DataFrame, expected_series_uid: str) -> bool:
    """
    Validate index DataFrame has required columns and metadata.

    Args:
        df: Polars DataFrame to validate
        expected_series_uid: Expected SeriesUID value

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails
    """
    # Check required columns exist
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Index missing required columns: {missing}. "
            f"Found columns: {df.columns}"
        )

    # Check if SeriesUID is a column
    if "SeriesUID" in df.columns:
        unique_uids = df.select("SeriesUID").unique().to_series().to_list()
        if len(unique_uids) != 1:
            raise ValueError(
                f"Index has multiple SeriesUIDs: {unique_uids}"
            )
        series_uid_from_data = unique_uids[0]
        if series_uid_from_data != expected_series_uid:
            raise ValueError(
                f"Index SeriesUID mismatch: expected {expected_series_uid}, "
                f"found {series_uid_from_data} in data"
            )
    else:
        raise ValueError(
            "Index has no SeriesUID column"
        )

    # Validate Index column values
    if not df["Index"].dtype == pl.UInt32:
        logger.warning(
            f"Index column has type {df['Index'].dtype}, expected UInt32"
        )

    return True


def _generate_parquet_table(
    datasets_by_uid: dict[str, pydicom.Dataset],
    series_uid: str,
    storage_root: str,
) -> pl.DataFrame:
    """
    Generate a Polars DataFrame from DICOM datasets for parquet export.

    Dynamically determines column types from DICOM VR (Value Representation) codes.
    Handles special fields like ImagePositionPatient by expanding into separate columns.
    Instances are sorted by z-position then instance number (same as main program).

    Args:
        datasets_by_uid: Dict mapping instance UID to pydicom.Dataset
        series_uid: Series UID for metadata
        storage_root: Root path for storage (e.g., s3://bucket/series-uid/)

    Returns:
        Polars DataFrame with strongly-typed columns ready for parquet export
    """
    if not datasets_by_uid:
        raise ValueError("No datasets provided")

    # Convert datasets to slice dict format for centralized sorting
    slice_dicts_with_uids = []
    for uid, dataset in datasets_by_uid.items():
        slice_dict = {"_uid": uid}

        # Extract fields needed for sorting
        if hasattr(dataset, "ImagePositionPatient"):
            try:
                slice_dict["ImagePositionPatient"] = list(dataset.ImagePositionPatient)
            except Exception:
                pass

        if hasattr(dataset, "InstanceNumber"):
            try:
                slice_dict["InstanceNumber"] = int(dataset.InstanceNumber)
            except Exception:
                pass

        if hasattr(dataset, "SliceLocation"):
            try:
                slice_dict["SliceLocation"] = float(dataset.SliceLocation)
            except Exception:
                pass

        slice_dicts_with_uids.append(slice_dict)

    # Use centralized sorting logic (detects axis automatically)
    sorted_slices = sort_slices(slice_dicts_with_uids)
    instance_uids = [s["_uid"] for s in sorted_slices]
    uid_to_sorted_slice = {s["_uid"]: s for s in sorted_slices}

    # Collect all tags and identify which vary
    all_tags = set()
    for dataset in datasets_by_uid.values():
        for elem in dataset:
            tag_key = f"{elem.tag.group:04X}{elem.tag.elem:04X}"
            all_tags.add(tag_key)

    # Determine varying fields with their VR codes
    varying_fields = {}  # tag_key -> (keyword, vr, values)

    for tag_key in sorted(all_tags):
        values = []
        keyword = None
        vr = None

        for uid in instance_uids:
            dataset = datasets_by_uid[uid]
            try:
                # Find element by tag key
                for elem in dataset:
                    if f"{elem.tag.group:04X}{elem.tag.elem:04X}" == tag_key:
                        keyword = keyword or (elem.keyword if elem.keyword else elem.name)
                        vr = vr or elem.VR
                        values.append(elem.value)
                        break
                else:
                    values.append(None)
            except Exception:
                values.append(None)

        # Skip internal/unknown tags
        if not keyword or keyword.startswith("_") or keyword == "?":
            continue

        # Check if values vary
        values_str = [str(v) for v in values]
        if len(set(values_str)) > 1:  # Has variation
            varying_fields[tag_key] = (keyword, vr, values)

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
    for tag_key, (keyword, vr, values) in varying_fields.items():
        # Skip ImagePositionPatient (already stored as PrimaryPosition/PrimaryAxis)
        if keyword == "ImagePositionPatient":
            continue

        # Special handling for ImageOrientationPatient (expand to 6 columns)
        elif keyword == "ImageOrientationPatient":
            col_vals = [[] for _ in range(6)]
            for val in values:
                if isinstance(val, (list, tuple)) and len(val) >= 6:
                    for i in range(6):
                        try:
                            col_vals[i].append(float(val[i]))
                        except (ValueError, TypeError):
                            col_vals[i].append(None)
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
                elif isinstance(val, bytes):
                    sizes.append(len(val))
                    hash_val = hashlib.sha256(val[:32]).hexdigest()
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
            logger.debug(f"Skipping sequence field: {keyword}")
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
        pl.lit(series_uid).alias("SeriesUID"),
        pl.lit(storage_root).alias("StorageRoot"),
    )

    return df


def load_or_generate_index(
    series_uid: str,
    root_path: str,
    index_dir: Optional[str] = None,
    logger_instance: Optional[logging.Logger] = None,
    save_to_cache: bool = True,
) -> Optional[pl.DataFrame]:
    """
    Load existing index or generate new one.

    Args:
        series_uid: Normalized series UID
        root_path: Root storage path
        index_dir: Optional cache directory (None to use defaults)
        logger_instance: Logger instance (uses module logger if None)
        save_to_cache: If True, save generated index to disk cache.
                      If False, keep in memory only (useful for one-off queries).
                      Cache loading is always attempted if save_to_cache=True.

    Returns:
        Polars DataFrame with index, or None on error
    """
    log = logger_instance or logger

    # Resolve cache directory
    cache_dir = get_cache_directory(index_dir)
    index_path = get_index_path(series_uid, cache_dir)

    # Try to load existing index (only if caching is enabled)
    if save_to_cache and index_path.exists():
        log.debug(f"Loading cached index from: {index_path}")
        try:
            df = _load_index(index_path)
            _validate_index(df, series_uid)
            log.info(f"Index loaded: {len(df)} instances")
            return df
        except (FileNotFoundError, ValueError) as e:
            log.error(f"Failed to load index: {e}")
            return None

    # Index doesn't exist or caching disabled, generate it
    log.info(f"Generating index for series {series_uid}...")

    try:
        # Create retriever and fetch instance UIDs
        retriever = DICOMRetriever(root_path)
        instance_uids = retriever.list_instances(series_uid)
        if not instance_uids:
            log.error(f"No instances found in series {series_uid}")
            return None

        log.info(f"Found {len(instance_uids)} instances")

        # Fetch headers in parallel using progressive range requests
        log.debug("Fetching DICOM headers...")
        urls = [f"{series_uid}/{uid}" for uid in instance_uids]
        datasets_list = retriever.get_instances(urls, headers_only=True)

        # Build dict mapping uid to dataset
        datasets_by_uid = {}
        for uid, dataset in zip(instance_uids, datasets_list):
            if dataset is not None:
                datasets_by_uid[uid] = dataset

        if not datasets_by_uid:
            log.error(f"Failed to fetch any headers from {len(instance_uids)} instances")
            return None

        log.info(f"Successfully fetched {len(datasets_by_uid)} instance headers")

        # Generate parquet table from datasets
        log.debug("Generating index parquet table...")
        storage_root = f"{root_path}/{series_uid}/"
        df = _generate_parquet_table(datasets_by_uid, series_uid, storage_root)

        # Save to cache only if enabled
        if save_to_cache:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(str(index_path))
            log.info(
                f"Index generated and saved to {index_path}: "
                f"{len(df)} rows, {len(df.columns)} columns"
            )
        else:
            log.info(
                f"Index generated (not saved): "
                f"{len(df)} rows, {len(df.columns)} columns"
            )

        return df

    except Exception as e:
        log.error(f"Failed to generate index: {e}")
        return None
