"""Utilities for parsing and normalizing DICOM series specifications."""

import logging
from typing import Optional


def parse_series_specification(
    series_spec: str, default_root: str
) -> tuple[str, str]:
    """
    Parse series specification, which can be either a series UID or a full path.

    Handles multiple formats:
    - Series UID only: "38902e14-b11f-4548-910e-771ee757dc82"
    - Full path: "s3://idc-open-data/38902e14-b11f-4548-910e-771ee757dc82"
    - Full path with slash: "s3://idc-open-data/38902e14-b11f-4548-910e-771ee757dc82/"
    - Local path: "file:///path/to/series/38902e14-b11f-4548-910e-771ee757dc82"
    - HTTP URL: "http://example.com/dicom/38902e14-b11f-4548-910e-771ee757dc82"

    Args:
        series_spec: Series specification (UID or full path)
        default_root: Default root path to use if only UID is provided

    Returns:
        Tuple of (root_path, series_uid)

    Raises:
        ValueError: If the specification format is invalid
    """
    # Check if this is a full path (starts with a storage scheme)
    if any(series_spec.startswith(scheme) for scheme in ("s3://", "http://", "https://", "file://")):
        # This is a full path - extract root and series UID
        # Remove trailing wildcards and slashes
        clean_spec = series_spec.rstrip("/*")

        # Find the last slash to separate root from series UID
        last_slash = clean_spec.rfind("/")
        if last_slash == -1:
            raise ValueError(f"Invalid full path format: {series_spec}")

        root = clean_spec[:last_slash]
        series_uid = clean_spec[last_slash + 1 :]

        if not series_uid:
            raise ValueError(f"No series UID found in path: {series_spec}")

        return root, series_uid
    else:
        # This is a series UID or prefix - use default root
        return default_root, series_spec


def normalize_series_uid(series_uid: str) -> str:
    """
    Normalize a series UID by adding hyphens if not present.

    Converts formats like:
    - 38902e14b11f4548910e771ee757dc82
    - 38902e14-b11f-4548-910e-771ee757dc82

    To standard UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

    Args:
        series_uid: Series UID with or without hyphens

    Returns:
        Normalized series UID with hyphens
    """
    # Check if this is a prefix search (contains wildcard)
    if '*' in series_uid:
        # This is a prefix search - clean it and return for searching
        prefix = series_uid.replace('*', '').replace('-', '').lower()
        if not prefix:
            raise ValueError("Prefix cannot be empty")
        return f"{prefix}*"  # Return as prefix pattern

    # Remove any existing hyphens for full UUID
    cleaned = series_uid.replace('-', '').lower()

    # UUID format: 8-4-4-4-12 characters
    if len(cleaned) != 32:
        raise ValueError(f"Series UID must be 32 hex characters (got {len(cleaned)}): {series_uid}")

    # Re-insert hyphens at correct positions
    formatted = f"{cleaned[0:8]}-{cleaned[8:12]}-{cleaned[12:16]}-{cleaned[16:20]}-{cleaned[20:32]}"

    return formatted


def parse_and_normalize_series(series_spec: str, root: str, logger: logging.Logger) -> Optional[tuple[str, str]]:
    """
    Parse, normalize, and resolve series specification.

    Handles full paths and series UIDs.

    Args:
        series_spec: Series specification (UID or full path)
        root: Default root path
        logger: Logger instance

    Returns:
        Tuple of (root_path, series_uid) on success
        None on error (error already logged)
    """
    from .retriever import DICOMRetriever

    # Parse series specification (can be UID or full path)
    try:
        root_path, parsed_spec = parse_series_specification(series_spec, root)

        # If a full path was provided and --root was also specified, note the override
        if root_path != root:
            logger.debug(f"Full path specified, overriding --root with: {root_path}")
    except ValueError as e:
        logger.error(f"Invalid series specification: {e}")
        return None

    # Normalize series UID (add hyphens if not present, or prepare for prefix search)
    try:
        series_uid = normalize_series_uid(parsed_spec)
    except ValueError as e:
        logger.error(f"Invalid series UID: {e}")
        return None

    # Handle prefix search (ends with *)
    if series_uid.endswith('*'):
        logger.info(f"Searching for series matching prefix: {parsed_spec}...")

        retriever_temp = DICOMRetriever(root_path)
        prefix = series_uid.rstrip('*')
        matches = retriever_temp.find_series_by_prefix(prefix)

        if not matches:
            logger.error(f"No series found matching prefix: {parsed_spec}")
            return None
        elif len(matches) > 1:
            logger.error(f"Prefix '{parsed_spec}' matches {len(matches)} series:")
            for match in matches[:10]:  # Show first 10
                logger.error(f"  - {match}")
            if len(matches) > 10:
                logger.error(f"  ... and {len(matches) - 10} more")
            logger.error("Please provide a more specific prefix")
            return None
        else:
            series_uid = matches[0]
            logger.info(f"Found matching series: {series_uid}")

    return root_path, series_uid
