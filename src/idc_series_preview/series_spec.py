"""Utilities for parsing and normalizing DICOM series specifications."""

import logging
from pathlib import Path
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

    DICOM numeric UIDs (e.g., 1.2.840.10008.1.2.1) are passed through unchanged.

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

    # If this looks like a numeric DICOM UID (digits and dots), return as-is
    if '.' in series_uid and all(part.isdigit() for part in series_uid.split('.') if part):
        return series_uid

    # Remove any existing hyphens for full UUID
    cleaned = series_uid.replace('-', '').lower()

    # UUID format: 8-4-4-4-12 characters
    if len(cleaned) != 32:
        raise ValueError(f"Series UID must be 32 hex characters (got {len(cleaned)}): {series_uid}")

    # Re-insert hyphens at correct positions
    formatted = f"{cleaned[0:8]}-{cleaned[8:12]}-{cleaned[12:16]}-{cleaned[16:20]}-{cleaned[20:32]}"

    return formatted


def parse_and_normalize_series(
    series_spec: str,
    root: str,
    logger: logging.Logger,
    cache_dir: Optional[str | Path] = None,
) -> Optional[tuple[str, str]]:
    """
    Parse, normalize, and resolve series specification.

    Handles full paths and series UIDs. When given just a series UID, attempts to
    resolve the IDC storage URL via the published parquet index.

    Args:
        series_spec: Series specification (UID or full path)
        root: Default root path
        logger: Logger instance

    Returns:
        Tuple of (series_uid, series_url) on success where series_url always
        has a trailing slash.
        None on error (error already logged)
    """
    from .retriever import DICOMRetriever

    # Parse series specification (can be UID or full path)
    try:
        root_path, parsed_spec = parse_series_specification(series_spec, root)
    except ValueError as e:
        logger.error(f"Invalid series specification: {e}")
        return None

    # Helper to normalize a series storage URL with trailing slash
    def _normalize_series_url(url: str) -> str:
        cleaned = url.rstrip("/*")
        return cleaned if cleaned.endswith("/") else cleaned + "/"

    # If a full path was provided, honor it directly (used for local/HTTP overrides)
    if series_spec.startswith(("s3://", "http://", "https://", "file://")):
        try:
            series_uid = normalize_series_uid(parsed_spec)
        except ValueError as e:
            logger.error(f"Invalid series UID: {e}")
            return None

        return series_uid, _normalize_series_url(series_spec)

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

    series_url: Optional[str] = None

    # Prefer cached index (if present) to avoid remote lookups on repeat calls
    try:
        if cache_dir is None:
            from .index_cache import get_cache_directory

            resolved_cache_dir = get_cache_directory()
        else:
            resolved_cache_dir = Path(cache_dir)

        from .index_cache import get_index_path

        index_path = get_index_path(series_uid, resolved_cache_dir)
        if index_path.exists():
            import polars as pl

            try:
                df = pl.read_parquet(str(index_path), columns=["_series_url"])
                cached_url = df["_series_url"][0] if len(df) else None
                if cached_url:
                    series_url = _normalize_series_url(str(cached_url))
                    logger.debug(f"Using cached series URL from index: {series_url}")
                    return series_uid, series_url
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"Failed to read cached index at {index_path}: {e}")
    except Exception as e:  # pragma: no cover - defensive
        logger.debug(f"Cache lookup failed for {series_uid}: {e}")
    # Only UUID-like SeriesInstanceUIDs should consult idc-index; dotted UIDs
    # (numeric DICOM) should fall back to the provided root.
    cleaned = series_uid.replace("-", "")
    is_uuid_like = len(cleaned) == 32 and all(c in "0123456789abcdef" for c in cleaned.lower())
    if is_uuid_like:
        try:
            from .idc_index_client import get_series_url

            series_url = get_series_url(series_uid)
            if series_url:
                logger.info(f"Resolved series via IDC index: {series_url}")
        except Exception as e:
            logger.debug(f"IDC index lookup failed for {series_uid}: {e}")

    # Fallback to provided root (primarily for local/override scenarios)
    if not series_url:
        normalized_root = root.rstrip("/")
        series_url = f"{normalized_root}/{series_uid}"
        logger.debug(f"Falling back to root-derived series URL: {series_url}")

    return series_uid, _normalize_series_url(series_url)
