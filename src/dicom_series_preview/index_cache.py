"""
DICOM series index caching and management.

Provides tools to load, validate, and use cached DICOM header indices
for avoiding redundant header retrieval from storage.

Index directory resolution (in order of precedence):
1. --index-directory CLI argument
2. DICOM_SERIES_PREVIEW_INDEX_DIR environment variable
3. Default: {platformdirs.user_cache_dir("dicom-series-preview")}/indices/
"""

import logging
import os
from pathlib import Path
from typing import Optional

import polars as pl
from platformdirs import user_cache_dir

logger = logging.getLogger(__name__)


class IndexCache:
    """
    Manages DICOM series index caching and loading.

    Provides methods to:
    - Resolve cache directory from CLI, env var, or defaults
    - Locate index files
    - Load and validate Parquet indices
    - Extract sorting metadata for instances
    - Generate indices on demand
    """

    # Required columns that must exist in a valid index
    REQUIRED_COLUMNS = {"Index", "PrimaryPosition", "PrimaryAxis", "SOPInstanceUID"}

    @staticmethod
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

    @staticmethod
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
            cache_dir = IndexCache.get_cache_directory()

        indices_dir = cache_dir / "indices"
        return indices_dir / f"{series_uid}_index.parquet"

    @staticmethod
    def load_index(index_path: Path) -> pl.DataFrame:
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

    @staticmethod
    def load_or_generate_index(
        series_uid: str,
        root_path: str,
        index_dir: Optional[str] = None,
        verbose: bool = False,
        logger_instance: Optional[logging.Logger] = None,
    ) -> Optional[pl.DataFrame]:
        """
        Load existing index or generate new one.

        Args:
            series_uid: Normalized series UID
            root_path: Root storage path
            index_dir: Optional cache directory (None to use defaults)
            verbose: Enable verbose logging
            logger_instance: Logger instance (uses module logger if None)

        Returns:
            Polars DataFrame with index, or None on error
        """
        log = logger_instance or logger

        # Resolve cache directory
        cache_dir = IndexCache.get_cache_directory(index_dir)
        index_path = IndexCache.get_index_path(series_uid, cache_dir)

        # Try to load existing index
        if index_path.exists():
            if verbose:
                log.info(f"Loading cached index from: {index_path}")
            try:
                df = IndexCache.load_index(index_path)
                IndexCache.validate_index(df, series_uid)
                if verbose:
                    log.info(f"Index loaded: {len(df)} instances")
                return df
            except (FileNotFoundError, ValueError) as e:
                log.error(f"Failed to load index: {e}")
                return None

        # Index doesn't exist, generate it
        if verbose:
            log.info(f"Generating index for series {series_uid}...")

        try:
            from .header_capture import HeaderCapture

            capture = HeaderCapture(root_path)
            headers_data = capture.capture_series_headers(series_uid)
            storage_root = f"{root_path}/{series_uid}/"

            if verbose:
                log.info("Generating index parquet table...")
            df = capture.generate_parquet_table(headers_data, storage_root)

            # Save the generated index (create indices subdirectory)
            index_path.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(str(index_path))

            if verbose:
                log.info(
                    f"Index generated and saved to {index_path}: "
                    f"{len(df)} rows, {len(df.columns)} columns"
                )

            return df

        except Exception as e:
            log.error(f"Failed to generate index: {e}")
            return None

    @staticmethod
    def validate_index(df: pl.DataFrame, expected_series_uid: str) -> bool:
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
        missing = IndexCache.REQUIRED_COLUMNS - set(df.columns)
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

    @staticmethod
    def get_sorting_info(
        df: pl.DataFrame, instance_uid: str
    ) -> Optional[dict]:
        """
        Extract PrimaryPosition/PrimaryAxis/Index for an instance.

        Used by other commands to skip header retrieval when index is available.

        Args:
            df: Polars DataFrame with index data
            instance_uid: SOPInstanceUID to look up

        Returns:
            Dict with 'index', 'primary_position', 'primary_axis', or None if not found
        """
        matches = df.filter(pl.col("SOPInstanceUID") == instance_uid)

        if matches.height == 0:
            return None

        if matches.height > 1:
            logger.warning(
                f"Multiple rows found for instance {instance_uid}, using first"
            )

        row = matches.row(0, named=True)
        return {
            "index": row["Index"],
            "primary_position": float(row["PrimaryPosition"]),
            "primary_axis": row["PrimaryAxis"],
        }

    @staticmethod
    def get_all_sorting_info(df: pl.DataFrame) -> dict[str, dict]:
        """
        Extract sorting info for all instances in the index.

        Args:
            df: Polars DataFrame with index data

        Returns:
            Dict mapping SOPInstanceUID to sorting info dicts
        """
        result = {}
        for row in df.iter_rows(named=True):
            instance_uid = row["SOPInstanceUID"]
            result[instance_uid] = {
                "index": row["Index"],
                "primary_position": float(row["PrimaryPosition"]),
                "primary_axis": row["PrimaryAxis"],
            }
        return result
