# Build-Index Implementation Plan

## Overview

The `build-index` subcommand and `--index` flag provide a caching mechanism to accelerate repeated operations on the same DICOM series by avoiding redundant header retrieval from storage (S3, HTTP, local).

## Build-Index Subcommand

### Syntax
```
idc-series-preview build-index SERIESUID [INDEXNAME]
```

### Arguments

#### SERIESUID (required)
The DICOM Series UID (same format options as other commands)
- Full hex UID: `38902e14b11f4548910e771ee757dc82`
- UUID format: `38902e14-b11f-4548-910e-771ee757dc82`

#### INDEXNAME (optional)
Output Parquet index file path.
- If omitted: defaults to `{SERIESUID}.index.parquet` in the current directory
- Example default: `38902e14-b11f-4548-910e-771ee757dc82.index.parquet`
- Can be any `.parquet` file path (relative or absolute)

### Options

`--root PATH`
: Root path for DICOM files (same as other commands).
: Default: `s3://idc-open-data`

`-v, --verbose`
: Enable verbose logging.

### Behavior

1. Retrieves all DICOM headers from the series (via HeaderCapture)
2. Generates Parquet table with sorting metadata (Index, PrimaryPosition, PrimaryAxis)
3. Writes to the specified INDEXNAME location
4. Reports success with file size and row count

### Example Commands

```bash
# Build index with default naming in current directory
idc-series-preview build-index 38902e14-b11f-4548-910e-771ee757dc82

# Build index with custom filename
idc-series-preview build-index 38902e14-b11f-4548-910e-771ee757dc82 my_series.index.parquet

# Build index in specific directory
idc-series-preview build-index 38902e14-b11f-4548-910e-771ee757dc82 /data/indices/series.index.parquet

# From custom root
idc-series-preview build-index 38902e14-b11f-4548-910e-771ee757dc82 \
  --root /local/dicom/path
```

## --Index Flag Integration

All other subcommands (mosaic, contrast-mosaic, etc.) will support a new optional `--index` flag:

### Syntax
```
idc-series-preview <command> SERIESUID OUTPUT --index [INDEXFILE] [other options]
```

### Behavior

#### No --index flag (default auto-detection)
1. Look for `{SERIESUID}.index.parquet` in the current directory
2. If found: Load and use it for sorting metadata
3. If not found: Retrieve headers directly from storage (no caching)

#### --index with filename
```
--index /path/to/custom.index.parquet
```
1. Load the specified index file
2. If file doesn't exist: Error (user explicitly requested it)

#### --index with no argument (detect default)
```
--index
```
Same as no flag (looks for `{SERIESUID}.index.parquet`)

#### --noindex flag
```
--noindex
```
Explicitly bypass index usage:
- Don't look for cached index even if it exists
- Always retrieve headers from storage
- Useful for forcing fresh data or bypassing potentially stale cache

### Index File Validation

When loading an index file:
1. Verify it's a valid Parquet file
2. Verify it contains required columns: Index, PrimaryPosition, PrimaryAxis, SOPInstanceUID
3. Verify SeriesUID in metadata matches the requested SERIESUID
4. Log warning if index is older than N days (TODO: define threshold)

### Examples

```bash
# Auto-detect: looks for 38902e14-b11f-4548-910e-771ee757dc82.index.parquet
idc-series-preview mosaic 38902e14-b11f-4548-910e-771ee757dc82 output.webp

# Explicitly provide index location
idc-series-preview mosaic 38902e14-b11f-4548-910e-771ee757dc82 output.webp \
  --index /data/indices/my_series.index.parquet

# Force fresh data (ignore cache)
idc-series-preview mosaic 38902e14-b11f-4548-910e-771ee757dc82 output.webp \
  --noindex

# Auto-detect with other options
idc-series-preview mosaic 38902e14-b11f-4548-910e-771ee757dc82 output.webp \
  --contrast-preset lung --image-width 128
```

## Implementation Details

### File Organization

```
Current Directory/
├── 38902e14-b11f-4548-910e-771ee757dc82.index.parquet  ← Auto-detected
├── custom_indices/
│   ├── series1.index.parquet
│   └── series2.index.parquet
```

### New Module/Functions

Create `src/idc_series_preview/index_cache.py`:

```python
class IndexCache:
    """
    Manages DICOM series index caching and loading.
    """

    @staticmethod
    def default_index_path(series_uid: str) -> Path:
        """Return default index filename for a series UID."""
        return Path.cwd() / f"{series_uid}.index.parquet"

    @staticmethod
    def find_index(series_uid: str, index_path: Path | None) -> Path | None:
        """
        Find index file given explicit path or series UID.
        Returns Path if found, None if not found (and not required).
        """

    @staticmethod
    def load_index(index_path: Path) -> pl.DataFrame:
        """
        Load and validate index file.
        Raises: FileNotFoundError, ValueError on invalid format
        """

    @staticmethod
    def validate_index(df: pl.DataFrame, expected_series_uid: str) -> bool:
        """
        Validate index DataFrame has required columns and metadata.
        Raises: ValueError on validation failure
        """

    @staticmethod
    def get_sorting_info(df: pl.DataFrame, instance_uid: str) -> dict:
        """
        Extract PrimaryPosition/PrimaryAxis/Index for an instance.
        Used by other commands to skip header retrieval.
        """
```

### Integration Points

#### In retriever.py or new module:
```python
def get_instance_sorting_info(series_uid, instance_uid, index_cache=None):
    """
    Get sorting metadata for an instance.

    Priority:
    1. Load from index_cache if provided
    2. Retrieve from DICOM headers if index_cache is None
    """
```

#### In __main__.py:
- Add `--index` and `--noindex` arguments to argument parsers
- Parse index flag and pass IndexCache to commands
- Commands load index and use it to skip header retrieval if available

### Performance Impact

**With Index (cached)**
- No header retrieval from storage → ~seconds saved depending on series size
- Parquet read from local disk → ~ms overhead
- Ideal for repeated operations on same series

**Without Index (first run)**
- Full header retrieval from storage → baseline performance
- Can auto-create index for future use

**Without Index (--noindex)**
- Forces header retrieval even if index exists
- Useful for fresh data or troubleshooting

## Backward Compatibility

All changes are additive:
- `--index` flag is optional
- Default behavior (auto-detect) is safe and transparent
- Existing scripts continue to work unchanged
- No breaking changes to command syntax

## Testing Strategy

1. **build-index command**
   - Build index with default naming
   - Build index with custom path
   - Verify Parquet structure and contents
   - Test with various series sizes

2. **Auto-detection**
   - Run command without --index when .index.parquet exists
   - Verify index is used (check logs, verify sorting)
   - Delete index and verify falls back to header retrieval

3. **Explicit --index**
   - Provide valid index → works
   - Provide invalid/nonexistent index → clear error
   - Provide invalid Parquet → validation error

4. **--noindex flag**
   - Verify header retrieval happens even with index present
   - Verify performance is normal (no index overhead)

5. **Compatibility**
   - All existing commands work unchanged
   - All existing command-line arguments still work
   - New --index doesn't interfere with other flags

## Future Enhancements

- Index staleness warning (if > N days old)
- Index versioning (if Parquet schema changes)
- Index cleanup utility (remove old indices)
- Batch index building (multiple series at once)
- Index compression/optimization
