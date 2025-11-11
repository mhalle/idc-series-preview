# dicom-series-preview(1) - DICOM series preview and visualization

## NAME

`dicom-series-preview` - Preview DICOM series with intelligent sampling, contrast adjustment, and flexible output formats

## SYNOPSIS

```
dicom-series-preview COMMAND [OPTIONS] [ARGUMENTS]
```

## DESCRIPTION

`dicom-series-preview` is a command-line tool for previewing DICOM medical imaging series stored on S3, HTTP, or local filesystems. It provides multiple visualization modes:

- **mosaic**: Generate tiled grids of images from a series
- **get-image**: Extract single images at specific positions
- **contrast-mosaic**: Create comparison grids with multiple contrast settings
- **build-index**: Pre-build cached indices for faster access
- **get-index**: Retrieve or create an index and return its path

The tool uses efficient retrieval strategies (headers first, then pixel data for selected instances) and supports advanced features like window/level presets, auto-contrast detection, and flexible series specification formats.

## COMMON OPTIONS

These options apply to most subcommands (where applicable):

### Series Specification

All commands accept flexible series specifications:

**Series UID Format**
- Full UUID: `38902e14-b11f-4548-910e-771ee757dc82`
- UUID without hyphens: `38902e14b11f4548910e771ee757dc82`
- Prefix search: `38902e14*` or `389*` (matches first occurrence)
- Full path: `s3://idc-open-data/38902e14-b11f-4548-910e-771ee757dc82`
- Full path: `http://example.com/dicom/38902e14-b11f-4548-910e-771ee757dc82`
- Local path: `file:///data/dicom/38902e14-b11f-4548-910e-771ee757dc82`

When a prefix or full path is used, it's automatically resolved to the complete series UID.

### Storage Options

`--root PATH`
: Root path for DICOM series. Can be S3, HTTP(S), or local filesystem path.
: Default: `s3://idc-open-data`
: Examples:
  - `s3://idc-open-data` (public IDC bucket)
  - `s3://my-private-bucket/dicom`
  - `http://hospital.example.com/dicom-server`
  - `/data/dicom` or `file:///data/dicom` (local)

### Image Output Options

`--image-width PIXELS`
: Width of each image or tile in pixels. Height is scaled proportionally to maintain aspect ratio.
: Default: `128`
: Larger values produce higher-quality output but larger files

### Contrast Options

`--contrast PRESET`
: Contrast/window-level settings. Repeatable for contrast-mosaic command.
: Can be one of:
  - **Presets**: `ct-lung`, `ct-bone`, `ct-brain`, `ct-abdomen`, `ct-liver`, `ct-mediastinum`, `ct-soft-tissue`
  - **Legacy shortcuts**: `lung`, `bone`, `brain`, `abdomen`, `liver`, `mediastinum`, `soft` (for soft-tissue)
  - **Special modes**: `auto` (auto-detect from statistics), `embedded` (use DICOM file's settings)
  - **Custom values**: `WINDOW/CENTER` or `WINDOW,CENTER` format (e.g., `1500/500` for lung, `1500,-500` for negative center)

### Cache Options

`--cache-dir PATH`
: Directory to store/load DICOM series index files (Parquet format).
: Index files are stored as `{CACHE_DIR}/indices/{SERIESUID}_index.parquet`
: Automatically loads from cache on subsequent runs (major performance improvement)
: Overrides platform-specific default cache location

`--no-cache`
: Disable index caching. Fetches DICOM headers fresh from storage every run.
: By default caching is enabled using `DICOM_SERIES_PREVIEW_CACHE_DIR` environment variable or platform cache directory

### Quality Options

`-q, --quality LEVEL`
: Output image compression quality (0-100).
: Default: `25`
: Recommendations:
  - WebP: 20-50 (good quality-to-size ratio)
  - JPEG: 70+ (acceptable quality)

### Utility Options

`-v, --verbose`
: Enable detailed logging. Shows progress, file retrieval information, and debug messages.

## COMMANDS

### mosaic

Generate a tiled mosaic/grid of images from a DICOM series.

**SYNOPSIS**

```
dicom-series-preview mosaic SERIESUID OUTPUT [OPTIONS]
```

**ARGUMENTS**

`SERIESUID`
: DICOM Series UID (see Series Specification above)

`OUTPUT`
: Output image path (must be `.webp`, `.jpg`, or `.jpeg`)

**OPTIONS**

`--tile-width WIDTH`
: Number of images per row in the mosaic grid.
: Default: `6`

`--tile-height HEIGHT`
: Number of images per column in the mosaic grid.
: Default: Same as `--tile-width`

`--start POSITION`
: Start of normalized z-position range (0.0-1.0). 0.0 is superior (head), 1.0 is inferior (tail).
: Default: `0.0`
: When combined with `--end`, only instances within the range are selected

`--end POSITION`
: End of normalized z-position range (0.0-1.0).
: Default: `1.0`

**EXAMPLES**

```bash
# Default 6x6 grid from entire series
dicom-series-preview mosaic 38902e14-b11f-4548-910e-771ee757dc82 mosaic.webp

# Custom grid: 8 columns, 5 rows
dicom-series-preview mosaic 38902e14-b11f-4548-910e-771ee757dc82 mosaic.webp \
  --tile-width 8 --tile-height 5

# Smaller tiles with better image width
dicom-series-preview mosaic 38902e14-b11f-4548-910e-771ee757dc82 mosaic.webp \
  --image-width 256 --tile-width 4 --tile-height 4

# Middle 60% of series with lung contrast
dicom-series-preview mosaic 38902e14-b11f-4548-910e-771ee757dc82 mosaic.webp \
  --start 0.2 --end 0.8 --contrast lung --quality 40

# High-quality output
dicom-series-preview mosaic 38902e14-b11f-4548-910e-771ee757dc82 mosaic.jpg \
  --quality 80 --image-width 200
```

**BEHAVIOR**

1. All instances are sorted by z-position (spatial location), then by instance number
2. Instances are evenly distributed across the normalized range `[--start, --end]`
3. Requested grid size determines how many instances are selected
4. Missing instances result in blank tiles
5. Contrast settings are applied uniformly to all tiles

---

### get-image

Extract a single image from a DICOM series at a specific position.

**SYNOPSIS**

```
dicom-series-preview get-image SERIESUID OUTPUT [OPTIONS]
```

**ARGUMENTS**

`SERIESUID`
: DICOM Series UID

`OUTPUT`
: Output image path (must be `.webp`, `.jpg`, or `.jpeg`)

**OPTIONS**

`--position POSITION` (required)
: Normalized z-position to extract (0.0-1.0).
: 0.0 = beginning (superior/head), 1.0 = end (inferior/tail), 0.5 = middle

`--slice-offset OFFSET`
: Offset from `--position` by number of slices (can be negative).
: Default: `0` (no offset)
: Examples:
  - `1` - next slice
  - `-1` - previous slice
  - `5` - 5 slices forward
  - `-3` - 3 slices backward
: Must stay within series bounds; out-of-bounds offset will error

**EXAMPLES**

```bash
# Image at 50% position (middle of series)
dicom-series-preview get-image 38902e14-b11f-4548-910e-771ee757dc82 middle.webp \
  --position 0.5

# Superior image with lung contrast
dicom-series-preview get-image 38902e14-b11f-4548-910e-771ee757dc82 superior.webp \
  --position 0.0 --contrast lung

# Inferior image with custom window/level
dicom-series-preview get-image 38902e14-b11f-4548-910e-771ee757dc82 inferior.webp \
  --position 1.0 --contrast 350/50

# Adjacent slices from same position
dicom-series-preview get-image 38902e14-b11f-4548-910e-771ee757dc82 slice_1.webp \
  --position 0.5 --slice-offset 1
dicom-series-preview get-image 38902e14-b11f-4548-910e-771ee757dc82 slice_-1.webp \
  --position 0.5 --slice-offset -1

# High-resolution single image
dicom-series-preview get-image 38902e14-b11f-4548-910e-771ee757dc82 hires.jpg \
  --position 0.5 --image-width 512 --quality 85
```

**BEHAVIOR**

1. Position is mapped to instances using an intelligent strategy:
   - If series has varying z-positions (e.g., multi-slice CT): position selects based on spatial location
   - If series has multiple instances at same location (e.g., temporal/cardiac): position selects by time
   - Otherwise: position selects by instance order
2. Slice offset is applied after position selection
3. Offset must stay within series bounds (0 to N-1 instances)
4. Output is always a single image (no tiling)

---

### contrast-mosaic

Create a grid comparing instance(s) under multiple contrast settings.

**SYNOPSIS**

```
dicom-series-preview contrast-mosaic SERIESUID OUTPUT --contrast PRESET [--contrast PRESET ...] [OPTIONS]
```

**ARGUMENTS**

`SERIESUID`
: DICOM Series UID

`OUTPUT`
: Output image path (must be `.webp`, `.jpg`, or `.jpeg`)

**OPTIONS (Instance Selection - choose one mode)**

**Single Instance Mode:**

`--position POSITION`
: Extract single instance at normalized position (0.0-1.0).
: Grid will show this instance under multiple contrasts (1 row × N contrasts columns)

`--slice-offset OFFSET`
: Offset from `--position` by number of slices.
: Only valid with `--position`

**Multiple Instance Mode:**

`--start POSITION`
: Start of z-position range (0.0-1.0).
: Creates multiple rows of instances

`--end POSITION`
: End of z-position range (0.0-1.0).
: Default: `1.0`

`--tile-height HEIGHT`
: Number of instances per column (rows).
: Only used with `--start`/`--end`.
: Default: `2`

**Contrast Settings:**

`--contrast PRESET` (required, repeatable)
: One or more contrast presets (repeatable flag).
: Each preset creates a column in the output grid.
: See Contrast Options in COMMON OPTIONS for valid presets.
: Examples:
  - `--contrast lung --contrast bone --contrast brain`
  - `--contrast auto --contrast embedded`
  - `--contrast 1500/500 --contrast 2000/300`

**EXAMPLES**

```bash
# Single instance, 3 contrast variations
dicom-series-preview contrast-mosaic 38902e14-b11f-4548-910e-771ee757dc82 contrasts.webp \
  --position 0.5 \
  --contrast lung --contrast bone --contrast brain

# Range of instances with 2 contrasts (2 instances × 2 contrasts = 2x2 grid)
dicom-series-preview contrast-mosaic 38902e14-b11f-4548-910e-771ee757dc82 comparison.webp \
  --start 0.3 --end 0.7 --tile-height 2 \
  --contrast ct-brain --contrast ct-abdomen

# Four contrasts of middle image
dicom-series-preview contrast-mosaic 38902e14-b11f-4548-910e-771ee757dc82 four_windows.webp \
  --position 0.5 \
  --contrast lung --contrast bone --contrast brain --contrast soft

# Auto vs embedded contrast comparison
dicom-series-preview contrast-mosaic 38902e14-b11f-4548-910e-771ee757dc82 auto_vs_embedded.webp \
  --position 0.5 \
  --contrast auto --contrast embedded

# Custom window/level comparisons
dicom-series-preview contrast-mosaic 38902e14-b11f-4548-910e-771ee757dc82 custom.webp \
  --position 0.5 \
  --contrast 1500/500 --contrast 2000/300 --contrast 400/40 --contrast 350/50
```

**BEHAVIOR**

1. Grid layout: contrasts on x-axis (columns), instances on y-axis (rows)
2. Single instance mode: 1 row, N columns (where N = number of contrasts)
3. Range mode: M rows (M = tile-height), N columns (N = number of contrasts)
4. Each cell is rendered independently with its contrast setting
5. Contrast settings are applied uniformly across instances

---

### build-index

Pre-build cached index files for one or more DICOM series.

**SYNOPSIS**

```
dicom-series-preview build-index SERIES [SERIES ...] [OPTIONS]
```

**ARGUMENTS**

`SERIES`
: One or more series UIDs/paths (repeatable positional arguments)

**OPTIONS**

`--root PATH`
: Root path for DICOM series.
: Default: `s3://idc-open-data`

`--cache-dir DIR`
: Output directory for index files.
: Indices stored as `{CACHE_DIR}/indices/{SERIESUID}_index.parquet`
: Mutually exclusive with `-o/--output`

`-o, --output DIR`
: Output directory for a single series.
: Only valid when exactly one series is specified.
: Mutually exclusive with `--cache-dir`

`--limit INT`
: Limit the number of instances to process (useful for very large series).
: Default: no limit (process all instances)

**EXAMPLES**

```bash
# Build index for single series in default cache
dicom-series-preview build-index 38902e14-b11f-4548-910e-771ee757dc82

# Build multiple indices in cache directory
dicom-series-preview build-index \
  38902e14-b11f-4548-910e-771ee757dc82 \
  45678abc-def0-1234-5678-90abcdef1234 \
  --cache-dir ~/.cache/dicom-indices

# Build index with verbose output
dicom-series-preview build-index 38902e14-b11f-4548-910e-771ee757dc82 \
  --cache-dir /tmp/indices -v

# Build index from custom HTTP server
dicom-series-preview build-index abc123def456 \
  --root http://hospital.example.com/dicom \
  --cache-dir /tmp/indices

# Limit to first 100 instances (for large series)
dicom-series-preview build-index 38902e14-b11f-4548-910e-771ee757dc82 \
  --limit 100 --cache-dir ~/.cache/dicom
```

**OUTPUT FORMAT**

Index files are Parquet tables with one row per DICOM instance:

**Metadata Columns**

- **Index** (UInt32): Zero-based sort position (0 to N-1)
- **FileName** (Utf8): Instance filename (derived from SOPInstanceUID)
- **SeriesUID** (Utf8): Series UID
- **StorageRoot** (Utf8): Storage root path

**Sorting Information**

- **PrimaryPosition** (Float32): Actual coordinate on primary axis
  - For spatial scans: X, Y, or Z coordinate in millimeters
  - For instance-number-only: InstanceNumber value
- **PrimaryAxis** (Utf8): Which axis was used for sorting
  - `'X'` - Sagittal (left-right varies most)
  - `'Y'` - Coronal (anterior-posterior varies most)
  - `'Z'` - Axial (superior-inferior varies most)
  - `'I'` - Instance number only (no spatial coordinates)

**Instance Information**

- **InstanceNumber** (Int32): DICOM InstanceNumber tag
- **SOPInstanceUID** (Utf8): Unique instance identifier
- **SliceLocation** (Float32): DICOM SliceLocation tag (if available)

**Dynamic Columns**

- All other DICOM header elements that vary across the series
- Numeric types (IS, DS, US, etc.) → Int32 or Float32
- Text types (CS, LO, PN, etc.) → Utf8
- Binary data (OB, OW) → Two columns: `{TagName}_Size` (Int32) and `{TagName}_Hash` (Utf8)

**Constant Values**

DICOM elements with identical values across all instances are excluded (derivable from metadata).

**BEHAVIOR**

1. Captures all DICOM headers from the series
2. Sorts instances by z-position (primary) and instance number (secondary)
3. Exports metadata to strongly-typed Parquet format
4. File size typically 1-5MB depending on series size and metadata

**PERFORMANCE**

- Typical time: 10-30 seconds per series depending on network
- Future image commands skip header fetching when index exists (massive speedup: 2-3 seconds vs 10-30 seconds)

---

### get-index

Retrieve or create a DICOM series index and return its path.

**SYNOPSIS**

```
dicom-series-preview get-index SERIES [OPTIONS]
```

**ARGUMENTS**

`SERIES`
: DICOM Series UID

**OPTIONS**

`--root PATH`
: Root path for DICOM series.
: Default: `s3://idc-open-data`

`--cache-dir DIR`
: Directory for storing/loading index files.
: If not specified, uses default cache location

**EXAMPLES**

```bash
# Get index path (create if doesn't exist)
dicom-series-preview get-index 38902e14-b11f-4548-910e-771ee757dc82

# Get index path in custom cache directory
dicom-series-preview get-index 38902e14-b11f-4548-910e-771ee757dc82 \
  --cache-dir /tmp/my-indices

# Verbose output showing what's happening
dicom-series-preview get-index 38902e14-b11f-4548-910e-771ee757dc82 -v
```

**BEHAVIOR**

1. Checks if index already exists in cache
2. If found: returns the path immediately
3. If not found: builds index (same as `build-index` command) and returns path
4. Returns absolute path to `.parquet` file
5. Useful for integrating into scripts or other tools

**OUTPUT**

Prints the full path to the index file, e.g.:
```
/Users/username/.cache/dicom-indices/indices/38902e14-b11f-4548-910e-771ee757dc82_index.parquet
```

---

## SORTING AND INSTANCE SELECTION

### Two-Level Sorting

All instances are sorted using a two-level strategy:

1. **Primary**: Z-position (spatial location) from `ImagePositionPatient[2]` or `SliceLocation`
2. **Secondary**: `InstanceNumber` (for multiple instances at same z-position)

This ensures:
- Instances follow radiological convention (superior → inferior)
- Multi-instance temporal sequences are properly ordered
- Both static 3D volumes and 4D temporal data are handled correctly

### Position Mapping Strategy

When selecting by position (0.0-1.0), the tool uses this priority:

1. **Spatial variation detected**: Maps position to z-position range
   - Example: In a 181-slice series from z=-792 to z=-488, position 0.5 selects the middle z-coordinate

2. **Temporal data detected**: Maps position to time range (via time tags or instance number)
   - Example: Cardiac series with 25 timepoints; position 0.5 selects middle timepoint

3. **Default**: Maps position to instance index
   - Example: Series with 50 instances; position 0.5 selects instance 25

### Slice Offset

Applied after position mapping:
- `--slice-offset 1` → one instance forward
- `--slice-offset -1` → one instance backward
- Must stay within series bounds (0 to N-1)

### Range Selection

`--start` and `--end` filter by z-position range:
- start_z = min_z + (max_z - min_z) × start
- end_z = min_z + (max_z - min_z) × end
- Only instances with z-position in [start_z, end_z] are selected
- Instances are then distributed across requested tiles

---

## CONTRAST SETTINGS

Contrast is determined in this priority order:

1. **Command-line**: `--contrast` argument (highest priority)
2. **DICOM file**: WindowWidth/WindowCenter in header (if available)
3. **Auto-detection**: Calculated from pixel statistics (2nd-98th percentile)

### Windowing Algorithm

Linear windowing with hard clipping:
- Values < (center - width/2) → black (0)
- Values > (center + width/2) → white (255)
- Values in between → linearly interpolated

### Common Presets

| Preset | Window | Center | Use Case |
|--------|--------|--------|----------|
| `ct-lung` | 1500 | -500 | Pulmonary imaging |
| `ct-bone` | 2000 | 300 | Bone, dental |
| `ct-brain` | 80 | 40 | Brain, stroke |
| `ct-abdomen` | 350 | 50 | Abdominal organs |
| `ct-liver` | 150 | 30 | Liver imaging |
| `ct-mediastinum` | 350 | 50 | Mediastinal structures |
| `ct-soft-tissue` | 400 | 50 | Soft tissue, muscles |

---

## CACHING SYSTEM

### Automatic Caching

By default, DICOM series indices are cached after first access:

1. After retrieving headers, index is saved to cache directory
2. On subsequent runs, index is loaded from cache (no header fetching)
3. **Performance impact**: Typical 2-3 second cached vs 10-30 second fresh

### Cache Location

Default locations (in order of preference):
1. `DICOM_SERIES_PREVIEW_CACHE_DIR` environment variable
2. Platform-specific cache directory:
   - Linux: `~/.cache/dicom-series-preview`
   - macOS: `~/Library/Caches/dicom-series-preview`
   - Windows: `%APPDATA%\dicom-series-preview\cache`

### Disabling Cache

Use `--no-cache` flag to fetch fresh on every run:
```bash
dicom-series-preview mosaic SERIES OUTPUT --no-cache
```

### Custom Cache Directory

Use `--cache-dir` to specify alternative location:
```bash
dicom-series-preview mosaic SERIES OUTPUT --cache-dir /tmp/my-indices
```

---

## STORAGE BACKENDS

### S3 (Default)

Accesses public IDC bucket by default:
```bash
# Uses s3://idc-open-data by default
dicom-series-preview mosaic SERIES output.webp

# Use private bucket with credentials
dicom-series-preview mosaic SERIES output.webp \
  --root s3://my-private-bucket
```

Environment variables (if credentials needed):
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`

### HTTP

Accesses DICOM files via HTTP:
```bash
dicom-series-preview mosaic SERIES output.webp \
  --root http://hospital.example.com/dicom-server
```

### Local Filesystem

Uses local DICOM files:
```bash
dicom-series-preview mosaic SERIES output.webp \
  --root /data/dicom

# Or with file:// prefix
dicom-series-preview mosaic SERIES output.webp \
  --root file:///data/dicom
```

---

## COMPRESSION CODEC SUPPORT

**Supported (always)**
- Uncompressed
- JPEG Baseline
- JPEG Lossless (non-hierarchical)
- RLE (Run-Length Encoding)

**Unsupported without additional libraries**
- JPEG Extended (12-bit)
- JPEG 2000
- MPEG

To support these, install optional gdcm dependency (Linux/Windows only):
```bash
pip install dicom-series-preview[gdcm]
```

---

## ERROR HANDLING

The tool exits with code 1 on errors:

**Common errors:**
- Invalid series UID format
- Series not found in storage
- Invalid output file extension (must be `.webp`, `.jpg`, `.jpeg`)
- Unsupported compression codec
- Network errors fetching DICOM files
- Invalid position/range parameters (must be 0.0-1.0)
- Slice offset out of bounds

**Troubleshooting:**

Use `--verbose/-v` for detailed error messages:
```bash
dicom-series-preview mosaic SERIES output.webp -v
```

---

## EXIT STATUS

- `0` - Success
- `1` - Error (invalid arguments, not found, processing failure, etc.)

---

## ENVIRONMENT

`DICOM_SERIES_PREVIEW_CACHE_DIR`
: Override default cache directory for indices

`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`
: AWS credentials for private S3 buckets (IDC bucket is public)

---

## PERFORMANCE NOTES

### Typical Timings

- **First run** (no cache): 10-30 seconds (headers must be fetched)
- **Cached run**: 2-3 seconds (headers already cached)
- **Network latency**: Dominant factor, varies by location/bandwidth

### Bandwidth Usage

- **Header retrieval**: ~5KB per instance × number of instances
- **Pixel data**: Depends on tile count and image size
- **Total**: Typically 5-50MB depending on series size

### Optimization Tips

- Use `--cache-dir` to reuse indices across multiple commands
- Use `--image-width 64-128` for faster preview operations
- Use `--tile-width/height` appropriate for use case (don't request huge grids)
- Smaller `--quality` values (20-30) for fast preview, larger (50+) for archival

### Quality Settings

| Format | Quality | Size/Tile | Use Case |
|--------|---------|-----------|----------|
| WebP | 20-30 | 1-5KB | Fast preview |
| WebP | 50+ | 5-20KB | Good quality |
| JPEG | 70+ | 10-30KB | High quality |
| JPEG | 85+ | 20-50KB | Archive |

---

## INSTALLATION

### Using uv (Recommended)

```bash
uv pip install .
uv run dicom-series-preview --help
```

### Using pip

```bash
pip install .
dicom-series-preview --help
```

### Optional Dependencies

For additional DICOM codec support:
```bash
pip install dicom-series-preview[gdcm]
```

---

## EXAMPLES

### Quick Preview

```bash
# Default 6x6 mosaic
dicom-series-preview mosaic 38902e14-b11f-4548-910e-771ee757dc82 preview.webp
```

### Detailed Analysis

```bash
# Compare contrasts at specific location
dicom-series-preview contrast-mosaic 38902e14-b11f-4548-910e-771ee757dc82 analysis.webp \
  --position 0.5 \
  --contrast lung --contrast bone --contrast brain --contrast soft

# View multiple slices in one image
dicom-series-preview mosaic 38902e14-b11f-4548-910e-771ee757dc82 multi_slice.webp \
  --tile-width 4 --tile-height 4 --image-width 192 \
  --start 0.3 --end 0.7
```

### Automated Processing

```bash
# Build cache first
dicom-series-preview build-index SERIES1 SERIES2 --cache-dir /cache

# Later: fast processing from cache
dicom-series-preview mosaic SERIES1 output1.webp --cache-dir /cache
dicom-series-preview mosaic SERIES2 output2.webp --cache-dir /cache
```

### Integration with Scripts

```bash
#!/bin/bash
# Get index and pass to external tool
INDEX=$(dicom-series-preview get-index $SERIES --cache-dir /cache)
python analyze_dicom.py --index "$INDEX"
```

---

## NOTES

### Series UID Format

Standard DICOM Series UID format is a 32-character hexadecimal string (UUID):
```
38902e14-b11f-4548-910e-771ee757dc82  (with hyphens)
38902e14b11f4548910e771ee757dc82      (without hyphens)
```

Both formats are accepted and normalized automatically.

### Radiological Convention

Images are displayed in standard radiological order:
- Axial: superior (head) → inferior (feet)
- Coronal: posterior → anterior
- Sagittal: right → left

This matches medical imaging conventions and is determined automatically from spatial metadata.

### Parquet Index Format

Index files are stored in Apache Parquet format:
- Strongly-typed columns (efficient storage)
- Supports columnar compression
- Compatible with: Pandas, Polars, DuckDB, Arrow, PySpark
- Self-documenting schema

---

## SEE ALSO

- DICOM Standard: ISO/IEC 23912
- pydicom documentation: https://pydicom.readthedocs.io/
- Imaging Data Commons: https://imagingdatacommons.cancer.gov/

---

## HISTORY

**Version 0.1.0** (current)
- Five commands: mosaic, get-image, contrast-mosaic, build-index, get-index
- S3, HTTP, and local filesystem support
- Caching system with Parquet indices
- Multiple contrast presets and custom windowing
- Prefix-based series search

---

## AUTHOR

Generated with Claude Code

## BUGS

Report issues at: https://github.com/anthropics/claude-code/issues
