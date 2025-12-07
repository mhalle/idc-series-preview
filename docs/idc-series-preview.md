# idc-series-preview(1) - DICOM series preview and visualization

## NAME

`idc-series-preview` - Preview DICOM series with intelligent sampling, contrast adjustment, and flexible output formats

## SYNOPSIS

```
idc-series-preview COMMAND [OPTIONS] [ARGUMENTS]
```

## DESCRIPTION

`idc-series-preview` is a command-line tool for previewing DICOM medical imaging series stored on S3, HTTP, or local filesystems. It provides multiple visualization modes:

- **mosaic**: Generate tiled grids of images from a series
- **image**: Extract single images at specific positions
- **header**: Export cached header metadata for a specific slice (JSON)
- **video**: Render canonical slices to an MP4 using ffmpeg (libx264)
- **contrast-mosaic**: Create comparison grids with multiple contrast settings
- **build-index**: Pre-build cached indices for faster access
- **get-index**: Retrieve or create an index and return its path
- **clear-index**: Delete cached index files

The tool uses efficient retrieval strategies (headers first, then pixel data for selected instances) and supports advanced features like window/level presets, auto-contrast detection, and flexible series specification formats.

> **Testing scope:** The CLI is validated against Imaging Data Commons (IDC) datasets. HTTP and local filesystem sources are provided for convenience but are not part of the regular test matrix.

## COMMON OPTIONS

These options apply to most subcommands (where applicable):

### Series Specification

All commands accept flexible series specifications:

**Series UID Format**
- Full UUID: `38902e14-b11f-4548-910e-771ee757dc82`
- UUID without hyphens: `38902e14b11f4548910e771ee757dc82`
- Full path: `s3://idc-open-data/38902e14-b11f-4548-910e-771ee757dc82`
- Full path: `http://example.com/dicom/38902e14-b11f-4548-910e-771ee757dc82`
- Local path: `file:///data/dicom/38902e14-b11f-4548-910e-771ee757dc82`

Partial prefixes and wildcards are not supported; always specify the complete SeriesInstanceUID or a fully-qualified path.

### Storage Options

`--root PATH`
: Root path for DICOM series. Can be S3, HTTP(S), or local filesystem path.
: Default: `s3://idc-open-data`
: Examples:
  - `s3://idc-open-data` (public IDC bucket)
  - `s3://my-private-bucket/dicom`
  - `http://hospital.example.com/dicom-server`
  - `/data/dicom` or `file:///data/dicom` (local)
: Note: only IDC (S3) paths are officially tested today; HTTP and local paths are experimental.

### Canvas Options

`--width PIXELS`
: Total canvas width for mosaics/contrast grids, or per-frame width for single-image/video commands.
: Defaults to the native slice width for single frames and a reasonable tiling width for mosaics.
: Larger values produce higher-quality output but larger files.

`--height PIXELS`
: Optional total canvas height. When provided with `--width`, both dimensions are enforced; otherwise the missing dimension is derived from the rendered aspect ratio.

`--shrink-to-fit`
: For mosaics and contrast grids, scale the rendered image down to fit within `--width`/`--height` without padding. The final canvas may be smaller than requested but never stretches or distorts the imagery.

### Sampling Options

`-n/--samples COUNT`
: Number of slices to sample evenly across `--start/--end`. Applies to `mosaic`, `video`, and the range mode of `contrast-mosaic`.
: Defaults to 36 for mosaics, 120 for video, and 1 for contrast grids.

`--columns COUNT`
: For mosaics, force a specific number of columns. Rows are derived automatically (and vice versa when `--rows` is supplied).

`--rows COUNT`
: For mosaics, force the number of rows. Combined with `--columns`, this locks the grid size (excess samples are truncated; missing slices shrink the grid).

### Contrast Options

`-c/--contrast PRESET`
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
: By default caching is enabled using `IDC_SERIES_PREVIEW_CACHE_DIR` environment variable or platform cache directory

### Quality Options (Still Images)

`-q, --quality LEVEL`
: Output image compression quality (0-100).
: Default: `60`
: Recommendations:
  - WebP: 20-50 (good quality-to-size ratio)
  - JPEG: 70+ (acceptable quality)

### Video Options

`--fps FPS`
: Frames per second for the MP4 encoder.
: Default: `24`
: Accepts fractional values (e.g., `23.976`).
: All frames are re-sampled to a consistent resolution (via `--width`) and streamed directly into ffmpeg.

: `-n/--samples COUNT`
: Sample exactly `COUNT` slices evenly across the normalized range (`video` default: 120).
: Useful when you want predictable clip duration regardless of series length.

### Utility Options

`-v, --verbose`
: Enable detailed logging. Shows progress, file retrieval information, and debug messages.

## COMMANDS

### mosaic

Generate a tiled mosaic/grid of images from a DICOM series.

**SYNOPSIS**

```
idc-series-preview mosaic SERIESUID OUTPUT [OPTIONS]
```

**ARGUMENTS**

`SERIESUID`
: DICOM Series UID (see Series Specification above)

`OUTPUT`
: Output image path (must be `.webp`, `.jpg`, or `.jpeg`)

**OPTIONS**

: `-n/--samples COUNT`
: Number of slices to sample evenly across the range.
: Default: `36`

`--columns COUNT`
: Force a specific number of columns. Rows are derived automatically unless you also provide `--rows`.

`--rows COUNT`
: Force a specific number of rows. Combined with `--columns`, this locks the grid size (excess samples are truncated; missing slices shrink the grid automatically).

`--width/--height`
: Total mosaic size. Width defaults to a sensible canvas width; height auto-scales unless you provide an explicit value.

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
idc-series-preview mosaic 38902e14-b11f-4548-910e-771ee757dc82 mosaic.webp

# Custom grid: 8 columns, 5 rows
idc-series-preview mosaic 38902e14-b11f-4548-910e-771ee757dc82 mosaic.webp \
  --samples 40 --columns 8 --rows 5

# Smaller tiles with better image width
idc-series-preview mosaic 38902e14-b11f-4548-910e-771ee757dc82 mosaic.webp \
  --samples 16 --columns 4 --rows 4 --width 1024

# Middle 60% of series with lung contrast
idc-series-preview mosaic 38902e14-b11f-4548-910e-771ee757dc82 mosaic.webp \
  --start 0.2 --end 0.8 --contrast lung --quality 40

# High-quality output
idc-series-preview mosaic 38902e14-b11f-4548-910e-771ee757dc82 mosaic.jpg \
  --quality 80 --width 1200
```

**BEHAVIOR**

1. All instances are sorted by z-position (spatial location), then by instance number
2. Instances are evenly distributed across the normalized range `[--start, --end]`
3. Requested grid size determines how many instances are selected
4. If the requested range yields fewer unique slices than `--samples`, the grid shrinks automatically instead of duplicating tiles.
5. Contrast settings are applied uniformly to all tiles

---

### image

Extract a single image from a DICOM series at a specific position.

**SYNOPSIS**

```
idc-series-preview image SERIESUID OUTPUT [OPTIONS]
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

`--width PIXELS`
: Desired frame width. Defaults to the native DICOM width if omitted.

---

### header

Export the cached header metadata (from the parquet index) for the instance closest to a given position.

**SYNOPSIS**

```
idc-series-preview header SERIESUID --position POS [OPTIONS]
```

**ARGUMENTS**

`SERIESUID`
: Series UID or fully-qualified path.

**OPTIONS**

`-p/--position`
: Normalized z-position (0.0-1.0) used to locate the slice. Required.

`--slice-offset`
: Integer offset from the resolved slice (positive or negative). Default `0`.

`--output header.json`
: Optional JSON file destination. When omitted, the header is printed to stdout.

`--indent N`
: JSON indentation (default `2`; set to `0` for compact output).

`-t/--tag KEYWORD`
: Filter output to the specified DICOM keywords (repeatable). Case-insensitive.
: Use `--quiet` to suppress warnings when a requested tag is absent.

`--root`, `--cache-dir`, `--no-cache`, `-v/--verbose`
: Same semantics as other commands.

**BEHAVIOR**

1. Builds or reuses the cached index for the requested series (no pixel data fetched).
2. Finds the slice whose normalized index is closest to `--position`, applies `--slice-offset`, and serializes the entire row (all cached header fields) to JSON.
3. Writes to stdout or the file specified via `--output`.

---

### video

Render every slice (or a normalized subset) into an MP4 video using ffmpeg.

**SYNOPSIS**

```
idc-series-preview video SERIESUID OUTPUT [OPTIONS]
```

**ARGUMENTS**

`SERIESUID`
: Series UID or full path (same formats described earlier)

`OUTPUT`
: Output path ending in `.mp4`

**OPTIONS**

`--fps FPS`
: Frames per second. Default `24`.
: Accepts integers or fractional values (`23.976` etc.)

`--start POSITION`, `--end POSITION`
: Normalized range `[0.0, 1.0]` to limit slices.
: Defaults to the entire series.

: `-n/--samples COUNT`
: Number of frames to sample evenly across the selected range.
: Default: `120`

`--width PIXELS`
: Frame width during rendering. Defaults to the native DICOM width; specify to downsample.

`-c/--contrast PRESET`
: Applies presets, `auto`, `embedded`, or custom WW/WL before encoding.

**EXAMPLES**

```bash
# Canonical ordering at 24fps
idc-series-preview video 38902e14-b11f-4548-910e-771ee757dc82 series.mp4

# Focus on the middle 50% at 30fps with lung preset
idc-series-preview video 38902e14-b11f-4548-910e-771ee757dc82 lung.mp4 \
  --start 0.25 --end 0.75 --fps 30 --contrast lung
```

**BEHAVIOR**

1. All slices are visited in canonical order (using cached indices when available).
2. The tool interpolates `--samples` evenly spaced positions between `--start` and `--end` before rendering.
3. Frames are rendered with the selected width/contrast and streamed directly into ffmpeg-python (`libx264`, `yuv420p`).
4. The encoder writes MP4 files with `faststart` for progressive playback.
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
idc-series-preview image 38902e14-b11f-4548-910e-771ee757dc82 middle.webp \
  --position 0.5

# Superior image with lung contrast
idc-series-preview image 38902e14-b11f-4548-910e-771ee757dc82 superior.webp \
  --position 0.0 --contrast lung

# Inferior image with custom window/level
idc-series-preview image 38902e14-b11f-4548-910e-771ee757dc82 inferior.webp \
  --position 1.0 --contrast 350/50

# Adjacent slices from same position
idc-series-preview image 38902e14-b11f-4548-910e-771ee757dc82 slice_1.webp \
  --position 0.5 --slice-offset 1
idc-series-preview image 38902e14-b11f-4548-910e-771ee757dc82 slice_-1.webp \
  --position 0.5 --slice-offset -1

# High-resolution single image
idc-series-preview image 38902e14-b11f-4548-910e-771ee757dc82 hires.jpg \
  --position 0.5 --width 512 --quality 85
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
idc-series-preview contrast-mosaic SERIESUID OUTPUT -c/--contrast PRESET [-c/--contrast PRESET ...] [OPTIONS]
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
: Creates multiple rows of instances.

`--end POSITION`
: End of z-position range (0.0-1.0).
: Default: `1.0`

: `-n/--samples COUNT`
: Number of slices to sample within the specified range (default: 1).
: If the range contains fewer unique slices than requested, idc-series-preview shrinks the grid instead of repeating images.

**Contrast Settings:**

`-c/--contrast PRESET` (required, repeatable)
: One or more contrast presets (repeatable flag).
: Each preset creates a column in the output grid.
: See Contrast Options in COMMON OPTIONS for valid presets.
: Examples:
  - `--contrast lung --contrast bone --contrast brain`
  - `--contrast auto --contrast embedded`
  - `--contrast 1500/500 --contrast 2000/300`

`--width/--height`
: Total grid size. Width defaults to `len(contrasts) * 256`; height auto-scales unless explicitly provided.

`--shrink-to-fit`
: Avoid padding when both width and height are specified. The grid is scaled uniformly so it fits inside the requested dimensions without distortion. Any leftover canvas space is dropped instead of filled.

**EXAMPLES**

```bash
# Single instance, 3 contrast variations
idc-series-preview contrast-mosaic 38902e14-b11f-4548-910e-771ee757dc82 contrasts.webp \
  --position 0.5 \
  --contrast lung --contrast bone --contrast brain

# Range of instances with 2 contrasts (2 instances × 2 contrasts = 2x2 grid)
idc-series-preview contrast-mosaic 38902e14-b11f-4548-910e-771ee757dc82 comparison.webp \
  --start 0.3 --end 0.7 --samples 2 \
  --contrast ct-brain --contrast ct-abdomen

# Four contrasts of middle image
idc-series-preview contrast-mosaic 38902e14-b11f-4548-910e-771ee757dc82 four_windows.webp \
  --position 0.5 \
  --contrast lung --contrast bone --contrast brain --contrast soft

# Auto vs embedded contrast comparison
idc-series-preview contrast-mosaic 38902e14-b11f-4548-910e-771ee757dc82 auto_vs_embedded.webp \
  --position 0.5 \
  --contrast auto --contrast embedded

# Custom window/level comparisons
idc-series-preview contrast-mosaic 38902e14-b11f-4548-910e-771ee757dc82 custom.webp \
  --position 0.5 \
  --contrast 1500/500 --contrast 2000/300 --contrast 400/40 --contrast 350/50
```

**BEHAVIOR**

1. Grid layout: contrasts on x-axis (columns), instances on y-axis (rows)
2. Single instance mode: 1 row, N columns (where N = number of contrasts)
3. Range mode: M rows (M = number of sampled slices), N columns (N = number of contrasts)
4. Each cell is rendered independently with its contrast setting
5. Contrast settings are applied uniformly across instances

---

### build-index

Pre-build cached index files for one or more DICOM series.

**SYNOPSIS**

```
idc-series-preview build-index SERIES [SERIES ...] [OPTIONS]
```

**ARGUMENTS**

`SERIES`
: One or more series UIDs/paths (repeatable). Required unless using `--rebuild --all`.

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

`--rebuild`
: Force regeneration of each index even if a cached parquet already exists.

`--all`
: With `--rebuild` (and no SERIES arguments), rebuild every cached index in the cache directory.

**EXAMPLES**

```bash
# Build index for single series in default cache
idc-series-preview build-index 38902e14-b11f-4548-910e-771ee757dc82

# Build multiple indices in cache directory
idc-series-preview build-index \
  38902e14-b11f-4548-910e-771ee757dc82 \
  45678abc-def0-1234-5678-90abcdef1234 \
  --cache-dir ~/.cache/dicom-indices

# Build index with verbose output
idc-series-preview build-index 38902e14-b11f-4548-910e-771ee757dc82 \
  --cache-dir /tmp/indices -v

# Build index from custom HTTP server
idc-series-preview build-index abc123def456 \
  --root http://hospital.example.com/dicom \
  --cache-dir /tmp/indices

```

**OUTPUT FORMAT**

Index files are Parquet tables with one row per DICOM instance:

**Metadata Columns**

- **Index** (UInt32): Zero-based sort position (0 to N-1)
- **DataURL** (Utf8): Fully-qualified resource URL identifying each instance
- **SeriesUID** (Utf8): Series UID

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
idc-series-preview get-index SERIES [OPTIONS]
```

**ARGUMENTS**

`SERIES`
: DICOM Series UID or path (same formats as other commands)

`OUTPUT`
: Optional destination for the exported index. Supports:
  - Format prefixes: `csv:/tmp/index.csv`, `jsonl:relative/path.jsonl`
  - File extensions: `.parquet`, `.csv`, `.json`, `.jsonl` / `.ndjson`
  - If omitted, the command prints the cached Parquet path

**OPTIONS**

`--root PATH`
: Root path for DICOM series.
: Default: `s3://idc-open-data`

`--cache-dir DIR`
: Directory for storing/loading index files.
: If not specified, uses default cache location

`--rebuild`
: Force regeneration of the cached index before exporting/printing.

`--format {csv,json,jsonl,parquet}`
: Force export format when `OUTPUT` is provided. Overrides prefixes/extensions.

**EXAMPLES**

```bash
# Get index path (create if doesn't exist)
idc-series-preview get-index 38902e14-b11f-4548-910e-771ee757dc82

# Get index path in custom cache directory
idc-series-preview get-index 38902e14-b11f-4548-910e-771ee757dc82 \
  --cache-dir /tmp/my-indices

# Export index as CSV using prefix syntax
idc-series-preview get-index 38902e14-b11f-4548-910e-771ee757dc82 \
  csv:/tmp/series.csv

# Export index as JSONL with explicit --format
idc-series-preview get-index 38902e14-b11f-4548-910e-771ee757dc82 \
  /tmp/series.ndjson --format jsonl

# Verbose output showing what's happening
idc-series-preview get-index 38902e14-b11f-4548-910e-771ee757dc82 -v
```

**BEHAVIOR**

1. Checks if index already exists in cache
2. If found: returns/exports immediately
3. If not found: builds index (same as `build-index` command) and then exports
4. Supports exporting as Parquet (copy), CSV, JSON (array), or JSONL/NDJSON (line-delimited)
5. When no destination is given, prints the cached Parquet path for scripting

**OUTPUT**

Prints the full path to the cached or exported index, e.g.:
```
/Users/username/.cache/dicom-indices/indices/38902e14-b11f-4548-910e-771ee757dc82_index.parquet
```

---

### clear-index

Delete cached index files for specific series or the entire cache.

**SYNOPSIS**

```
idc-series-preview clear-index [SERIES ...] [OPTIONS]
```

**ARGUMENTS**

`SERIES`
: Optional series UIDs/paths. If omitted, `--all` is required.

**OPTIONS**

`--root PATH`
: Root path used when resolving SERIES inputs. Default: `s3://idc-open-data`

`--cache-dir DIR`
: Cache directory containing index files. Default: platform cache

`--all`
: Remove every cached index in the cache directory (cannot be combined with SERIES)

`-v, --verbose`
: Enable detailed logging of deleted files

**EXAMPLES**

```bash
# Remove cache entry for a single series
idc-series-preview clear-index 38902e14-b11f-4548-910e-771ee757dc82

# Clear entire cache directory
idc-series-preview clear-index --all --cache-dir ~/.cache/dicom-indices
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

1. **Command-line**: `-c/--contrast` argument (highest priority)
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
1. `IDC_SERIES_PREVIEW_CACHE_DIR` environment variable
2. Platform-specific cache directory:
   - Linux: `~/.cache/idc-series-preview`
   - macOS: `~/Library/Caches/idc-series-preview`
   - Windows: `%APPDATA%\idc-series-preview\cache`

### Disabling Cache

Use `--no-cache` flag to fetch fresh on every run:
```bash
idc-series-preview mosaic SERIES OUTPUT --no-cache
```

### Custom Cache Directory

Use `--cache-dir` to specify alternative location:
```bash
idc-series-preview mosaic SERIES OUTPUT --cache-dir /tmp/my-indices
```

---

## STORAGE BACKENDS

### S3 (Default)

Accesses public IDC bucket by default:
```bash
# Uses s3://idc-open-data by default
idc-series-preview mosaic SERIES output.webp

# Use private bucket with credentials
idc-series-preview mosaic SERIES output.webp \
  --root s3://my-private-bucket
```

Environment variables (if credentials needed):
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`

### HTTP

Accesses DICOM files via HTTP:
```bash
idc-series-preview mosaic SERIES output.webp \
  --root http://hospital.example.com/dicom-server
```

### Local Filesystem

Uses local DICOM files:
```bash
idc-series-preview mosaic SERIES output.webp \
  --root /data/dicom

# Or with file:// prefix
idc-series-preview mosaic SERIES output.webp \
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
pip install idc-series-preview[gdcm]
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
idc-series-preview mosaic SERIES output.webp -v
```

---

## EXIT STATUS

- `0` - Success
- `1` - Error (invalid arguments, not found, processing failure, etc.)

---

## ENVIRONMENT

`IDC_SERIES_PREVIEW_CACHE_DIR`
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
- Use smaller `--width` values (512-1024) for faster preview mosaics/videos
- Keep `--samples` modest or specify `--columns/--rows` appropriate for your use case
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
uv run idc-series-preview --help
```

### Using pip

```bash
pip install .
idc-series-preview --help
```

### Optional Dependencies

For additional DICOM codec support:
```bash
pip install idc-series-preview[gdcm]
```

---

## EXAMPLES

### Quick Preview

```bash
# Default 6x6 mosaic
idc-series-preview mosaic 38902e14-b11f-4548-910e-771ee757dc82 preview.webp
```

### Detailed Analysis

```bash
# Compare contrasts at specific location
idc-series-preview contrast-mosaic 38902e14-b11f-4548-910e-771ee757dc82 analysis.webp \
  --position 0.5 \
  --contrast lung --contrast bone --contrast brain --contrast soft

# View multiple slices in one image
idc-series-preview mosaic 38902e14-b11f-4548-910e-771ee757dc82 multi_slice.webp \
  --samples 16 --columns 4 --rows 4 --width 768 \
  --start 0.3 --end 0.7
```

### Automated Processing

```bash
# Build cache first
idc-series-preview build-index SERIES1 SERIES2 --cache-dir /cache

# Later: fast processing from cache
idc-series-preview mosaic SERIES1 output1.webp --cache-dir /cache
idc-series-preview mosaic SERIES2 output2.webp --cache-dir /cache
```

### Integration with Scripts

```bash
#!/bin/bash
# Get index and pass to external tool
INDEX=$(idc-series-preview get-index $SERIES --cache-dir /cache)
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

**Version 0.5.0** (current)
- New `SeriesIndex` API-first architecture for programmatic use
- CLI refactored to use unified API layer
- Five commands: mosaic, image, contrast-mosaic, build-index, get-index
- Added MR contrast presets: T1, T2, Proton Density
- Improved defaults: 256px images, 3x3 mosaics, quality=60
- Short command aliases: -p, -s, -e, -w, -q
- S3, HTTP, and local filesystem support
- Caching system with Parquet indices
- Multiple contrast presets (CT and MR) and custom windowing
- Prefix-based series search

**Version 0.1.0**
- Initial release with five commands
- Support for S3, HTTP, and local DICOM retrieval
- Basic caching and contrast presets

---

## AUTHOR

Generated with Claude Code

## BUGS

Report issues at: https://github.com/anthropics/claude-code/issues
