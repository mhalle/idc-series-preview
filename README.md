# DICOM Mosaic Generator

Generate tiled mosaic images from DICOM series stored on S3, HTTP, or local filesystem.

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python: 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)

## Overview

`dicom-mosaic` is a command-line tool that retrieves DICOM medical imaging instances from various storage backends, arranges them in a tiled grid, and exports the result as a WebP or JPEG image. It's designed for efficient processing of large series using:

- **Range requests**: Retrieve only DICOM headers initially (5KB) to determine instance ordering
- **Distributed sampling**: Select evenly-spaced instances across the full series range
- **Parallel I/O**: Concurrent fetching of headers and pixel data for speed
- **Smart windowing**: Support for anatomical presets, file-stored settings, or auto-detection
- **Flexible storage**: Works with S3, HTTP, local paths, or any obstore-compatible backend

## Quick Start

### Installation

Using [uv](https://docs.astral.sh/uv/) (recommended):

```bash
git clone https://github.com/yourusername/dicom-mosaic.git
cd dicom-mosaic
uv sync
```

Using pip:

```bash
pip install -e .
```

### Basic Usage

```bash
# Generate a 6x6 mosaic from an IDC series (default S3 bucket)
dicom-series-preview mosaic 38902e14-b11f-4548-910e-771ee757dc82 output.webp

# With custom tile grid
dicom-series-preview mosaic 38902e14-b11f-4548-910e-771ee757dc82 output.webp \
  --tile-width 8 --tile-height 6 --image-width 64

# With lung contrast preset
dicom-series-preview mosaic 38902e14-b11f-4548-910e-771ee757dc82 output.webp \
  --contrast lung -q 50

# From local filesystem
dicom-series-preview mosaic d94176e6-bc8e-4666-b143-639754258d06 output.webp \
  --root /path/to/dicom/series

# With verbose output
dicom-series-preview mosaic 38902e14-b11f-4548-910e-771ee757dc82 output.webp -v
```

## Features

### 🎯 Instance Selection

- Automatically selects distributed subset of instances across the series
- Maintains radiological ordering (superior to inferior)
- Always includes first and last instance in series
- Falls back to all instances if series smaller than tile count

### 📊 Contrast Presets

Built-in presets for common anatomical regions:
- **lung**: WW=1500, WC=-500
- **bone**: WW=2000, WC=300
- **brain**: WW=80, WC=40
- **abdomen**: WW=350, WC=50
- **mediastinum**: WW=350, WC=50 (shortcut: **media**)
- **liver**: WW=150, WC=30
- **soft-tissue**: WW=400, WC=50 (shortcut: **soft**)
- **auto**: Auto-detect from image statistics
- **embedded**: Use window/level from DICOM file (falls back to auto if not present)

Custom values also supported via window,level format: `--contrast 1500,500`

### 🚀 Performance

- **Two-pass retrieval**: Headers first for sorting, then pixel data for selected instances
- **Parallel I/O**: Up to 8 concurrent connections for faster downloads
- **Efficient storage**: Range requests reduce bandwidth for header-only pass
- **Smart fallback**: Full file retrieval if range requests not supported

### 📁 Storage Backends

- **S3**: Public and authenticated buckets via boto3/obstore
- **HTTP(S)**: Direct access to web servers
- **Local filesystem**: Direct file access
- **Other**: Any backend supported by obstore (GCS, Azure, etc.)

## Command-Line Options

### Mosaic Subcommand

```
Usage: dicom-series-preview mosaic [OPTIONS] SERIESUID OUTPUT

Arguments:
  SERIESUID              DICOM Series UID (full, partial with hyphens, or prefix with wildcard)
  OUTPUT                 Output image path (.webp or .jpg)

Options:
  --root PATH            Root path for DICOM files (default: s3://idc-open-data)
  --tile-width WIDTH     Images per row (default: 6)
  --tile-height HEIGHT   Images per column (default: same as tile-width)
  --image-width PIXELS   Width of each tile in pixels (default: 128)
  --contrast SPEC        Contrast settings: preset (lung, bone, brain, abdomen, liver, mediastinum, soft-tissue),
                         shortcut (soft, media), 'auto', 'embedded', or window,level (e.g., 1500,500)
  --start FLOAT          Start of normalized z-position range 0.0-1.0 (default: 0.0)
  --end FLOAT            End of normalized z-position range 0.0-1.0 (default: 1.0)
  -q, --quality LEVEL    Output quality 0-100 (default: 25)
  -v, --verbose          Enable detailed logging
  --help                 Show this help message
```

### Get-Image Subcommand

```
Usage: dicom-series-preview get-image [OPTIONS] SERIESUID OUTPUT

Arguments:
  SERIESUID              DICOM Series UID (full, partial with hyphens, or prefix with wildcard)
  OUTPUT                 Output image path (.webp or .jpg)

Options:
  --root PATH            Root path for DICOM files (default: s3://idc-open-data)
  --image-width PIXELS   Width of image in pixels (default: 128)
  --contrast SPEC        Contrast settings: preset (lung, bone, brain, abdomen, liver, mediastinum, soft-tissue),
                         shortcut (soft, media), 'auto', 'embedded', or window,level (e.g., 1500,500)
  --position FLOAT       Extract image at normalized z-position 0.0-1.0 (required)
  --slice-offset INT     Offset from position by number of slices (default: 0)
  -q, --quality LEVEL    Output quality 0-100 (default: 25)
  -v, --verbose          Enable detailed logging
  --help                 Show this help message
```

## Examples

### From IDC (default S3 bucket)
```bash
dicom-series-preview mosaic 38902e14-b11f-4548-910e-771ee757dc82 output.webp
```

### Custom size and quality
```bash
dicom-series-preview mosaic 38902e14-b11f-4548-910e-771ee757dc82 output.webp \
  --tile-width 10 --tile-height 8 --image-width 100 -q 40
```

### With lung preset contrast
```bash
dicom-series-preview mosaic 38902e14-b11f-4548-910e-771ee757dc82 output.webp \
  --contrast lung -q 50
```

### With custom window,level values
```bash
dicom-series-preview mosaic 38902e14-b11f-4548-910e-771ee757dc82 output.webp \
  --contrast 350,50
```

### Using shortcut for soft-tissue preset
```bash
dicom-series-preview mosaic 38902e14-b11f-4548-910e-771ee757dc82 output.webp \
  --contrast soft
```

### From custom HTTP server
```bash
dicom-series-preview mosaic 38902e14-b11f-4548-910e-771ee757dc82 output.webp \
  --root http://dicom-server.example.com/series
```

### From local directory
```bash
dicom-series-preview mosaic d94176e6-bc8e-4666-b143-639754258d06 output.webp \
  --root /mnt/dicom-storage
```

### With all verbose details
```bash
dicom-series-preview mosaic 38902e14-b11f-4548-910e-771ee757dc82 output.webp \
  --contrast lung -q 50 -v
```

### Using range selection (middle 60% of series)
```bash
dicom-series-preview mosaic 38902e14-b11f-4548-910e-771ee757dc82 output.webp \
  --start 0.2 --end 0.8
```

### Using range selection (first 25% of series)
```bash
dicom-series-preview mosaic 38902e14-b11f-4548-910e-771ee757dc82 output.webp \
  --start 0.0 --end 0.25
```

### Extract single image at position (no tiling)
```bash
# Image at the beginning
dicom-series-preview get-image 38902e14-b11f-4548-910e-771ee757dc82 output.webp \
  --position 0.0

# Image at the middle
dicom-series-preview get-image 38902e14-b11f-4548-910e-771ee757dc82 output.webp \
  --position 0.5

# Image at the end
dicom-series-preview get-image 38902e14-b11f-4548-910e-771ee757dc82 output.webp \
  --position 1.0
```

## Input Format

### Series UID Specification

The tool accepts series UIDs in multiple formats:

- **Full UUID**: `38902e14-b11f-4548-910e-771ee757dc82`
- **32-char hex**: `38902e14b11f4548910e771ee757dc82`
- **Partial prefix**: `38902e14*` or `389*` (searches and selects matching series)

### DICOM Storage Structure

The tool expects DICOM files organized as:
```
{ROOT}/{SERIES_UID}/{INSTANCE_UID}.dcm
```

Example:
```
s3://idc-open-data/38902e14-b11f-4548-910e-771ee757dc82/instance-001.dcm
s3://idc-open-data/38902e14-b11f-4548-910e-771ee757dc82/instance-002.dcm
...
```

## Output Format

### Supported Output Formats

- **WebP** (recommended): Modern format with excellent compression. Quality 25-30 provides good balance.
- **JPEG**: Traditional format with wider compatibility. Quality 70+ recommended.

### Quality Settings

| Format | Quality | File Size | Use Case |
|--------|---------|-----------|----------|
| WebP | 20-25 | 1-3 KB/tile | Web, thumbnails |
| WebP | 30-40 | 3-8 KB/tile | Web viewing |
| WebP | 50+ | 8-15 KB/tile | High quality |
| JPEG | 70-80 | 10-20 KB/tile | High quality, compatibility |

## Behavior Details

### Instance Ordering

DICOM instances are sorted by a two-level key for proper spatial and temporal ordering:

**Primary sort (Spatial)**: z-position using (in order of preference):
1. `ImagePositionPatient[2]` (z-coordinate in patient space)
2. `SliceLocation` (deprecated but still used)

**Secondary sort (Temporal)**: `InstanceNumber` for sequences with multiple instances at the same spatial location

This two-level sorting ensures that:
- Instances are correctly arranged from superior (head) to inferior (tail)
- Temporal sequences (e.g., cardiac imaging at different time points) maintain proper temporal order within the same spatial location
- The tool correctly handles both 3D static volumes and 4D temporal sequences

### Windowing Strategy

Window/level settings are applied in this priority:
1. **User explicitly requests "embedded"**: Uses DICOM file's window/level, falls back to auto-detection
2. **Command-line --contrast argument**: Preset name, shortcut, custom values, "auto", or "embedded"
3. **Default behavior** (no --contrast specified): Try file metadata, fall back to auto-detection

Window/level sources:
- **Presets**: Built-in anatomical presets (lung, bone, brain, etc.)
- **Shortcuts**: Quick aliases (soft for soft-tissue, media for mediastinum)
- **Custom values**: Direct window,level specification (e.g., 1500,500)
- **Embedded**: WindowWidth/WindowCenter from DICOM file metadata
- **Auto**: Calculated from pixel statistics (2nd to 98th percentile)

### Retrieval Optimization

The tool uses a smart two-pass strategy:

**Pass 1 - Header Retrieval**:
- Fetch 5KB from each instance (sufficient for DICOM headers)
- Parallel requests (up to 8 concurrent)
- Extract metadata needed for sorting and window/level

**Pass 2 - Pixel Data Retrieval**:
- Fetch full pixel data only for selected instances
- Parallel requests (up to 8 concurrent)
- Converts to PIL images with windowing applied

This reduces bandwidth significantly for large series.

### Range Selection

The `--start` and `--end` options allow you to select a subset of the series based on normalized z-position range:

- **Range**: 0.0 (beginning) to 1.0 (end) of the series
- **Default**: --start 0.0 --end 1.0 (full series)

**Examples:**
- `--start 0.0 --end 0.5`: First 50% of series (head/superior region)
- `--start 0.5 --end 1.0`: Last 50% of series (tail/inferior region)
- `--start 0.2 --end 0.8`: Middle 60% of series

**Behavior:**
1. Instances are sorted by z-position (spatial location)
2. Min and max z-values are calculated
3. Range is mapped to actual z-values: start_z = min_z + (max_z - min_z) × start
4. Only instances with z-position between start_z and end_z (inclusive) are selected
5. If fewer instances are found than tiles requested, returns all instances in range
6. Remaining tiles in the grid are filled with black/blank tiles

This is useful for:
- Focusing on specific anatomical regions (e.g., upper/lower abdomen, proximal/distal limb)
- Creating separate mosaics for different parts of a large series
- Sampling specific slices without needing to know exact indices

### Single Image Extraction

The `--position` option extracts a single DICOM instance at a specific normalized z-position without creating a mosaic:

- **Range**: 0.0 (beginning/superior) to 1.0 (end/inferior) of the series
- **Output**: Single image at `--image-width` pixels wide, aspect ratio preserved
- **No tiling**: Outputs just one image, not a grid
- **Incompatible with**: `--start`, `--end`, `--tile-width`, `--tile-height`

**How it works:**
1. All instances sorted by z-position
2. Position mapped to closest instance: `target_z = min_z + (max_z - min_z) × position`
3. Instance with z-value closest to target_z is selected
4. Image extracted, windowed, and resized to `--image-width`

**Use cases:**
- Extract key anatomical slice (e.g., --position 0.5 for middle image)
- Sample representative images from different parts of series
- Create individual images for presentation or comparison
- Reference images for specific anatomical levels

**Examples:**
```bash
# Superior slice at beginning
dicom-series-preview get-image <uid> superior.webp --position 0.0

# Middle slice
dicom-series-preview get-image <uid> middle.webp --position 0.5

# Inferior slice at end
dicom-series-preview get-image <uid> inferior.webp --position 1.0

# Specific level with bone preset
dicom-series-preview get-image <uid> image.webp --position 0.33 --contrast bone

# With custom window,level
dicom-series-preview get-image <uid> image.webp --position 0.5 --contrast 1500,500
```

## Supported DICOM Codecs

### ✅ Fully Supported

- Uncompressed (no compression)
- JPEG Baseline (8-bit)
- JPEG Lossless (non-hierarchical)
- RLE (Run-Length Encoding)

### ❌ Not Supported (without gdcm)

- JPEG Extended (12-bit or higher precision)
- JPEG 2000
- HEVC (H.265)
- Other advanced codecs

**Note**: The tool will error cleanly if it encounters unsupported codecs. To support additional codecs, install the optional `gdcm` dependency (Linux/Windows only):

```bash
pip install dicom-mosaic[gdcm]
```

## Project Structure

```
dicom-mosaic/
├── README.md                      # This file
├── pyproject.toml                 # Project configuration and dependencies
├── dicom_mosaic.py                # CLI entry point and argument parsing
├── src/
│   ├── retriever.py               # DICOM instance retrieval from storage
│   ├── mosaic.py                  # Image tiling and mosaic assembly
│   └── contrast.py                # Windowing presets and contrast algorithms
├── docs/
│   ├── dicom-mosaic.md            # Markdown version of man page
│   └── dicom-mosaic.1             # Traditional man page (troff format)
└── tests/                         # Test suite

```

## Architecture

### DICOMRetriever

Handles all storage I/O and instance retrieval:
- Initializes storage backend (S3, HTTP, local, etc.)
- Retrieves DICOM headers with range requests
- Implements two-pass parallel fetching
- Manages connection pooling and error handling

### MosaicGenerator

Processes DICOM data and assembles final image:
- Extracts pixel arrays from DICOM datasets
- Applies windowing/level contrast
- Resizes tiles while maintaining aspect ratio
- Arranges tiles in grid and creates final mosaic

### ContrastPresets

Manages window/level settings:
- Provides anatomical presets (lung, bone, brain, etc.)
- Implements percentile-based auto-detection
- Applies linear windowing algorithm

## Performance Characteristics

### Typical Execution

| Operation | Time | Notes |
|-----------|------|-------|
| Series discovery | 1-2s | Lists available instances |
| Header retrieval | 2-4s | 36 instances at 5KB each |
| Pixel data retrieval | 3-5s | Full instance sizes vary |
| Image processing | 1-2s | Windowing and resizing |
| Mosaic assembly | <1s | Grid assembly |
| Output encoding | 1-2s | WebP/JPEG compression |
| **Total** | **8-15s** | For typical 36-instance mosaic |

### Bandwidth Usage

| Phase | Typical | Notes |
|-------|---------|-------|
| Headers | 180 KB | 36 instances × 5KB |
| Pixel data | 50-100 MB | 36 instances × 1-3MB each |
| **Total** | **50-100 MB** | Depends on instance resolution |

## Troubleshooting

### No instances found
- Verify series UID is correct
- Check that root path is accessible
- For S3, ensure bucket is public or credentials are configured

### Unsupported compression codec error
- JPEG Extended with 12-bit precision requires gdcm
- Install: `pip install dicom-mosaic[gdcm]` (Linux/Windows only)
- Or provide a series with supported codecs

### Output image is too bright/dark
- Check if image uses non-standard WindowWidth/WindowCenter
- Try `--contrast auto` to auto-detect from pixel statistics
- Try `--contrast embedded` to use DICOM file's stored window/level
- Adjust manually with `--contrast 1500,500` (custom window,level values)

### Slow performance
- Reduce `--image-width` for faster processing
- Use smaller tile grid (`--tile-width 4 --tile-height 4`)
- Check network connectivity to storage backend

## Dependencies

- **obstore** (≥0.1.0): High-performance object storage abstraction
- **pydicom** (≥2.4.0): DICOM file reading and parsing
- **numpy** (≥1.24.0): Array operations and image processing
- **pillow** (≥10.0.0): Image creation, resizing, and encoding
- **gdcm** (optional): Support for advanced DICOM codecs

## Development

### Setup Development Environment

```bash
# Clone and install with dev dependencies
git clone https://github.com/yourusername/dicom-mosaic.git
cd dicom-mosaic
uv sync --all-extras
```

### Running Tests

```bash
pytest tests/
pytest --cov=src tests/  # With coverage
```

### Code Quality

```bash
black src/ dicom_mosaic.py
ruff check src/ dicom_mosaic.py
mypy src/
```

## Documentation

- **Man page**: See `docs/dicom-mosaic.1` (can be installed to `/usr/share/man/man1/`)
- **Markdown docs**: See `docs/dicom-mosaic.md`
- **API documentation**: Generated from docstrings in source code

To view the man page:
```bash
man ./docs/dicom-mosaic.1
# Or after installation:
man dicom-mosaic
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Run quality checks (`black`, `ruff`, `mypy`)
5. Commit with clear messages
6. Push to your fork
7. Open a Pull Request

## Citation

If you use this tool in research, please cite:

```bibtex
@software{dicom-mosaic,
  title={DICOM Mosaic Generator},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/dicom-mosaic}
}
```

## Acknowledgments

- [Imaging Data Commons (IDC)](https://imaging.datacommons.cancer.gov/) - Public DICOM dataset
- [pydicom](https://pydicom.readthedocs.io/) - DICOM file format support
- [obstore](https://github.com/chanzuckerberg/obstore) - Object storage abstraction
- [Pillow](https://python-pillow.org/) - Image processing

## Support

- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Ask questions on GitHub Discussions
- **Documentation**: See `docs/` directory

## Changelog

### v0.1.0 (2025-11-09)

Initial release with:
- Support for S3, HTTP, and local DICOM retrieval
- Efficient two-pass header/pixel data fetching
- Anatomical contrast presets
- Auto-detection of windowing
- WebP and JPEG output formats
- Verbose logging
- Comprehensive documentation
