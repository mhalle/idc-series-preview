# dicom-mosaic(1) - Generate DICOM image mosaics

## SYNOPSIS

```
dicom-mosaic [OPTIONS] SERIESUID OUTPUT
```

## DESCRIPTION

`dicom-mosaic` retrieves DICOM instances from a series stored on S3, HTTP, or local filesystem, creates a tiled mosaic image, and exports the result in WebP or JPEG format.

The tool uses efficient range requests to retrieve only DICOM headers initially, then fetches full pixel data only for the selected instances. It supports various window/level (contrast) presets and auto-detection of optimal display settings.

## ARGUMENTS

### SERIESUID
The DICOM Series UID. Can be specified in multiple formats:
- Full 32-character hex UID: `38902e14b11f4548910e771ee757dc82`
- Standard UUID format: `38902e14-b11f-4548-910e-771ee757dc82`
- Partial prefix with wildcard: `38902e14*` or `389*`

### OUTPUT
Output image file path. Must have a `.webp`, `.jpg`, or `.jpeg` extension.

## OPTIONS

### Storage Options

`--root PATH`
: Root path for DICOM files. Supports S3 URLs, HTTP(S) URLs, or local filesystem paths.
: Default: `s3://idc-open-data`
: Examples:
  - `s3://my-bucket/dicom-data`
  - `http://example.com/medical-images`
  - `/data/dicom` or `file:///data/dicom` for local paths

### Tiling Options

`--tile-width WIDTH`
: Number of images per row in the mosaic.
: Default: `6`

`--tile-height HEIGHT`
: Number of images per column in the mosaic.
: Default: Same as `--tile-width`

`--image-width PIXELS`
: Width of each image tile in pixels. Height is scaled proportionally to maintain aspect ratio.
: Default: `128`

### Contrast Options

`--contrast-preset PRESET`
: Use a predefined contrast preset for the anatomical region. Available presets:
  - `auto` - Auto-detect optimal window/center from image statistics
  - `lung` - Lung window (WW=1500, WC=-500)
  - `bone` - Bone window (WW=2000, WC=300)
  - `brain` - Brain window (WW=80, WC=40)
  - `abdomen` - Abdomen window (WW=350, WC=50)
  - `mediastinum` - Mediastinum window (WW=350, WC=50)
  - `liver` - Liver window (WW=150, WC=30)
  - `soft_tissue` - Soft tissue window (WW=400, WC=50)

`--window-width WIDTH`
: Window width for manual contrast adjustment (in Hounsfield Units).

`--window-center CENTER`
: Window center for manual contrast adjustment (in Hounsfield Units).

### Output Options

`-q, --quality LEVEL`
: Output image quality (0-100).
: Default: `25`
: Lower values produce smaller files but reduced quality. Values of 25-50 are recommended for WebP.

### Utility Options

`-v, --verbose`
: Enable verbose logging output. Shows detailed progress information and debug messages.

## EXAMPLES

### Basic mosaic from default S3 bucket
```
dicom-mosaic 38902e14-b11f-4548-910e-771ee757dc82 output.webp
```

### Custom tile grid and size
```
dicom-mosaic 38902e14-b11f-4548-910e-771ee757dc82 output.webp \
  --tile-width 8 --tile-height 6 --image-width 64
```

### With lung contrast preset and custom quality
```
dicom-mosaic 38902e14-b11f-4548-910e-771ee757dc82 output.webp \
  --contrast-preset lung -q 50
```

### Manual window/center settings
```
dicom-mosaic 38902e14-b11f-4548-910e-771ee757dc82 output.webp \
  --window-width 350 --window-center 50
```

### Auto-detect contrast
```
dicom-mosaic 38902e14-b11f-4548-910e-771ee757dc82 output.webp \
  --contrast-preset auto
```

### From local filesystem
```
dicom-mosaic d94176e6-bc8e-4666-b143-639754258d06 output.webp \
  --root /local/dicom/path
```

### From custom HTTP server
```
dicom-mosaic 38902e14-b11f-4548-910e-771ee757dc82 output.webp \
  --root http://dicom-server.example.com
```

### Verbose output with logging
```
dicom-mosaic 38902e14-b11f-4548-910e-771ee757dc82 output.webp -v
```

## BEHAVIOR

### Instance Selection

By default, the tool selects a distributed subset of instances from the series to create the mosaic:

- If the series has fewer instances than requested tiles, all instances are used
- If the series has more instances, they are:
  1. Sorted by z-position (anatomical location) using `ImagePositionPatient`, `InstanceNumber`, or `SliceLocation`
  2. Evenly distributed across the full z-range
  3. Always includes the first and last instance in the series

### Contrast/Window Settings

Contrast settings are determined in this priority order:

1. **Command-line arguments** - `--contrast-preset`, `--window-width`/`--window-center`
2. **File metadata** - WindowWidth/WindowCenter stored in DICOM headers
3. **Auto-detection** - Calculated from image statistics (2nd to 98th percentile)

### Instance Retrieval

The tool uses a two-pass retrieval strategy for efficiency:

- **First pass**: Retrieve headers (5KB) for all instances in parallel to determine z-positions
- **Second pass**: Fetch full pixel data only for selected instances

This reduces S3 bandwidth for large series while ensuring all metadata needed for proper sorting is available.

### Windowing

Linear windowing with hard clipping is applied:
- Values below (center - width/2) are mapped to black (0)
- Values above (center + width/2) are mapped to white (255)
- Values in between are linearly scaled

## ERROR HANDLING

The tool will exit with error code 1 if:

- Series UID is invalid or not found
- Output file format is not .webp or .jpg/.jpeg
- No valid images can be retrieved from the series
- DICOM files use unsupported compression codecs (e.g., JPEG Extended with 12-bit precision)

On error, detailed messages are logged to help diagnose the issue. Use `-v/--verbose` for debug information.

## COMPRESSION CODEC SUPPORT

Currently supported DICOM compression codecs:
- Uncompressed (no compression)
- JPEG Baseline
- JPEG Lossless (non-hierarchical)
- RLE (Run-Length Encoding)

Unsupported codecs (without gdcm):
- JPEG Extended (with 12-bit precision)
- JPEG 2000

To support additional codecs, install the optional `gdcm` dependency (available on Linux/Windows):
```
pip install dicom-mosaic[gdcm]
```

## INSTALLATION

Using uv (recommended):
```
uv pip install .
uv run dicom-mosaic --help
```

Using pip:
```
pip install .
dicom-mosaic --help
```

## ENVIRONMENT

The tool respects standard environment variables for S3 access when using credentials (though the default IDC S3 bucket is public):

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`

## EXIT STATUS

- `0` - Success
- `1` - Error (invalid arguments, file not found, processing error, etc.)

## NOTES

### Performance

- Typical mosaic generation takes 5-10 seconds depending on network latency and series size
- Bandwidth usage: 5-10MB for header retrieval + pixel data for selected instances
- Use smaller `--image-width` and `--tile-width` values for faster processing

### Quality Settings

- WebP quality 20-30: Good balance of quality and file size (~1-5KB per tile)
- WebP quality 50+: High quality (~5-20KB per tile)
- JPEG quality 70+: Good quality, larger files (~10-30KB per tile)

### Z-Position Sorting

Medical imaging convention is to display slices in radiological order (superior to inferior). The tool automatically sorts instances by their spatial location and ensures consistent display order.

## AUTHOR

Generated with Claude Code

## SEE ALSO

For more information about DICOM file format, see the DICOM standard (ISO/IEC 23912).
For pydicom documentation, visit https://pydicom.readthedocs.io/
