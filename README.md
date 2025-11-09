# DICOM Mosaic Generator

Generate DICOM mosaics from medical imaging series stored on S3, HTTP, or local filesystems.

## Features

- **Efficient Retrieval**: Uses range requests to fetch only DICOM headers, minimizing data transfer
- **Distributed Sampling**: Selects representative images distributed across the entire series
- **Flexible Storage**: Supports S3, HTTP, and local file storage via obstore
- **Customizable Output**:
  - Adjustable tile grid size (default 6x6)
  - Configurable tile and output image dimensions
  - Support for WebP and JPEG output formats
  - Quality control (0-100)
- **Contrast Adjustment**:
  - Preset window/center settings for different anatomies (lung, bone, brain, etc.)
  - Auto-detection of optimal contrast parameters
  - Manual window/center specification

## Installation

```bash
# Install with uv
uv sync

# Or install dependencies directly
pip install obstore pydicom numpy pillow
```

### AWS Credentials for S3 Access

The IDC S3 bucket is publicly accessible but requires AWS credentials for obstore to make signed requests. Configure credentials before use:

**Option 1: AWS CLI** (Recommended)
```bash
aws configure
# Enter your AWS Access Key ID and Secret Access Key
# Even free AWS tier credentials work for reading public IDC data
```

**Option 2: Environment Variables**
```bash
export AWS_ACCESS_KEY_ID=your_key_id
export AWS_SECRET_ACCESS_KEY=your_secret_key
```

**Option 3: AWS Credentials File** (~/.aws/credentials)
```ini
[default]
aws_access_key_id = your_key_id
aws_secret_access_key = your_secret_key
```

**Verify Access Without Credentials**
You can test if the bucket is accessible using s5cmd with unsigned requests:
```bash
s5cmd --no-sign-request ls 's3://idc-open-data/38902e14-b11f-4548-910e-771ee757dc82/*'
```

If this works, the data is publicly accessible. Configure AWS credentials above to use with dicom-mosaic.

## Usage

```bash
# Basic usage with auto-detected contrast
python dicom_mosaic.py <series_uid> <output_file>

# With custom grid size (4x4 instead of default 6x6)
python dicom_mosaic.py <series_uid> <output_file> --tile-width 4

# With preset contrast (lung, bone, brain, etc.)
python dicom_mosaic.py <series_uid> <output_file> --contrast-preset lung

# With manual window/center settings
python dicom_mosaic.py <series_uid> <output_file> --window-width 1500 --window-center -500

# Adjust output quality
python dicom_mosaic.py <series_uid> <output_file> -q 90

# Verbose output for debugging
python dicom_mosaic.py <series_uid> <output_file> -v

# Custom storage location
python dicom_mosaic.py <series_uid> <output_file> --root s3://my-bucket/dicom/
```

## Arguments

### Required Arguments
- `seriesuid`: DICOM Series UID
- `output`: Output file path (.webp or .jpg)

### Optional Arguments

**Storage:**
- `--root`: Root path for DICOM files (default: `s3://idc-open-data`)

**Tiling:**
- `--tile-width`: Number of images per row (default: 6)
- `--tile-height`: Number of images per column (default: same as --tile-width)

**Image Scaling:**
- `--image-width`: Width of each tile in pixels (default: 128)
  - Height is scaled proportionally based on aspect ratio

**Contrast:**
- `--contrast-preset`: Preset name: `lung`, `bone`, `abdomen`, `brain`, `mediastinum`, `liver`, `soft_tissue`, or `auto`
- `--window-width`: Window width in HU (Hounsfield Units)
- `--window-center`: Window center in HU

**Output:**
- `-q, --quality`: Output quality 0-100 (default: 85)

**Utility:**
- `-v, --verbose`: Enable verbose logging

## Contrast Presets

The following standard contrast presets are available:

| Preset | Window Width | Window Center | Use Case |
|--------|--------------|---------------|----------|
| lung | 1500 | -500 | Lung tissue, respiratory anatomy |
| bone | 2000 | 300 | Bone structures |
| abdomen | 350 | 50 | Abdominal organs |
| brain | 80 | 40 | Brain tissue, intracranial |
| mediastinum | 350 | 50 | Mediastinal structures |
| liver | 150 | 30 | Liver tissue |
| soft_tissue | 400 | 50 | Soft tissue imaging |
| auto | - | - | Auto-detect from image |

## Example Commands

```bash
# Generate a 6x6 mosaic from IDC DICOM series with lung contrast
python dicom_mosaic.py 38902e14-b11f-4548-910e-771ee757dc82 mosaic.webp \
  --contrast-preset lung -q 90

# Create a smaller 4x4 mosaic with larger tiles
python dicom_mosaic.py 38902e14-b11f-4548-910e-771ee757dc82 mosaic.jpg \
  --tile-width 4 --tile-height 4 --image-width 256

# Local DICOM series with auto-detected contrast
python dicom_mosaic.py my-series-uid output.webp \
  --root /path/to/dicom/root --contrast-preset auto
```

## How It Works

1. **Instance Discovery**: Lists all DICOM instances in the specified series
2. **Distributed Sampling**: Selects N instances evenly distributed across the series
3. **Header Retrieval**: Fetches only DICOM headers via range requests (5-20KB per file)
4. **Image Processing**:
   - Extracts pixel data and applies rescale slope/intercept
   - Applies windowing (contrast) based on preset or auto-detection
   - Resizes images to match the specified tile width
5. **Mosaic Creation**: Arranges tiles left-to-right, top-to-bottom
6. **Output**: Saves as WebP or JPEG with specified quality

## Performance Considerations

- **Minimal Data Transfer**: Uses 5KB range requests for headers, only downloads full images if needed
- **Distributed Sampling**: Retrieves only ~36 images for a default 6x6 mosaic, regardless of series size
- **Parallel Potential**: Currently sequential but can be parallelized for multiple series
- **Memory Efficient**: Processes images on-the-fly without loading entire series into memory

## Architecture

```
dicom_mosaic.py     - CLI interface and argument parsing
├── src/
│   ├── retriever.py - DICOM instance retrieval and listing
│   ├── mosaic.py    - Mosaic creation and image output
│   └── contrast.py  - Window/level presets and windowing logic
```

## Dependencies

- **obstore**: High-performance object storage interface
- **pydicom**: DICOM file reading and parsing
- **numpy**: Numerical operations
- **pillow**: Image processing and output

## License

MIT
