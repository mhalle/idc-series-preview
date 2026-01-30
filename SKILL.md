---
name: idc-series-preview
description: Preview and visualize DICOM imaging series from the NCI Imaging Data Commons (IDC). Use when the user wants to view, preview, or visualize IDC series data. Triggers on "IDC image", "IDC series", "IDC preview", "Imaging Data Commons", or when working with IDC series UIDs.
license: MIT
compatibility: Requires uv and Python 3.9+. Requires network access to IDC S3 storage. Video generation requires ffmpeg to be installed.
metadata:
  author: mhalle
  repository: https://github.com/mhalle/idc-series-preview
  download: https://github.com/mhalle/idc-series-preview/releases/latest/download/idc-series-preview.skill
---

# idc-series-preview: DICOM Series Preview Tools

CLI tools for previewing and visualizing DICOM imaging series from the NCI Imaging Data Commons (IDC) and other sources.

## Download

Latest release:
- https://github.com/mhalle/idc-series-preview/releases/latest/download/idc-series-preview.zip
- https://github.com/mhalle/idc-series-preview/releases/latest/download/idc-series-preview.skill

## Version Check and Update

To check if this skill is up to date:

1. **Get current version**: Read `$SKILL_DIR/src/idc_series_preview/_version.py` or check `fallback-version` in `$SKILL_DIR/pyproject.toml`

2. **Get latest version**: Fetch from GitHub API:
   ```
   https://api.github.com/repos/mhalle/idc-series-preview/releases/latest
   ```
   The `tag_name` field contains the latest version (e.g., `v0.6.1`).

3. **If outdated**: Download the latest skill file and present it to the user:
   ```
   https://github.com/mhalle/idc-series-preview/releases/latest/download/idc-series-preview.skill
   ```
   Presenting the downloaded `.skill` file may trigger a dialog allowing the user to update their installed skill.

## Installation

Run the install script:

```bash
sh $SKILL_DIR/scripts/install.sh
```

Or install manually with uv (recommended) or pip:

```bash
uv tool install $SKILL_DIR
# or
pip install $SKILL_DIR
```

Then run from any directory:

```bash
idc-series-preview <command> [options]
```

To reinstall/upgrade:

```bash
sh $SKILL_DIR/scripts/install.sh --force
```

Note: `$SKILL_DIR` refers to the skill directory path (e.g., `/mnt/skills/idc-series-preview` depending on platform).

## Critical: Series UID Resolution for IDC Data

**When working with IDC data, you must use the `crdc_series_uuid`, NOT the DICOM `SeriesInstanceUID`.**

IDC's S3 bucket (`s3://idc-open-data`) organizes files by `crdc_series_uuid` (a UUID like `ca81385b-facf-487e-aa50-2a5d0b97e173`), not by DICOM SeriesInstanceUID (like `1.3.6.1.4.1.14519...`). If you pass a DICOM SeriesInstanceUID, the tool will fail to find files.

### When querying with idc-index, always include crdc_series_uuid:

```python
from idc_index import IDCClient
client = IDCClient()

df = client.sql_query("""
    SELECT SeriesInstanceUID, crdc_series_uuid, series_size_MB, 
           Modality, SeriesDescription
    FROM index 
    WHERE collection_id = 'nlst'
    AND Modality = 'CT'
    LIMIT 5
""")

# Use crdc_series_uuid with idc-series-preview, NOT SeriesInstanceUID
uuid = df['crdc_series_uuid'].iloc[0]
```

Then use the UUID:
```bash
idc-series-preview mosaic "ca81385b-facf-487e-aa50-2a5d0b97e173" output.webp
```

### Accepted UID formats:
- **IDC UUID (required for IDC data)**: `38902e14-b11f-4548-910e-771ee757dc82`
- Dotted UUID: `38902e14.b11f.4548.910e.771ee757dc82`
- Full DICOM UID: Only works for non-IDC sources where files are organized by SeriesInstanceUID

## Known Limitations: JPEG Transfer Syntaxes

Some older DICOM data uses **JPEG Extended with 12-bit precision**, which Pillow cannot decode. This is common in certain collections (e.g., some NLST series). 

**Symptoms:**
```
NotImplementedError: Pillow does not support 'JPEG Extended' for samples with 12-bit precision
```

**Workaround:** Try a different series from the same collection. There's no way to filter by transfer syntax in idc-index queries, so this requires trial and error. Larger series (50-100MB) from the same patient often use different encoding than smaller ones.

## Commands

### image
Generate a single image from a DICOM series at a specified position.

```bash
idc-series-preview image <series-uuid> output.webp --position 0.5 --width 512
```

### mosaic
Create a tiled grid of images sampling across the series.

```bash
idc-series-preview mosaic <series-uuid> output.webp --samples 9 --width 768
```

### contrast-mosaic
Compare the same slice under different window/level contrast settings.

```bash
idc-series-preview contrast-mosaic <series-uuid> output.webp --position 0.5 -c lung -c bone -c soft
```

### video
Generate an MP4 video scrolling through the series.

```bash
idc-series-preview video <series-uuid> output.mp4 --fps 15 --width 512
```

### headers
Display DICOM header metadata for a series.

```bash
idc-series-preview headers <series-uuid> --format json
```

## Contrast Presets

Built-in presets for common viewing windows:
- `lung` (W1500/L-500), `bone` (W2000/L300), `brain` (W80/L40)
- `abdomen` (W350/L50), `liver` (W150/L30), `soft` (W400/L50)
- `t1`, `t2`, `proton` for MR imaging
- `auto` - automatically detect from pixel statistics
- `embedded` - use values from DICOM file
- Custom: `1500/-500` or `1500,-500`

## Labels

Images include overlay labels showing:
- **Top-left**: Normalized position (0.0000 to 1.0000)
- **Bottom-right**: Window/Level contrast (e.g., W2000/L-600)

Use `--no-labels` to disable.

## Complete Workflow Example

```python
# 1. Query IDC for series (note: include crdc_series_uuid!)
from idc_index import IDCClient
client = IDCClient()

df = client.sql_query("""
    SELECT SeriesInstanceUID, crdc_series_uuid, series_size_MB, 
           PatientID, Modality, SeriesDescription
    FROM index 
    WHERE collection_id = 'nlst'
    AND Modality = 'CT'
    AND series_size_MB BETWEEN 50 AND 100
    LIMIT 5
""")

# 2. Get the UUID (not the SeriesInstanceUID!)
series_uuid = df['crdc_series_uuid'].iloc[0]
print(f"Using series: {series_uuid}")
```

```bash
# 3. Generate preview using the UUID
idc-series-preview mosaic "ca81385b-facf-487e-aa50-2a5d0b97e173" output.webp \
    --samples 9 --width 768 --contrast lung
```

## Documentation

Full documentation is available in the `docs/` directory:
- `docs/idc-series-preview.md` - Complete CLI reference (man page format)
- `docs/API_GUIDE.md` - Python API usage guide
- `docs/DEPENDENCY_DIAGRAM.md` - Module architecture
