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

## Commands

### image
Generate a single image from a DICOM series at a specified position.

```bash
idc-series-preview image <series-uid> output.webp --position 0.5 --width 512
```

### mosaic
Create a tiled grid of images sampling across the series.

```bash
idc-series-preview mosaic <series-uid> output.webp --samples 9 --width 768
```

### contrast-mosaic
Compare the same slice under different window/level contrast settings.

```bash
idc-series-preview contrast-mosaic <series-uid> output.webp --position 0.5 -c lung -c bone -c soft
```

### video
Generate an MP4 video scrolling through the series.

```bash
idc-series-preview video <series-uid> output.mp4 --fps 15 --width 512
```

### headers
Display DICOM header metadata for a series.

```bash
idc-series-preview headers <series-uid> --format json
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

## Series UID Resolution

Accepts multiple UID formats:
- Full DICOM UID: `1.3.6.1.4...`
- IDC UUID: `38902e14-b11f-4548-910e-771ee757dc82`
- Dotted UUID: `38902e14.b11f.4548.910e.771ee757dc82`

## Documentation

Full documentation is available in the `docs/` directory:
- `docs/idc-series-preview.md` - Complete CLI reference (man page format)
- `docs/API_GUIDE.md` - Python API usage guide
- `docs/DEPENDENCY_DIAGRAM.md` - Module architecture
