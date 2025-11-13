# IDC Series Preview

Command-line utilities for sampling, rendering, and indexing DICOM series in the Imaging Data Commons (IDC). The tooling focuses on fast header retrieval, cached indices, and repeatable window/level presets so you can inspect series without pulling every instance.

> **Test scope:** Only IDC (S3) data paths are exercised in the CI/release flow. HTTP and local filesystem roots exist for convenience but are not guaranteed.

## Installation

```bash
git clone https://github.com/mhalle/idc-series-preview.git
cd idc-series-preview
uv sync            # sets up the virtual environment
```

Run commands through `uv run` unless you activate the venv manually:

```bash
uv run idc-series-preview --help

# or run directly without cloning
uvx git+https://www.github.com/mhalle/idc-series-preview --help
```

## Quick Examples

```bash
# Default 6x6 mosaic from IDC
uv run idc-series-preview mosaic 38902e14-b11f-4548-910e-771ee757dc82 out.webp

# Middle 60% only, lung preset
uv run idc-series-preview mosaic 38902e14-b11f-4548-910e-771ee757dc82 lung.webp \
  --start 0.2 --end 0.8 -c lung

# Single slice, high quality JPEG
uv run idc-series-preview image 38902e14-b11f-4548-910e-771ee757dc82 slice.jpg \
  --position 0.5 --image-width 512 -q 85

# Contrast comparison grid
uv run idc-series-preview contrast-mosaic 38902e14-b11f-4548-910e-771ee757dc82 cmp.webp \
  --position 0.5 -c lung -c bone -c mediastinum

# Warm the cache for multiple series
uv run idc-series-preview build-index 38902e14-b11f-4548-910e-771ee757dc82 \
  45678abc-def0-1234-5678-90abcdef1234 --cache-dir ~/.cache/idc-indices
```

## Core Commands

| Command | Purpose |
| --- | --- |
| `mosaic` | Evenly sample a series and tile the images into a WebP/JPEG grid. Automatically shrinks rows when the requested range has fewer slices than slots. |
| `image` | Grab a single slice by normalized position with optional slice-offset. |
| `contrast-mosaic` | Compare one or more slices across multiple window/level presets. |
| `build-index` | Create parquet indices (`indices/{uid}_index.parquet`) for faster later access. |
| `get-index` | Ensure an index exists and either print its path or export it as parquet/JSON/JSONL. |
| `clear-index` | Remove cached index files by UID prefix or via `--all`. |

All commands share:
- `--root PATH` (defaults to `s3://idc-open-data`)
- `--image-width`, `-q/--quality`, `-c/--contrast`
- Cache flags: `--cache-dir PATH`, `--no-cache`
- `-v/--verbose`

### Contrast Presets

The preset system accepts both canonical names and shortcuts:

| Names | Window / Level |
| --- | --- |
| `ct-lung`, `lung` | WW 1500 / WL -500 |
| `ct-bone`, `bone` | 2000 / 300 |
| `ct-abdomen`, `abdomen` | 350 / 50 |
| `ct-brain`, `brain` | 80 / 40 |
| `ct-mediastinum`, `mediastinum`, `media` | 350 / 50 |
| `ct-vascular`, `vascular` | 700 / 200 |
| `ct-liver`, `liver` | 150 / 30 |
| `ct-soft-tissue`, `soft-tissue`, `soft` | 400 / 50 |
| `mr-t1`, `t1` | 700 / 300 |
| `mr-t2`, `t2` | 475 / 155 |
| `mr-proton`, `proton` | 920 / 420 |

You can also supply `window/level` or `window,level` directly, or use `auto` / `embedded`.

## Index Workflow

1. `build-index` generates parquet indices under `CACHE/indices/`. Each row stores the SOPInstanceUID, DataURL, primary axis metadata, normalized index (`IndexNormalized`), and every header field that varies.
2. `get-index` reuses those files when present. If you need JSON or JSONL output, use `--output format:/path`. CSV support was intentionally removed to avoid lossy list serialization.
3. `clear-index` deletes cached entries by UID or via `--all`.

## Development

- Run tests with `uv run python -m pytest`
- Lint/format using `uv run ruff check` and `uv run ruff format`
- Docs live in `docs/idc-series-preview.md`; update the README simultaneously when CLI behavior changes.

## License

MIT © Massachusetts General Hospital (MGH).
