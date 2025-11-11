# Dependency Diagram

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    SeriesIndex API                       │
│  (api.py: SeriesIndex, Instance, Contrast classes)      │
└────┬──────────────┬──────────────┬─────────────┬────────┘
     │              │              │             │
     ▼              ▼              ▼             ▼
┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────┐
│ __main__ │  │IndexCache │  │Retriever │  │ Mosaic   │
│ (parse)  │  │ (caching) │  │(fetching)│  │Generator │
└──────────┘  └─────┬─────┘  └────┬─────┘  │(tiling)  │
                    │             │        └──────────┘
                    ▼             ▼
              ┌──────────────────────────┐
              │  HeaderCapture           │
              │  (index generation)      │
              └──────────┬───────────────┘
                         │
                         ▼
              ┌──────────────────────────┐
              │  Retriever               │
              │  (parallel fetching)     │
              └──────────┬───────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
      ┌─────────┐  ┌──────────┐  ┌──────────┐
      │ obstore │  │ pydicom  │  │ polars   │
      │(S3/HTTP)│  │(DICOM)   │  │(DataF)   │
      └─────────┘  └──────────┘  └──────────┘
```

## Detailed Dependencies

### API Layer (api.py)

**SeriesIndex class** uses:
- `IndexCache.load_or_generate_index()` → gets Polars DataFrame with columns: {Index, PrimaryPosition, FileName, SOPInstanceUID, ...}
- `DICOMRetriever(root_path, index_df)` → initialized once, reused
  - `get_instance_at_position(series_uid, position)` → (uid, dataset)
  - `_get_instance_headers(series_uid, uid)` → (dataset, size)
  - `get_instance_data(series_uid, filename)` → dataset
- `MosaicGenerator.tile_images(images)` → PIL Image
- `_parse_and_normalize_series()` → path parsing utility

**Instance class** uses:
- `dataset` (pydicom.Dataset) - cached in memory
- `MosaicGenerator.create_single_image()` → render

**Contrast class** uses:
- Pure data class, no dependencies

### Index Caching (index_cache.py)

```
load_or_generate_index()
├─ Try: load existing Parquet cache
└─ Fallback:
   └─ HeaderCapture(root_path)
      └─ capture_series_headers(series_uid)
         └─ retriever.list_instances()
         └─ retriever.get_instance_data() [parallel]
         └─ generates Polars DataFrame
      └─ save to Parquet cache
```

### Core Fetching (retriever.py)

```
DICOMRetriever(root_path, index_df)
├─ store (obstore) → S3/HTTP/local access
├─ index_df (optional Polars DataFrame)
│
├─ _get_instance_headers(series_uid, uid, max_bytes=10KB)
│  ├─ store.get_range() [5KB range request, fallback to full]
│  └─ pydicom.dcmread(stop_before_pixels=True)
│
├─ get_instance_data(series_uid, filename)
│  ├─ store.get()
│  └─ pydicom.dcmread()
│
├─ get_instance_at_position(series_uid, position)
│  ├─ Fast path: uses index_df for instant selection
│  └─ Slow path: fetch all headers, sort, select
│
└─ fetch_parallel(urls, headers_only, max_workers)
   ├─ calls _get_instance_headers() or get_instance_data()
   └─ ThreadPoolExecutor [parallel]
```

### Index Generation (header_capture.py)

```
HeaderCapture(root_path)
└─ capture_series_headers(series_uid)
   ├─ retriever.list_instances()
   ├─ retriever._get_instance_headers() [parallel]
   ├─ convert to dict with DICOM fields
   └─ generate_parquet_table()
      └─ slice_sorting.sort_slices() [for sorting info]
```

### Image Rendering (mosaic.py)

```
MosaicGenerator(tile_width, tile_height, image_width, window_settings)
├─ create_single_image(instance_uid, dataset, retriever)
│  └─ pixel_array → apply window/level → PIL Image
│
└─ tile_images(images)
   ├─ pad to grid size
   ├─ standardize sizes
   └─ paste into mosaic
```

## Data Flow

### 1. Initialization

```
SeriesIndex("series-uid")
  ↓
_parse_and_normalize_series() [get root_path]
  ↓
IndexCache.load_or_generate_index()
  ├─ Load: Parquet cache → Polars DataFrame
  └─ Generate: HeaderCapture → save Parquet → Polars DataFrame
  ↓
DICOMRetriever(root_path, index_df) [lazy init]
```

### 2. Fetching Instances

```
get_instances(positions=[...], headers_only=True)
  ↓
retriever.get_instance_at_position(position)
  ├─ Fast path (if index_df): use PrimaryPosition column to select
  └─ Slow path (no index): fetch all headers, sort, select
  ↓
retriever._get_instance_headers() [if headers_only]
  or
retriever.get_instance_data() [if full data]
  ↓
Instance(uid, dataset) [cached in memory]
```

### 3. Rendering

```
get_images(positions=[...], contrast="lung")
  ↓
get_instances(...) [fetch]
  ↓
instance.get_image(contrast=..., image_width=...)
  ├─ normalize contrast
  ├─ MosaicGenerator.create_single_image()
  │  └─ apply window/level
  │  └─ PIL Image
  └─ PIL Image
```

### 4. Mosaicing

```
images = get_images(positions=[...])
  ↓
MosaicGenerator(tile_width=6, tile_height=6)
  ├─ tile_images(images)
  │  ├─ pad with black tiles
  │  ├─ standardize sizes
  │  └─ paste into grid
  └─ PIL Image (mosaic)
```

## Key Design Patterns

### 1. Lazy Initialization
- `DICOMRetriever` created only when needed
- Index cached, so no rebuild on each call

### 2. Caching
- **Index cache**: Parquet file (~KB, reused across runs)
- **Dataset cache**: Instance objects in memory (user-controlled lifetime)

### 3. Optional Index
- Retriever works with or without index_df
- With index: O(1) selection via PrimaryPosition
- Without index: O(n) fetch all headers + sort

### 4. Parallel Fetching
- `fetch_parallel()` in retriever
- ThreadPoolExecutor with configurable max_workers
- Order-preserving: returns results in input URL order

### 5. Composition
- `get_instances()` core method
- `get_instance()` wraps `get_instances()`
- `get_images()` wraps `get_instances()` + render
- `get_image()` wraps `get_instance()` + render
- Users compose MosaicGenerator + get_images for custom layouts

## External Dependencies

**Direct imports:**
- `pydicom` - DICOM parsing
- `obstore` - S3/HTTP/local storage
- `polars` - DataFrame operations
- `numpy` - array operations
- `PIL` - image creation
- `platformdirs` - cache directory resolution

**Module interdependencies:**
- `api` ← `retriever`, `index_cache`, `mosaic`, `__main__`
- `retriever` ← `slice_sorting`
- `index_cache` ← `retriever`, `header_capture`
- `header_capture` ← `retriever`, `slice_sorting`
- `mosaic` ← `contrast`
