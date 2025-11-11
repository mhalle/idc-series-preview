# Proposed API-First Architecture

## Executive Summary

This document proposes restructuring `dicom-series-preview` from a **CLI-first** architecture to an **API-first** architecture. Currently, the CLI drives everything and the Python API is secondary. The proposal inverts this: build a complete, standalone API that is also used by the CLI.

**Key benefit**: Users can use the same API whether they're writing Python code or using the command line.

---

## Current Problem

### CLI-First Architecture

```
CLI Handler (mosaic_command)
  ↓
Parse & validate args
  ↓
_parse_and_normalize_series()
  ↓
_initialize_retriever_with_cache()
  ↓
DICOMRetriever.get_instances_distributed()
  ↓
MosaicGenerator.create_mosaic()
  ↓
Save to disk
```

**Issues:**
- Every command handler repeats the same boilerplate (parse → validate → initialize → retrieve → generate → save)
- Helpers like `_parse_and_normalize_series()` are internal functions, not API
- Users wanting to use Python must construct all this logic themselves
- The low-level classes (DICOMRetriever, MosaicGenerator) exist but are not the focus
- Series parsing/resolution logic is scattered and not composable

### Example: Current CLI Code
```python
def mosaic_command(args, logger):
    result = _parse_and_normalize_series(args.seriesuid, args.root, logger)
    if result is None:
        return 1
    root_path, series_uid = result

    # ... validation ...

    retriever = _initialize_retriever_with_cache(root_path, series_uid, args, logger)
    instances = retriever.get_instances_distributed(
        series_uid, args.tile_width * tile_height,
        start=args.start, end=args.end
    )

    generator = MosaicGenerator(...)
    output_image = generator.create_mosaic(instances, retriever, series_uid)
    generator.save_image(output_image, args.output, quality=args.quality)
```

This 50+ line function could be 10 lines if the API was designed properly.

---

## Proposed Solution: API-First Architecture

### Inverted Dependency Graph

**From:**
```
CLI → Helpers → Low-level classes
```

**To:**
```
CLI ┐
    └→ API (SeriesIndex) → Low-level classes
    ↑
Python users
```

Both CLI and Python users use the same API layer.

---

## Three-Layer Architecture

```
┌─────────────────────────────────────────────────────┐
│   PRESENTATION LAYER (CLI)                          │
│   - Argument parsing                                │
│   - Formatting output                               │
│   - Error/validation messages                       │
│   - Thin handlers that call API                     │
└──────────────┬──────────────────────────────────────┘
               │ uses
┌──────────────▼──────────────────────────────────────┐
│   API LAYER (SeriesIndex + convenience functions)   │
│   - SeriesIndex: unified entry point                │
│   - Query methods: get_instance_uid()               │
│   - Generation methods: get_image(), get_mosaic()   │
│   - Convenience functions for one-off operations    │
│   - All orchestration logic                         │
└──────────────┬──────────────────────────────────────┘
               │ uses
┌──────────────▼──────────────────────────────────────┐
│   CORE LAYER (Technical implementations)            │
│   - DICOMRetriever: storage access (S3/HTTP/local)  │
│   - MosaicGenerator: image rendering                │
│   - ContrastPresets: windowing logic                │
│   - HeaderCapture: metadata extraction              │
│   - IndexCache: caching management                  │
│   - slice_sorting: spatial ordering                 │
└─────────────────────────────────────────────────────┘
```

### Design Principles

1. **Each layer depends only on layers below it** - No circular dependencies
2. **API is the focal point** - Both CLI and Python code use it
3. **Core classes remain unchanged** - Backwards compatible
4. **Explicit boundaries** - Clear what each layer does

---

## File Organization

### Current Structure
```
src/dicom_series_preview/
├── __main__.py           (100+ CLI handler functions)
├── retriever.py
├── mosaic.py
├── contrast.py
├── header_capture.py
├── index_cache.py
├── slice_sorting.py
└── __init__.py
```

### Proposed Structure
```
src/dicom_series_preview/
├── __main__.py           (Simple: argparse setup, dispatcher)
├── cli.py               (NEW: Command handlers using API)
├── api.py               (NEW/EXISTING: SeriesIndex, convenience functions)
├── core/                (OPTIONAL: Better organization)
│   ├── retriever.py
│   ├── mosaic.py
│   └── contrast.py
├── retriever.py         (Keep for compatibility)
├── mosaic.py
├── contrast.py
├── header_capture.py
├── index_cache.py
├── slice_sorting.py
└── __init__.py
```

---

## SeriesIndex: The Core API Class

`SeriesIndex` is the primary entry point. It:

1. **Encapsulates** series resolution and index loading
2. **Provides** query methods for the dataframe
3. **Offers** image generation methods
4. **Manages** the internal retriever and generator

### Responsibilities

**Initialization:**
- Parse series specification (UID, prefix, full path)
- Resolve against storage backend
- Load or generate index
- Cache for future use

**Properties:**
- `series_uid` - Normalized UID
- `root_path` - Storage root
- `instance_count` - Number of instances
- `primary_axis` - Which axis was used for sorting
- `position_range` - Min/max position values
- `df` - Raw Polars dataframe (for advanced users)

**Query Methods:**
- `get_instance_uid(position, slice_offset)` - Get single instance UID
- `get_instance_uids(count, start, end)` - Get N distributed instance UIDs

**Generation Methods:**
- `get_image(position, contrast, image_width)` - Single image
- `get_images(count, start, end, contrast)` - Multiple images
- `get_mosaic(width, height, start, end, contrast)` - Mosaic grid
- `get_contrast_grid(contrasts, ...)` - Contrast comparison

**Advanced Access:**
- `.retriever` - Low-level DICOMRetriever (for power users)

### Example API Usage

```python
from dicom_series_preview import SeriesIndex

# Initialize (handles all the complex stuff)
index = SeriesIndex("38902e14-b11f-4548-910e-771ee757dc82")

# Query
print(index.instance_count)  # 305
instance_uid = index.get_instance_uid(position=0.5)

# Generate images
img = index.get_image(position=0.5, contrast="lung")
mosaic = index.get_mosaic(width=6, height=6)
grid = index.get_contrast_grid(
    position=0.5,
    contrasts=["lung", "bone", "brain"]
)
```

---

## CLI Refactoring

### Before (Current)

```python
def mosaic_command(args, logger):
    """Generate a tiled mosaic from a DICOM series."""
    try:
        # Parse and normalize
        result = _parse_and_normalize_series(args.seriesuid, args.root, logger)
        if result is None:
            return 1
        root_path, series_uid = result

        # Validate
        if not _validate_output_format(Path(args.output)):
            logger.error("Invalid format")
            return 1
        if not (0.0 <= args.start <= 1.0):
            logger.error("Invalid range")
            return 1
        # ... more validation ...

        # Initialize
        retriever = _initialize_retriever_with_cache(...)

        # Retrieve
        instances = retriever.get_instances_distributed(...)
        if not instances:
            logger.error("No instances found")
            return 1

        # Generate
        generator = MosaicGenerator(...)
        output_image = generator.create_mosaic(...)
        if not output_image:
            logger.error("Failed to generate")
            return 1

        # Save
        generator.save_image(output_image, args.output, quality=args.quality)

        return 0
    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1
```

**Lines of code: ~50**
**Concerns: 7 (parsing, validation, initialization, retrieval, generation, saving, error handling)**
**Readability: Low (lot of boilerplate)**

### After (API-First)

```python
def mosaic_command(args, logger):
    """Generate a tiled mosaic from a DICOM series."""
    try:
        # Validate CLI-specific concerns
        if not _validate_output_format(Path(args.output)):
            logger.error("Output must be .webp or .jpg/.jpeg")
            return 1
        if not _validate_ranges(args.start, args.end):
            logger.error("Range must be 0.0-1.0")
            return 1

        # Use API (series handling is now hidden)
        index = SeriesIndex(
            args.seriesuid,
            root=args.root,
            cache_dir=args.cache_dir
        )

        mosaic = index.get_mosaic(
            width=args.tile_width,
            height=args.tile_height or args.tile_width,
            start=args.start,
            end=args.end,
            contrast=args.contrast or "auto",
            image_width=args.image_width
        )

        if not mosaic:
            logger.error("Failed to generate mosaic")
            return 1

        # Save (same as before)
        from .core import MosaicGenerator
        MosaicGenerator().save_image(mosaic, args.output, quality=args.quality)

        return 0
    except ValueError as e:
        logger.error(f"Invalid series: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1
```

**Lines of code: ~35 (and much clearer)**
**Concerns: 3 (validation, API call, saving)**
**Readability: High (intent is clear)**

---

## Refactoring Strategy

### Phase 1: Complete SeriesIndex (Current Status)

- [x] Base SeriesIndex class with:
  - [x] Series initialization and resolution
  - [x] Index loading/caching
  - [x] Properties (instance_count, primary_axis, etc.)
- [ ] Add dataframe access (`.df` property)
- [ ] Add query methods (get_instance_uid, get_instance_uids)

### Phase 2: Add Generation Methods to SeriesIndex

- [ ] `get_image()` - Single image extraction
- [ ] `get_images()` - Multiple distributed images
- [ ] `get_mosaic()` - Mosaic grid
- [ ] `get_contrast_grid()` - Contrast comparison grid

### Phase 3: Refactor CLI to Use SeriesIndex

- [ ] Move CLI handlers from `__main__.py` to `cli.py`
- [ ] Rewrite handlers to use SeriesIndex API
- [ ] Keep `__main__.py` minimal (just argparse + dispatch)
- [ ] Test that all commands still work

### Phase 4: Add Convenience Functions

- [ ] High-level functions that wrap SeriesIndex:
  - [ ] `get_image(series, position, ...)`
  - [ ] `get_mosaic(series, width, ...)`
  - [ ] `get_contrast_grid(series, ...)`
- [ ] Export from `__init__.py`

### Phase 5: Documentation & Testing

- [ ] API reference documentation
- [ ] Tutorial with examples
- [ ] Cookbook with common patterns
- [ ] Update CLI docs to reflect structure
- [ ] Test coverage for API layer

---

## Benefits of This Approach

### For Users

1. **One API for everyone**: Use the same interface from CLI or Python
2. **Simpler to get started**: One class, straightforward methods
3. **More flexible**: Can combine operations programmatically
4. **Better discoverability**: IDE autocomplete works

### For Developers

1. **Cleaner codebase**: Clear separation of concerns
2. **Easier to test**: Each layer tested independently
3. **Easier to maintain**: Changes to core don't affect CLI
4. **Easier to extend**: Add new operations in one place
5. **Backwards compatible**: Old code still works

### For Architecture

1. **Composable**: Operations can be combined
2. **Single responsibility**: Each layer has clear job
3. **No circular dependencies**: Clean dependency graph
4. **Testable**: No integration required for unit tests

---

## Backwards Compatibility

All existing classes remain public and unchanged:

- `DICOMRetriever` - Still works exactly as before
- `MosaicGenerator` - Still works exactly as before
- `ContrastPresets` - Still works exactly as before
- `HeaderCapture` - Still works exactly as before
- `IndexCache` - Still works exactly as before

Advanced users who want low-level control can continue using these directly. `SeriesIndex` is the **recommended** path, but not required.

---

## What Changes and What Doesn't

### Doesn't Change
- Core logic (retriever, generator, contrast, etc.)
- DICOM file handling
- Image rendering
- Caching mechanism
- Index format
- CLI commands and options

### Does Change
- Where `__main__.py` lives (refactored from 1300+ lines to ~50)
- How series are initialized (via SeriesIndex instead of scattered helpers)
- How users integrate the library (via SeriesIndex instead of raw classes)
- File organization (cli.py separated from api.py)

---

## Example: Using the Refactored API

### Jupyter Notebook

```python
from dicom_series_preview import SeriesIndex

# Get index (automatic caching)
index = SeriesIndex("38902e14-b11f-4548-910e-771ee757dc82")

# Quick stats
print(f"Series has {index.instance_count} instances")
print(f"Primary axis: {index.primary_axis}")

# Get images
superior = index.get_image(position=0.0, contrast="lung")
middle = index.get_image(position=0.5, contrast="lung")
inferior = index.get_image(position=1.0, contrast="lung")

# Get mosaic
mosaic = index.get_mosaic(width=8, height=6, contrast="lung")

# Compare contrasts
comparison = index.get_contrast_grid(
    position=0.5,
    contrasts=["lung", "bone", "brain", "soft"]
)

# Access raw dataframe for analysis
df = index.df
print(df.select(["Index", "InstanceNumber", "PrimaryPosition"]))
```

### Python Script

```python
from dicom_series_preview import SeriesIndex

def process_series(series_uid):
    """Process a series and generate outputs."""
    try:
        index = SeriesIndex(series_uid)

        # Generate for different contrasts
        for contrast in ["lung", "bone", "brain"]:
            mosaic = index.get_mosaic(width=6, height=6, contrast=contrast)
            mosaic.save(f"output_{contrast}.webp")

        print(f"✓ Processed {series_uid}")
        return True
    except ValueError as e:
        print(f"✗ Failed: {e}")
        return False

# Use it
process_series("38902e14-b11f-4548-910e-771ee757dc82")
```

### CLI (Still Works)

```bash
# These all still work, but handlers are simpler now
dicom-series-preview mosaic SERIES output.webp --tile-width 8 --contrast lung
dicom-series-preview get-image SERIES output.webp --position 0.5
dicom-series-preview contrast-mosaic SERIES output.webp --position 0.5 \
  --contrast lung --contrast bone --contrast brain
```

---

## Open Questions & Decisions

1. **Where should image generation live?**
   - Current proposal: In SeriesIndex methods
   - Alternative: Separate `ImageGenerator` class
   - Decision: SeriesIndex methods (cleaner API surface)

2. **Should SeriesIndex expose the retriever?**
   - Current proposal: Yes, as `.retriever` property
   - Alternative: No, hide it completely
   - Decision: Yes (power users need it, backwards compatible)

3. **What about filtering/subsetting?**
   - Should `index.filter_by_position(start, end)` return a new SeriesIndex?
   - Current proposal: Yes
   - This enables: `filtered = index.filter_by_position(0.3, 0.7); img = filtered.get_image()`

4. **How to handle errors?**
   - Should SeriesIndex raise exceptions or return None?
   - Current proposal: Raise exceptions (fails fast, explicit)
   - SeriesIndex.__init__ raises ValueError on failure
   - Methods like get_image() also raise on error

---

## Migration Path

### For CLI Users
- No changes needed! CLI continues to work exactly the same.

### For Library Users (Python API)
- **Current** (still works):
  ```python
  from dicom_series_preview import DICOMRetriever, IndexCache
  index_df = IndexCache.load_or_generate_index(series_uid, root_path)
  retriever = DICOMRetriever(root_path, index_df=index_df)
  ```

- **Recommended** (new):
  ```python
  from dicom_series_preview import SeriesIndex
  index = SeriesIndex(series_uid, root=root_path)
  ```

---

## Testing Strategy

### Unit Tests
- Test SeriesIndex initialization with various input formats
- Test query methods
- Test generation methods (mocking lower layers)

### Integration Tests
- Test with real S3 data
- Test with real HTTP server
- Test with local files
- Test caching behavior

### CLI Tests
- Test each command handler
- Test error cases
- Test with various argument combinations

---

## Timeline Estimate

- Phase 1 (Complete SeriesIndex): ✅ Done
- Phase 2 (Generation methods): 2-3 hours
- Phase 3 (CLI refactor): 2-3 hours
- Phase 4 (Convenience functions): 1 hour
- Phase 5 (Docs & tests): 2-3 hours

**Total**: ~8-10 hours of implementation

---

## Conclusion

The proposed API-first architecture:

1. **Solves the core problem**: API is no longer secondary, it's the foundation
2. **Benefits everyone**: CLI users, library users, and developers
3. **Maintains compatibility**: Existing code still works
4. **Sets up for growth**: Easy to add features in the future
5. **Improves code quality**: Clear layers, testable, maintainable

The key insight is that **SeriesIndex becomes the single entry point** for all operations on a DICOM series, whether through CLI or programmatically.
