# SeriesIndex API Guide

Complete reference for the `SeriesIndex` class and composable patterns for working with DICOM series.

## Table of Contents

1. [Basic Concepts](#basic-concepts)
2. [Core Methods](#core-methods)
3. [Instance Methods](#instance-methods)
4. [Contrast Specifications](#contrast-specifications)
5. [Composable Patterns](#composable-patterns)
6. [Examples](#examples)

## Basic Concepts

### SeriesIndex

Entry point for accessing a DICOM series. Handles instance retrieval, caching, and image rendering.

```python
from dicom_series_preview import SeriesIndex

# Initialize with series UID
index = SeriesIndex("38902e14-b11f-4548-910e-771ee757dc82")
print(index.instance_count)  # 181
print(index.primary_axis)     # 'Z'
```

### Instance

Wraps a DICOM dataset with caching and rendering methods. Datasets are cached in memory for efficient multi-use.

```python
instance = index.get_instance(position=0.5)
print(instance.instance_uid)
print(instance.dataset)  # pydicom.Dataset, cached in memory
```

### Contrast

Specifies how to apply window/level to DICOM pixel data. Supports presets, auto-detection, or custom values.

```python
from dicom_series_preview import Contrast

# Multiple ways to specify contrast
c1 = Contrast("lung")                          # Preset
c2 = Contrast("auto")                          # Auto-detect from statistics
c3 = Contrast("embedded")                      # DICOM file values or auto
c4 = Contrast(window_width=1500, window_center=500)  # Custom values
```

---

## Core Methods

### `get_instances()`

**Core method for fetching instances.** Handles parallel fetching, deduplication, and optional headers-only mode. Other methods build on this.

```python
# Fetch by positions (normalized 0.0-1.0)
instances = index.get_instances(positions=[0.2, 0.5, 0.8])

# Fetch by slice numbers (direct indexing)
instances = index.get_instances(slice_numbers=[0, 50, 100])

# Optional: remove duplicates when you don't know series size
positions = [i/99 for i in range(100)]
instances = index.get_instances(positions=positions, remove_duplicates=True)

# Optional: headers-only mode for fast metadata access
instances = index.get_instances(positions=[0.5], headers_only=True)
# Dataset has metadata but pixel data fetched on-demand when rendering

# Customize parallel fetching
instances = index.get_instances(positions=[...], max_workers=4)
```

**Parameters:**
- `positions`: list of float (0.0-1.0), mutually exclusive with `slice_numbers`
- `slice_numbers`: list of int, mutually exclusive with `positions`
- `max_workers`: int, default 8 - parallel fetch threads
- `remove_duplicates`: bool, default False - remove duplicate requests
- `headers_only`: bool, default False - fetch headers only for fast metadata access

**Returns:** list of `Instance` objects in requested order

---

### `get_instance()`

**Convenience wrapper** for single instance retrieval. Calls `get_instances()` internally.

```python
instance = index.get_instance(position=0.5)
instance = index.get_instance(slice_number=50)
```

**Parameters:**
- `position`: float (0.0-1.0), mutually exclusive with `slice_number`
- `slice_number`: int, mutually exclusive with `position`

**Returns:** single `Instance` object

---

### `get_images()`

**Render multiple images in parallel.** Fetches instances and renders them as PIL Images.

```python
# Render at specific positions
images = index.get_images(positions=[0.2, 0.5, 0.8])

# Render at specific slices
images = index.get_images(slice_numbers=[0, 50, 100])

# Apply contrast to all tiles
images = index.get_images(positions=[...], contrast="lung")

# Customize tile size
images = index.get_images(positions=[...], image_width=256)

# Remove duplicates when rendering
images = index.get_images(positions=[...], remove_duplicates=True)

# Control parallel rendering
images = index.get_images(positions=[...], max_workers=4)
```

**Parameters:**
- `positions`: list of float, mutually exclusive with `slice_numbers`
- `slice_numbers`: list of int, mutually exclusive with `positions`
- `contrast`: str, `Contrast` object, or None - applied to all images
- `image_width`: int, default 128 - width in pixels
- `max_workers`: int, default 8 - parallel render threads
- `remove_duplicates`: bool, default False

**Returns:** list of `PIL.Image.Image` objects

---

### `get_image()`

**Convenience wrapper** for single image rendering. Calls `get_instance()` then renders.

```python
image = index.get_image(position=0.5, contrast="bone", image_width=256)
image = index.get_image(slice_number=50)
```

**Parameters:**
- `position`: float, mutually exclusive with `slice_number`
- `slice_number`: int, mutually exclusive with `position`
- `contrast`: str, `Contrast` object, or None
- `image_width`: int, default 128

**Returns:** single `PIL.Image.Image`

---

## Instance Methods

### `get_image()`

Render instance as image with specified contrast and size.

```python
instance = index.get_instance(position=0.5)

image = instance.get_image(contrast="lung", image_width=128)
image = instance.get_image(contrast=Contrast("auto"))
```

Dataset is cached, so rendering with different contrasts reuses the same data.

---

### `get_pixel_array()`

Get raw pixel array with DICOM RescaleSlope/Intercept applied.

```python
instance = index.get_instance(position=0.5)

pixel_array = instance.get_pixel_array()  # numpy.ndarray
print(pixel_array.shape)
print(pixel_array.dtype)
```

---

### `get_contrast_grid()`

Render instance with multiple contrasts in a horizontal grid.

```python
instance = index.get_instance(position=0.5)

grid = instance.get_contrast_grid(
    contrasts=["lung", "bone", "brain"],
    image_width=128
)
# Returns PIL Image: 3 contrasts in a row
```

---

## Contrast Specifications

### Preset Names

Common presets available:
- `"lung"` - Lung window (1500/500)
- `"bone"` - Bone window (2500/500)
- `"brain"` - Brain window (80/40)
- `"abdomen"` - Abdomen window (400/40)
- `"mediastinum"` - Mediastinum window (350/50)

### Auto-Detection

```python
# Detect from pixel statistics
images = index.get_images(positions=[...], contrast="auto")

# Use DICOM file's window/level, fall back to auto if not present
images = index.get_images(positions=[...], contrast="embedded")
```

### Custom Window/Level

```python
# Via Contrast object
contrast = Contrast(window_width=1500, window_center=500)
images = index.get_images(positions=[...], contrast=contrast)

# Via string (window_width/window_center)
images = index.get_images(positions=[...], contrast="1500/500")
```

---

## Composable Patterns

The API is designed for composition. Build higher-level operations from primitives.

### Pattern 1: Mosaic Grid

Tile multiple rendered images into a grid.

```python
from dicom_series_preview import MosaicGenerator

# Get images
images = index.get_images(positions=[i/35 for i in range(36)], contrast="lung")

# Tile into 6x6 grid
mosaic = MosaicGenerator(tile_width=6, tile_height=6).tile_images(images)
mosaic.save("output.png")
```

### Pattern 2: Multi-Contrast Mosaic

Mosaic showing multiple contrasts for multiple instances. Each row is an instance, each column is a contrast.

```python
from dicom_series_preview import Contrast, MosaicGenerator

# Fetch instances
instances = index.get_instances(positions=[0.2, 0.5, 0.8])

# Define contrasts
contrasts = [Contrast("lung"), Contrast("bone"), Contrast("brain")]

# Render each instance with each contrast
images = []
for instance in instances:
    for contrast in contrasts:
        img = instance.get_image(contrast=contrast, image_width=128)
        images.append(img)

# Tile into 3 columns × 3 rows (3 instances, 3 contrasts each)
mosaic = MosaicGenerator(
    tile_width=len(contrasts),  # 3 columns
    tile_height=len(instances)  # 3 rows
).tile_images(images)
mosaic.save("multi_contrast_mosaic.png")
```

### Pattern 3: Headers-Only Then On-Demand Rendering

Fetch metadata fast, render only what you need.

```python
# Fast metadata fetch (5KB per instance instead of full DICOM)
instances = index.get_instances(
    positions=[i/99 for i in range(100)],
    headers_only=True
)

# Browse metadata
for inst in instances[:5]:
    print(inst.dataset.PatientName)
    print(inst.dataset.SeriesDescription)

# Render only interesting ones
image_of_interest = instances[42].get_image(contrast="lung")
```

### Pattern 4: Safe Limits with Deduplication

When you don't know series size, use `remove_duplicates` as a safety valve.

```python
# User provides 100 evenly-spaced positions
# But series only has 50 instances
positions = [i / 99 for i in range(100)]

# Without deduplication: may render same instance twice
images = index.get_images(positions=positions, remove_duplicates=False)
print(len(images))  # Could be up to 100

# With deduplication: removes duplicates
images = index.get_images(positions=positions, remove_duplicates=True)
print(len(images))  # At most 50 unique
```

### Pattern 5: Pixel Array Access

Get raw pixel data for custom processing.

```python
instances = index.get_instances(positions=[0.2, 0.5, 0.8])

for instance in instances:
    pixels = instance.get_pixel_array()

    # Custom processing
    mean_intensity = pixels.mean()
    min_val, max_val = pixels.min(), pixels.max()

    # Or pass to ML model
    prediction = model.predict(pixels)
```

### Pattern 6: Efficient Batch Processing

Fetch instances once, render multiple ways without re-fetching.

```python
# Fetch all instances (once)
instances = index.get_instances(positions=[i/9 for i in range(10)])

# Render with multiple contrasts (dataset already cached)
lung_images = [inst.get_image(contrast="lung") for inst in instances]
bone_images = [inst.get_image(contrast="bone") for inst in instances]

# Get pixel data (uses cached dataset)
pixel_arrays = [inst.get_pixel_array() for inst in instances]

# All three operations reuse the same cached datasets!
```

---

## Examples

### Example 1: View Series Overview

```python
from dicom_series_preview import SeriesIndex

# Create index
index = SeriesIndex("38902e14-b11f-4548-910e-771ee757dc82")

# Get overview info
print(f"Series: {index.series_uid}")
print(f"Instances: {index.instance_count}")
print(f"Primary axis: {index.primary_axis}")
min_pos, max_pos = index.position_range
print(f"Position range: {min_pos} to {max_pos}")

# Get single image at middle
middle_img = index.get_image(position=0.5, contrast="lung")
middle_img.save("middle.png")
```

### Example 2: Create Diagnostic Grid

```python
from dicom_series_preview import SeriesIndex, MosaicGenerator

index = SeriesIndex("38902e14-b11f-4548-910e-771ee757dc82")

# Sample 36 evenly-spaced slices
positions = [i / 35 for i in range(36)]
images = index.get_images(positions=positions, contrast="auto")

# Arrange in 6x6 grid
mosaic = MosaicGenerator(tile_width=6, tile_height=6).tile_images(images)
mosaic.save("diagnostic_overview.png")
```

### Example 3: Multi-Contrast Analysis

```python
from dicom_series_preview import SeriesIndex, Contrast, MosaicGenerator

index = SeriesIndex("38902e14-b11f-4548-910e-771ee757dc82")

# Get 3 key instances
instances = index.get_instances(slice_numbers=[50, 90, 130])

# Show all contrasts for each instance
contrasts = ["lung", "bone", "mediastinum"]
images = [
    inst.get_image(contrast=c, image_width=128)
    for inst in instances
    for c in contrasts
]

# 3 rows (instances) × 3 columns (contrasts)
mosaic = MosaicGenerator(tile_width=3, tile_height=3).tile_images(images)
mosaic.save("multi_contrast_analysis.png")
```

### Example 4: Efficient Metadata Browse

```python
from dicom_series_preview import SeriesIndex

index = SeriesIndex("38902e14-b11f-4548-910e-771ee757dc82")

# Fast metadata-only fetch
instances = index.get_instances(
    positions=[i / 99 for i in range(100)],
    headers_only=True
)

# Browse metadata without loading full DICOM files
for i, inst in enumerate(instances[:10]):
    print(f"{i}: {inst.dataset.PatientName} - {inst.dataset.SeriesDescription}")

# Render only what interests you
interesting_inst = instances[5]
image = interesting_inst.get_image(contrast="auto")  # Fetches full data on demand
image.save("interesting.png")
```

### Example 5: Custom Pixel Processing

```python
from dicom_series_preview import SeriesIndex
import numpy as np

index = SeriesIndex("38902e14-b11f-4548-910e-771ee757dc82")

# Get instances
instances = index.get_instances(positions=[0.25, 0.5, 0.75])

# Process pixel data
for i, instance in enumerate(instances):
    pixels = instance.get_pixel_array()

    # Custom analysis
    histogram, _ = np.histogram(pixels, bins=256)
    mean = pixels.mean()
    std = pixels.std()

    print(f"Instance {i}: mean={mean:.1f}, std={std:.1f}")

    # Could pass to ML model, signal processing, etc.
```

---

## Design Philosophy

The API follows these principles:

1. **Composability**: Build complex operations from simple primitives
2. **Efficiency**: Reuse fetched and rendered data through Instance caching
3. **Flexibility**: Users control contrast, tile size, grid layout, etc.
4. **Simplicity**: Minimal API surface - only core methods, no convenience wrappers that hide composition
5. **Parallelism**: Concurrent fetching and rendering where it matters

This means:
- No `get_mosaic()` - compose `get_images()` + `MosaicGenerator`
- No `get_multi_contrast_grid()` - loop `get_image()` with different contrasts
- No `get_contrast_mosaic()` - compose Instance methods with tiling

Users get maximum flexibility by understanding the building blocks.
