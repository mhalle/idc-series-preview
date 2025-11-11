# Refactoring Proposal: Centralize Instance Fetching

## Current State
- `get_instance()`: Single instance fetch
- `get_instances()` doesn't exist yet
- `get_images()`: Parallel image rendering with deduplication logic
- `get_image()`: Single image rendering
- Deduplication and parallelization code lives in `get_images()`

## Problem
Deduplication and parallel fetching logic is duplicated across methods. Adding similar functionality to instance fetching would repeat this pattern again. Need a single source of truth for these concerns.

## Proposed Solution: Add `get_instances()` as Core Method

### New `get_instances()` Method
```python
def get_instances(
    self,
    positions: Optional[list[float]] = None,
    slice_numbers: Optional[list[int]] = None,
    max_workers: int = 8,
    remove_duplicates: bool = False,
    headers_only: bool = False,
) -> list[Instance]:
    """
    Fetch multiple instances in parallel.

    Parameters
    ----------
    positions : list of float, optional
        Normalized positions (0.0-1.0) for instances to fetch.
        Mutually exclusive with slice_numbers.

    slice_numbers : list of int, optional
        Zero-indexed slice numbers for instances to fetch.
        Mutually exclusive with positions.

    max_workers : int, default 8
        Maximum number of parallel fetch threads

    remove_duplicates : bool, default False
        If True, removes duplicate positions/slice_numbers, keeping only
        the first occurrence. Useful when you don't know the series size.

    headers_only : bool, default False
        If True, fetches only DICOM headers (fast, lightweight metadata).
        Instance.get_image() will still work, fetching full data when needed.

    Returns
    -------
    list of Instance
        Instance objects in original order, with datasets cached in memory.
    """
```

**Responsibilities:**
1. Deduplication (if `remove_duplicates=True`)
2. Parallel instance fetching using `ThreadPoolExecutor`
   - Reuses retriever's `_get_instance_headers()` if `headers_only=True`
   - Reuses retriever's `get_instance_data()` if `headers_only=False`
3. Returns list of `Instance` objects in original order
4. Preserves first-occurrence order when duplicates are removed

### Refactor Other Methods as Thin Wrappers

**`get_instance(position=..., slice_number=...)`**
```python
def get_instance(self, position=None, slice_number=None) -> Instance:
    """Wrapper: fetch single instance."""
    instances = self.get_instances(
        positions=[position] if position is not None else None,
        slice_numbers=[slice_number] if slice_number is not None else None,
    )
    return instances[0]
```

**`get_images(positions=..., slice_numbers=..., contrast=..., ...)`**
```python
def get_images(self, positions=None, slice_numbers=None, contrast=None,
               image_width=128, max_workers=8, remove_duplicates=False) -> list[Image.Image]:
    """Wrapper: fetch instances, render images."""
    instances = self.get_instances(
        positions=positions,
        slice_numbers=slice_numbers,
        max_workers=max_workers,
        remove_duplicates=remove_duplicates,
        headers_only=False,  # Need full data to render
    )
    return [inst.get_image(contrast=contrast, image_width=image_width) for inst in instances]
```

**`get_image(position=..., slice_number=..., contrast=..., ...)`**
```python
def get_image(self, position=None, slice_number=None, contrast=None, image_width=128) -> Image.Image:
    """Wrapper: fetch single instance, render image."""
    instance = self.get_instance(position=position, slice_number=slice_number)
    return instance.get_image(contrast=contrast, image_width=image_width)
```

## Benefits

1. **DRY Principle**: Deduplication and parallelization logic in one place
2. **Flexibility**: `headers_only` flag enables lightweight metadata-only access
3. **Composability**: Users can fetch instances once and render in multiple ways
4. **Consistency**: All methods follow singular/plural pattern with wrappers
5. **Performance**: Instance caching happens naturally - fetched once, used multiple ways
6. **Testability**: Core logic in `get_instances()` is easier to test

## Example Usage

### Lightweight metadata fetch
```python
# Get headers only, very fast
instances = index.get_instances(positions=[0.2, 0.5, 0.8], headers_only=True)
print(instances[0].dataset.PatientName)  # Works with headers

# Full data fetched on-demand when rendering
img = instances[0].get_image()
```

### Render same instances multiple ways
```python
instances = index.get_instances(positions=[0.2, 0.5, 0.8])

# Render with different contrasts - reuses cached dataset
images_lung = [inst.get_image(contrast='lung') for inst in instances]
images_bone = [inst.get_image(contrast='bone') for inst in instances]

# Get pixel arrays - reuses cached dataset
pixel_arrays = [inst.get_pixel_array() for inst in instances]
```

### Deduplication for safe limits
```python
# User doesn't know series size, uses safe limit with deduplication
positions = [i/99 for i in range(100)]  # 100 positions
instances = index.get_instances(positions=positions, remove_duplicates=True)
# Returns only unique positions actually in series, no duplicates rendered
```

## Implementation Notes

- `get_instances()` is the core method - should be well-tested with full coverage
- Wrappers are thin and mostly trivial - easier to maintain
- `headers_only` mode reuses existing `retriever._get_instance_headers()` logic
- Full data mode reuses existing `retriever.get_instance_data()` logic
- Instance caching is automatic - users keep references to instances to maintain cache

## Testing Strategy

1. Test `get_instances()` thoroughly:
   - Parallel fetching with various worker counts
   - Deduplication with duplicates, mixed, and no duplicates
   - `headers_only=True` mode
   - `headers_only=False` mode
   - Order preservation

2. Test wrappers:
   - `get_instance()` returns first of `get_instances([...])`
   - `get_images()` renders instances from `get_instances()`
   - `get_image()` is just convenience wrapper

3. Integration:
   - Instance caching works correctly
   - Multiple renders of same instance reuse dataset
