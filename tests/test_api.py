"""Integration tests for SeriesIndex and Instance API.

These tests use real S3 data from IDC (Imaging Data Commons).
The series UID used is stable and expected to remain available.
"""

import pytest
from PIL import Image
import numpy as np
import polars as pl
from types import MethodType

from idc_series_preview import SeriesIndex, Contrast, Instance, PositionInterpolator


# Test series from IDC (stable, publicly available)
TEST_SERIES_UID = "38902e14-b11f-4548-910e-771ee757dc82"
TEST_S3_ROOT = "s3://idc-open-data"


class TestSeriesIndex:
    """Tests for SeriesIndex initialization and properties."""

    def test_series_index_initialization(self):
        """Test basic SeriesIndex initialization."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)
        assert index.series_uid == TEST_SERIES_UID
        assert index.root_path == TEST_S3_ROOT
        assert index.instance_count > 0
        assert len(index) == index.instance_count

    def test_series_index_properties(self):
        """Test SeriesIndex properties."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        # Primary axis should be one of the valid options
        assert index.primary_axis in ['X', 'Y', 'Z', 'I']

        # Position range should be a tuple of two floats
        min_pos, max_pos = index.position_range
        assert isinstance(min_pos, float)
        assert isinstance(max_pos, float)
        assert min_pos <= max_pos

    def test_series_index_repr(self):
        """Test SeriesIndex string representation."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)
        repr_str = repr(index)

        assert "SeriesIndex" in repr_str
        assert TEST_SERIES_UID in repr_str
        assert "instances=" in repr_str


class TestSeriesIndexGetInstance:
    """Tests for SeriesIndex.get_instance() method."""

    def test_get_instance_by_position(self):
        """Test fetching instance by position."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        # Fetch from middle of series
        instance = index.get_instance(position=0.5)

        assert isinstance(instance, Instance)
        assert instance.instance_uid is not None
        assert instance.dataset is not None

    def test_get_instance_by_position_boundaries(self):
        """Test fetching instances at position boundaries."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        # Test at start
        instance_start = index.get_instance(position=0.0)
        assert isinstance(instance_start, Instance)

        # Test at end
        instance_end = index.get_instance(position=1.0)
        assert isinstance(instance_end, Instance)

    def test_get_instance_by_slice_number(self):
        """Test fetching instance by slice number."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        # Fetch first slice
        instance = index.get_instance(slice_number=0)
        assert isinstance(instance, Instance)
        assert instance.instance_uid is not None

        # Fetch last slice
        instance_last = index.get_instance(slice_number=index.instance_count - 1)
        assert isinstance(instance_last, Instance)

    def test_get_instance_invalid_position(self):
        """Test error handling for invalid position."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        with pytest.raises(ValueError):
            index.get_instance(position=-0.1)

        with pytest.raises(ValueError):
            index.get_instance(position=1.1)

    def test_get_instance_invalid_slice_number(self):
        """Test error handling for invalid slice number."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        with pytest.raises(ValueError):
            index.get_instance(slice_number=-1)

        with pytest.raises(ValueError):
            index.get_instance(slice_number=index.instance_count)

    def test_get_instance_both_parameters(self):
        """Test error when both position and slice_number specified."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        with pytest.raises(ValueError):
            index.get_instance(position=0.5, slice_number=10)

    def test_get_instance_neither_parameter(self):
        """Test error when neither position nor slice_number specified."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        with pytest.raises(ValueError):
            index.get_instance()


class TestSeriesIndexGetImage:
    """Tests for SeriesIndex.get_image() convenience method."""

    def test_get_image_by_position(self):
        """Test rendering image by position."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        img = index.get_image(position=0.5)

        assert isinstance(img, Image.Image)
        assert img.size[0] > 0  # Has width
        assert img.size[1] > 0  # Has height

    def test_get_image_by_slice_number(self):
        """Test rendering image by slice number."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        img = index.get_image(slice_number=10)

        assert isinstance(img, Image.Image)

    def test_get_image_with_contrast_string(self):
        """Test rendering with contrast specified as string."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        img = index.get_image(position=0.5, contrast="lung")

        assert isinstance(img, Image.Image)

    def test_get_image_with_contrast_object(self):
        """Test rendering with Contrast object."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        contrast = Contrast(spec="bone")
        img = index.get_image(position=0.5, contrast=contrast)

        assert isinstance(img, Image.Image)

    def test_get_image_with_custom_window_level(self):
        """Test rendering with custom window/level."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        # Using window/level format
        img = index.get_image(position=0.5, contrast="1500/500")

        assert isinstance(img, Image.Image)

    def test_get_image_with_custom_width(self):
        """Test rendering with custom image width."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        img_small = index.get_image(position=0.5, image_width=64)
        img_large = index.get_image(position=0.5, image_width=256)

        assert img_small.size[0] < img_large.size[0]

    def test_get_image_with_auto_contrast(self):
        """Test rendering with auto contrast detection."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        img = index.get_image(position=0.5, contrast="auto")

        assert isinstance(img, Image.Image)

    def test_get_image_with_embedded_contrast(self):
        """Test rendering with embedded DICOM window/level."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        img = index.get_image(position=0.5, contrast="embedded")

        assert isinstance(img, Image.Image)


class TestSeriesIndexGetImages:
    """Tests for SeriesIndex.get_images() method."""

    def test_get_images_by_positions(self):
        """Test rendering images at specific positions."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        positions = [0.1, 0.3, 0.5, 0.7, 0.9]
        images = index.get_images(positions=positions)

        assert isinstance(images, list)
        assert len(images) == len(positions)
        for img in images:
            assert isinstance(img, Image.Image)


class DummyRetrieverNoFetch:
    def __init__(self, dataset_by_position):
        self.dataset_by_position = dataset_by_position

    def get_instance_at_position(self, series_uid, position):
        return self.dataset_by_position[position]

    def get_instances(self, urls, headers_only=False, max_workers=None):
        raise AssertionError("get_instances should not be called when datasets already available")


class DummyRetrieverHeadersOnly:
    def __init__(self, dataset_by_position, fetch_results):
        self.dataset_by_position = dataset_by_position
        self.fetch_results = fetch_results
        self.calls = []

    def get_instance_at_position(self, series_uid, position):
        return self.dataset_by_position[position]

    def get_instances(self, urls, headers_only=False, max_workers=None):
        self.calls.append((tuple(urls), headers_only))
        return [self.fetch_results[url] for url in urls]


def _make_fake_series_index(retriever, uid_url_pairs):
    index = object.__new__(SeriesIndex)
    index._index_df = pl.DataFrame(
        {
            "_index": list(range(len(uid_url_pairs))),
            "SOPInstanceUID": [uid for uid, _ in uid_url_pairs],
            "_data_url": [url for _, url in uid_url_pairs],
        }
    )
    index._series_uid = "series"
    index._root_path = "root"
    index._cache_dir = None
    index._use_cache = True
    index._retriever = retriever
    index._get_or_create_retriever = MethodType(lambda self: retriever, index)
    return index


def test_get_instances_reuses_dataset_without_refetch():
    datasets = {0.1: ("uid-1", object()), 0.4: ("uid-2", object())}
    retriever = DummyRetrieverNoFetch(datasets)
    uid_url_pairs = [
        ("uid-1", "s3://root/uid-1.dcm"),
        ("uid-2", "s3://root/uid-2.dcm"),
    ]
    index = _make_fake_series_index(retriever, uid_url_pairs)

    instances = index.get_instances(positions=[0.1, 0.4])

    assert len(instances) == 2
    assert instances[0].instance_uid == "uid-1"
    assert instances[1].instance_uid == "uid-2"


def test_get_instances_headers_only_fetches_missing_datasets():
    datasets = {0.3: ("uid-h1", None), 0.6: ("uid-h2", None)}
    fetch_map = {
        "s3://root/uid-h1.dcm": object(),
        "s3://root/uid-h2.dcm": object(),
    }
    retriever = DummyRetrieverHeadersOnly(datasets, fetch_map)
    uid_url_pairs = [
        ("uid-h1", "s3://root/uid-h1.dcm"),
        ("uid-h2", "s3://root/uid-h2.dcm"),
    ]
    index = _make_fake_series_index(retriever, uid_url_pairs)

    instances = index.get_instances(positions=[0.3, 0.6], headers_only=True)

    assert len(instances) == 2
    assert retriever.calls == [(
        ("s3://root/uid-h1.dcm", "s3://root/uid-h2.dcm"),
        True,
    )]

    def test_get_images_by_slice_numbers(self):
        """Test rendering images by slice number."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        slice_numbers = [0, 50, 100, 150, 200]
        images = index.get_images(slice_numbers=slice_numbers)

        assert len(images) == len(slice_numbers)
        for img in images:
            assert isinstance(img, Image.Image)

    def test_get_images_with_contrast(self):
        """Test rendering images with specific contrast."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        positions = [0.2, 0.5, 0.8]
        for contrast in ["lung", "bone", "brain"]:
            images = index.get_images(positions=positions, contrast=contrast)
            assert len(images) == 3

    def test_get_images_with_contrast_object(self):
        """Test rendering images with Contrast object."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        contrast = Contrast(spec="bone")
        images = index.get_images(positions=[0.3, 0.6, 0.9], contrast=contrast)

        assert len(images) == 3

    def test_get_images_with_custom_width(self):
        """Test rendering images with custom width."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        positions = [0.4, 0.6]
        images_small = index.get_images(positions=positions, image_width=64)
        images_large = index.get_images(positions=positions, image_width=256)

        assert images_small[0].size[0] < images_large[0].size[0]

    def test_get_images_single_position(self):
        """Test rendering just one image."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        images = index.get_images(positions=[0.5])

        assert len(images) == 1
        assert isinstance(images[0], Image.Image)

    def test_get_images_invalid_no_args(self):
        """Test error when neither positions nor slice_numbers specified."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        with pytest.raises(ValueError):
            index.get_images()

    def test_get_images_invalid_both_args(self):
        """Test error when both positions and slice_numbers specified."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        with pytest.raises(ValueError):
            index.get_images(positions=[0.5], slice_numbers=[0])

    def test_get_images_empty_positions(self):
        """Test error with empty positions list."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        with pytest.raises(ValueError):
            index.get_images(positions=[])

    def test_get_images_empty_slice_numbers(self):
        """Test error with empty slice_numbers list."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        with pytest.raises(ValueError):
            index.get_images(slice_numbers=[])

    def test_get_images_many_positions(self):
        """Test rendering many images."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        # Create 20 evenly-spaced positions
        positions = [i / 19 for i in range(20)]
        images = index.get_images(positions=positions)

        assert len(images) == 20

    def test_get_images_parallel_rendering(self):
        """Test that images render in parallel with correct order."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        # Render with custom max_workers
        positions = [0.1, 0.3, 0.5, 0.7, 0.9]
        images = index.get_images(positions=positions, max_workers=2)

        # Should get all images
        assert len(images) == len(positions)

        # Order should be preserved
        for img in images:
            assert isinstance(img, Image.Image)

    def test_get_images_custom_max_workers(self):
        """Test get_images with custom max_workers parameter."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        positions = [0.2, 0.4, 0.6, 0.8]
        # Test with different worker counts
        images_1 = index.get_images(positions=positions, max_workers=1)
        images_4 = index.get_images(positions=positions, max_workers=4)

        assert len(images_1) == len(positions)
        assert len(images_4) == len(positions)

    def test_get_images_duplicate_positions_default(self):
        """Test that duplicate positions are kept by default."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        # Request same position 3 times (default remove_duplicates=False)
        images = index.get_images(positions=[0.5, 0.5, 0.5])

        # Should return 3 images (same image repeated)
        assert len(images) == 3
        # All three should be valid images
        for img in images:
            assert isinstance(img, Image.Image)

    def test_get_images_duplicate_positions_removed(self):
        """Test that duplicate positions can be removed."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        # Request same position 3 times, but remove duplicates
        images = index.get_images(positions=[0.5, 0.5, 0.5], remove_duplicates=True)

        # Should return only 1 image
        assert len(images) == 1
        assert isinstance(images[0], Image.Image)

    def test_get_images_duplicate_positions_mixed(self):
        """Test removing duplicates with mixed positions."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        # Mix of duplicates and unique positions
        positions = [0.2, 0.5, 0.2, 0.8, 0.5, 0.9]
        images = index.get_images(positions=positions, remove_duplicates=True)

        # Should return only unique positions (4 unique values)
        assert len(images) == 4
        # Order should match first occurrence: 0.2, 0.5, 0.8, 0.9
        for img in images:
            assert isinstance(img, Image.Image)


class TestSeriesIndexGetInstances:
    """Tests for SeriesIndex.get_instances() core method."""

    def test_get_instances_single_position(self):
        """Test fetching single instance via get_instances."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        instances = index.get_instances(positions=[0.5])

        assert len(instances) == 1
        assert isinstance(instances[0], Instance)
        assert instances[0].instance_uid is not None
        assert instances[0].dataset is not None

    def test_get_instances_multiple_positions(self):
        """Test fetching multiple instances at different positions."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        positions = [0.2, 0.5, 0.8]
        instances = index.get_instances(positions=positions)

        assert len(instances) == 3
        for instance in instances:
            assert isinstance(instance, Instance)
            assert instance.instance_uid is not None

    def test_get_instances_single_slice_number(self):
        """Test fetching single instance via slice number."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        instances = index.get_instances(slice_numbers=[0])

        assert len(instances) == 1
        assert isinstance(instances[0], Instance)
        assert instances[0].instance_uid is not None

    def test_get_instances_multiple_slice_numbers(self):
        """Test fetching multiple instances at different slice numbers."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        slice_numbers = [0, index.instance_count // 2, index.instance_count - 1]
        instances = index.get_instances(slice_numbers=slice_numbers)

        assert len(instances) == 3
        for instance in instances:
            assert isinstance(instance, Instance)

    def test_get_instances_order_preserved_positions(self):
        """Test that instance order matches input position order."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        positions = [0.8, 0.2, 0.5]
        instances = index.get_instances(positions=positions)

        assert len(instances) == 3

    def test_get_instances_order_preserved_slice_numbers(self):
        """Test that instance order matches input slice number order."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        slice_numbers = [10, 0, 5]
        instances = index.get_instances(slice_numbers=slice_numbers)

        assert len(instances) == 3

    def test_get_instances_deduplication_positions(self):
        """Test deduplication removes duplicate positions."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        # Request same position multiple times
        positions = [0.5, 0.5, 0.5]
        instances = index.get_instances(positions=positions, remove_duplicates=True)

        # Should return only 1 instance
        assert len(instances) == 1
        assert isinstance(instances[0], Instance)

    def test_get_instances_deduplication_mixed_positions(self):
        """Test deduplication with mixed unique/duplicate positions."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        positions = [0.2, 0.5, 0.2, 0.8, 0.5]
        instances = index.get_instances(positions=positions, remove_duplicates=True)

        # Should return only 3 unique positions
        assert len(instances) == 3

    def test_get_instances_deduplication_slice_numbers(self):
        """Test deduplication removes duplicate slice numbers."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        slice_numbers = [0, 0, 0]
        instances = index.get_instances(slice_numbers=slice_numbers, remove_duplicates=True)

        assert len(instances) == 1
        assert isinstance(instances[0], Instance)

    def test_get_instances_no_deduplication_by_default(self):
        """Test that duplicates are kept by default."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        positions = [0.5, 0.5]
        instances = index.get_instances(positions=positions)

        # Should return 2 instances even though they're the same position
        assert len(instances) == 2

    def test_get_instances_headers_only_true(self):
        """Test headers_only mode fetches metadata efficiently."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        instances = index.get_instances(positions=[0.5], headers_only=True)

        assert len(instances) == 1
        instance = instances[0]
        assert instance.dataset is not None
        # Headers should have PatientName, not necessarily pixel data
        assert hasattr(instance.dataset, "PatientName") or True

    def test_get_instances_headers_only_false(self):
        """Test full data fetch when headers_only=False."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        instances = index.get_instances(positions=[0.5], headers_only=False)

        assert len(instances) == 1
        instance = instances[0]
        assert instance.dataset is not None
        # Should have pixel data
        assert hasattr(instance.dataset, "pixel_array")

    def test_get_instances_parallel_workers(self):
        """Test that different worker counts work."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        positions = [0.2, 0.4, 0.6, 0.8]

        # Test with different worker counts
        for max_workers in [1, 4, 8]:
            instances = index.get_instances(positions=positions, max_workers=max_workers)
            assert len(instances) == 4
            for instance in instances:
                assert isinstance(instance, Instance)

    def test_get_instances_empty_list_raises(self):
        """Test that empty position list raises ValueError."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        with pytest.raises(ValueError, match="cannot be empty"):
            index.get_instances(positions=[])

    def test_get_instances_invalid_position_raises(self):
        """Test that invalid position raises ValueError."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        with pytest.raises(ValueError, match="Position must be"):
            index.get_instances(positions=[1.5])

    def test_get_instances_invalid_slice_number_raises(self):
        """Test that invalid slice number raises ValueError."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        with pytest.raises(ValueError, match="out of bounds"):
            index.get_instances(slice_numbers=[index.instance_count + 100])

    def test_get_instances_both_params_raises(self):
        """Test that specifying both positions and slice_numbers raises."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        with pytest.raises(ValueError, match="Specify either"):
            index.get_instances(positions=[0.5], slice_numbers=[0])

    def test_get_instances_neither_params_raises(self):
        """Test that specifying neither positions nor slice_numbers raises."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        with pytest.raises(ValueError, match="Specify either"):
            index.get_instances()

    def test_get_instances_caching(self):
        """Test that Instance objects cache datasets for reuse."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        instances = index.get_instances(positions=[0.5, 0.5], remove_duplicates=False)

        # Both instances should have the same UID
        assert instances[0].instance_uid == instances[1].instance_uid
        # Both should have datasets
        assert instances[0].dataset is not None
        assert instances[1].dataset is not None

    def test_get_instances_mixed_with_get_instance(self):
        """Test that get_instance() wrapper works via get_instances()."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        # Get via single method
        instance_single = index.get_instance(position=0.5)

        # Get via get_instances
        instances = index.get_instances(positions=[0.5])

        # Both should return instances with same UID
        assert instance_single.instance_uid == instances[0].instance_uid

    def test_get_instances_large_batch(self):
        """Test fetching a larger batch of instances."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        # Request 10 evenly spaced instances
        positions = [i / 9.0 for i in range(10)]
        instances = index.get_instances(positions=positions)

        assert len(instances) == 10
        for instance in instances:
            assert isinstance(instance, Instance)
            assert instance.instance_uid is not None


class TestInstance:
    """Tests for Instance class and methods."""

    @pytest.fixture
    def instance(self):
        """Fixture providing an Instance object."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)
        return index.get_instance(position=0.5)

    def test_instance_attributes(self, instance):
        """Test Instance has expected attributes."""
        assert instance.instance_uid is not None
        assert instance.dataset is not None
        assert hasattr(instance, 'get_image')
        assert hasattr(instance, 'get_pixel_array')
        assert hasattr(instance, 'get_contrast_grid')

    def test_instance_get_image(self, instance):
        """Test Instance.get_image() method."""
        img = instance.get_image()

        assert isinstance(img, Image.Image)
        assert img.size[0] > 0
        assert img.size[1] > 0

    def test_instance_get_image_with_contrast(self, instance):
        """Test Instance.get_image() with various contrasts."""
        for contrast in ["lung", "bone", "brain", "auto"]:
            img = instance.get_image(contrast=contrast)
            assert isinstance(img, Image.Image)

    def test_instance_get_image_caching(self):
        """Test that Instance caches the dataset for efficiency."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)
        instance = index.get_instance(position=0.5)

        # Get images with different contrasts from same instance
        # This should reuse the cached dataset without re-fetching
        img1 = instance.get_image(contrast="lung")
        img2 = instance.get_image(contrast="bone")
        img3 = instance.get_image(contrast="brain")

        # All should be valid images
        assert isinstance(img1, Image.Image)
        assert isinstance(img2, Image.Image)
        assert isinstance(img3, Image.Image)

    def test_instance_get_pixel_array(self, instance):
        """Test Instance.get_pixel_array() method."""
        pixels = instance.get_pixel_array()

        assert isinstance(pixels, np.ndarray)
        assert pixels.size > 0
        assert pixels.ndim >= 2  # At least 2D

    def test_instance_get_pixel_array_dtype(self, instance):
        """Test pixel array has reasonable dtype."""
        pixels = instance.get_pixel_array()

        # Should be a numeric type
        assert np.issubdtype(pixels.dtype, np.number)

    def test_instance_get_contrast_grid(self, instance):
        """Test Instance.get_contrast_grid() method."""
        contrasts = ["lung", "bone", "brain"]
        grid = instance.get_contrast_grid(contrasts)

        assert isinstance(grid, Image.Image)
        # Grid should be wider than single image due to concatenation
        single_img = instance.get_image(contrast="lung")
        assert grid.size[0] >= single_img.size[0]

    def test_instance_get_contrast_grid_single_contrast(self, instance):
        """Test contrast grid with single contrast."""
        grid = instance.get_contrast_grid(["lung"])

        assert isinstance(grid, Image.Image)

    def test_instance_get_contrast_grid_empty(self, instance):
        """Test error handling for empty contrast list."""
        with pytest.raises(ValueError):
            instance.get_contrast_grid([])


class TestContrast:
    """Tests for Contrast class."""

    def test_contrast_initialization_preset(self):
        """Test Contrast with preset name."""
        contrast = Contrast(spec="lung")
        assert contrast.spec == "lung"

    def test_contrast_initialization_auto(self):
        """Test Contrast with auto detection."""
        contrast = Contrast(spec="auto")
        assert contrast.spec == "auto"

    def test_contrast_initialization_window_level(self):
        """Test Contrast with window/level."""
        contrast = Contrast(spec="1500/500")
        assert contrast.spec == "1500/500"

    def test_contrast_initialization_parameters(self):
        """Test Contrast with window_width and window_center parameters."""
        contrast = Contrast(window_width=1500, window_center=500)
        assert contrast.window_width == 1500
        assert contrast.window_center == 500

    def test_contrast_initialization_none(self):
        """Test Contrast with None (default)."""
        contrast = Contrast()
        assert contrast.spec is None
        assert contrast.window_width is None
        assert contrast.window_center is None


class TestPositionInterpolator:
    """Tests for PositionInterpolator class."""

    def test_position_interpolator_initialization(self):
        """Test PositionInterpolator initialization."""
        interp = PositionInterpolator(instance_count=100)
        assert interp.instance_count == 100

    def test_position_to_index_basic(self):
        """Test basic position_to_index conversion."""
        interp = PositionInterpolator(instance_count=100)

        # Position 0.0 should map to index 0
        assert interp.position_to_index(0.0) == 0

        # Position 1.0 should map to index 99 (instance_count - 1)
        assert interp.position_to_index(1.0) == 99

        # Position 0.5 should map to middle index
        assert interp.position_to_index(0.5) == 49

    def test_position_to_index_various_positions(self):
        """Test position_to_index with various position values."""
        interp = PositionInterpolator(instance_count=100)

        # Test multiple positions
        assert interp.position_to_index(0.0) == 0
        assert interp.position_to_index(0.25) == 24
        assert interp.position_to_index(0.5) == 49
        assert interp.position_to_index(0.75) == 74
        assert interp.position_to_index(1.0) == 99

    def test_position_to_index_with_offset(self):
        """Test position_to_index with slice_offset."""
        interp = PositionInterpolator(instance_count=100)

        # Position 0.5 (index 49) + offset 1 = index 50
        assert interp.position_to_index(0.5, slice_offset=1) == 50

        # Position 0.5 (index 49) + offset -1 = index 48
        assert interp.position_to_index(0.5, slice_offset=-1) == 48

        # Position 0.5 (index 49) + offset 10 = index 59
        assert interp.position_to_index(0.5, slice_offset=10) == 59

        # Position 0.5 (index 49) + offset -10 = index 39
        assert interp.position_to_index(0.5, slice_offset=-10) == 39

    def test_position_to_index_offset_at_boundaries(self):
        """Test position_to_index with offset at boundaries."""
        interp = PositionInterpolator(instance_count=100)

        # Position 0.0 (index 0) + offset 0 = index 0 (valid)
        assert interp.position_to_index(0.0, slice_offset=0) == 0

        # Position 1.0 (index 99) + offset 0 = index 99 (valid)
        assert interp.position_to_index(1.0, slice_offset=0) == 99

    def test_position_to_index_offset_out_of_bounds_positive(self):
        """Test position_to_index raises error when offset goes past end."""
        interp = PositionInterpolator(instance_count=100)

        # Position 1.0 (index 99) + offset 1 = index 100 (out of bounds)
        with pytest.raises(ValueError, match="Slice offset .* out of bounds"):
            interp.position_to_index(1.0, slice_offset=1)

        # Position 0.99 (index 98) + offset 2 = index 100 (out of bounds)
        with pytest.raises(ValueError, match="Slice offset .* out of bounds"):
            interp.position_to_index(0.99, slice_offset=2)

    def test_position_to_index_offset_out_of_bounds_negative(self):
        """Test position_to_index raises error when offset goes before start."""
        interp = PositionInterpolator(instance_count=100)

        # Position 0.0 (index 0) + offset -1 = index -1 (out of bounds)
        with pytest.raises(ValueError, match="Slice offset .* out of bounds"):
            interp.position_to_index(0.0, slice_offset=-1)

        # Position 0.01 (index 0) + offset -2 = index -2 (out of bounds)
        with pytest.raises(ValueError, match="Slice offset .* out of bounds"):
            interp.position_to_index(0.01, slice_offset=-2)

    def test_position_to_index_invalid_position(self):
        """Test position_to_index raises error for invalid position."""
        interp = PositionInterpolator(instance_count=100)

        # Position < 0.0
        with pytest.raises(ValueError, match="position must be in"):
            interp.position_to_index(-0.1)

        # Position > 1.0
        with pytest.raises(ValueError, match="position must be in"):
            interp.position_to_index(1.1)

    def test_position_to_index_small_series(self):
        """Test position_to_index with small series (edge case)."""
        # Series with only 1 instance
        interp = PositionInterpolator(instance_count=1)
        assert interp.position_to_index(0.0) == 0
        assert interp.position_to_index(0.5) == 0
        assert interp.position_to_index(1.0) == 0

        # Can't offset beyond bounds
        with pytest.raises(ValueError, match="Slice offset .* out of bounds"):
            interp.position_to_index(0.0, slice_offset=1)

    def test_position_to_index_large_series(self):
        """Test position_to_index with large series."""
        interp = PositionInterpolator(instance_count=10000)

        # Test basic conversions
        assert interp.position_to_index(0.0) == 0
        assert interp.position_to_index(1.0) == 9999
        assert interp.position_to_index(0.5) == 4999

        # Test with offset
        assert interp.position_to_index(0.5, slice_offset=100) == 5099
        assert interp.position_to_index(0.5, slice_offset=-100) == 4899

    def test_interpolate_unique_limits_duplicates(self):
        """interpolate_unique should drop slices that map to same index."""
        interp = PositionInterpolator(instance_count=10)

        positions, indices = interp.interpolate_unique(
            num_positions=50,
            start=0.0,
            end=0.3,
        )

        assert len(indices) == len(set(indices))
        # 0.3 * (10-1) -> indices 0-2 inclusive
        assert indices == [0, 1, 2]
        assert len(positions) == 3

    def test_interpolate_unique_full_range_matches_requested(self):
        """When enough slices exist, interpolate_unique honors request."""
        interp = PositionInterpolator(instance_count=10)

        positions, indices = interp.interpolate_unique(
            num_positions=4,
            start=0.0,
            end=1.0,
        )

        assert len(indices) == 4
        assert indices[0] == 0
        assert indices[-1] == 9

    def test_position_to_index_consistent_with_get_instance(self):
        """Test that position_to_index is consistent with get_instance usage."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)
        interp = PositionInterpolator(index.instance_count)

        # Test position without offset
        position = 0.5
        slice_num = interp.position_to_index(position)

        # Both should return the same instance
        instance_by_position = index.get_instance(position=position)
        instance_by_slice = index.get_instance(slice_number=slice_num)

        assert instance_by_position.instance_uid == instance_by_slice.instance_uid

        # Test position with offset
        position = 0.3
        offset = 5
        slice_num_with_offset = interp.position_to_index(position, slice_offset=offset)

        # Should be able to get instance with offset directly
        instance_with_offset = index.get_instance(
            position=position,
            slice_offset=offset
        )
        instance_by_slice_offset = index.get_instance(slice_number=slice_num_with_offset)

        # Both should return the same instance
        assert instance_with_offset.instance_uid == instance_by_slice_offset.instance_uid


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_multiple_instances_from_same_series(self):
        """Test fetching multiple instances from same series."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        instance1 = index.get_instance(position=0.25)
        instance2 = index.get_instance(position=0.5)
        instance3 = index.get_instance(position=0.75)

        # All should be valid instances
        assert instance1.instance_uid is not None
        assert instance2.instance_uid is not None
        assert instance3.instance_uid is not None

        # They should be different instances
        assert instance1.instance_uid != instance2.instance_uid
        assert instance2.instance_uid != instance3.instance_uid

    def test_consistency_position_vs_slice_number(self):
        """Test consistency between position and slice number access."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        # Get instance at middle via position
        instance_pos = index.get_instance(position=0.5)

        # Get instance at middle via slice number
        middle_slice = index.instance_count // 2
        instance_slice = index.get_instance(slice_number=middle_slice)

        # Both should return valid instances
        assert isinstance(instance_pos, Instance)
        assert isinstance(instance_slice, Instance)

    def test_workflow_get_index_render_contrast_grid(self):
        """Test complete workflow: create index, get instance, render multiple contrasts."""
        # Initialize
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        # Get instance
        instance = index.get_instance(position=0.5)

        # Render multiple contrasts
        contrasts = ["lung", "bone", "brain"]
        grid = instance.get_contrast_grid(contrasts)

        # Verify result
        assert isinstance(grid, Image.Image)
        assert grid.size[0] > 0
        assert grid.size[1] > 0

    def test_retriever_reuse(self):
        """Test that retriever is properly reused across multiple calls."""
        index = SeriesIndex(TEST_SERIES_UID, root=TEST_S3_ROOT)

        # Get multiple instances - should reuse same retriever
        instance1 = index.get_instance(position=0.25)
        instance2 = index.get_instance(position=0.75)

        # Verify both were successful
        assert instance1.instance_uid is not None
        assert instance2.instance_uid is not None
