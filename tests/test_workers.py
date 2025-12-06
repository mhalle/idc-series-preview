from idc_series_preview.workers import optimal_workers


def test_optimal_workers_enforces_minimum_for_small_jobs():
    assert optimal_workers(4, max_workers=10, min_workers=5) == 5


def test_optimal_workers_prefers_nearby_divisor_when_available():
    assert optimal_workers(12, max_workers=10, min_workers=3) == 6
