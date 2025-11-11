"""Worker pool optimization utilities."""


def optimal_workers(n, max_workers, min_workers=1):
    """
    Find optimal number of workers for n items.

    Prefers divisors of n for even work distribution, but ensures
    we use reasonable parallelism even when perfect divisors don't exist.

    Args:
        n: number of items to process
        max_workers: maximum workers allowed
        min_workers: minimum acceptable workers (default 1)

    Returns:
        number of workers to use

    Examples:
        >>> optimal_workers(100, 50)  # 100 items, max 50 workers
        50
        >>> optimal_workers(9, 10, min_workers=5)  # 3x3 mosaic
        9
        >>> optimal_workers(36, 10, min_workers=5)  # 6x6 mosaic
        6
        >>> optimal_workers(97, 50, min_workers=1)  # Prime number
        1
    """
    if n <= max_workers:
        return n

    # Find all divisors of n in the valid range
    divisors = []
    for d in range(min_workers, max_workers + 1):
        if n % d == 0:
            divisors.append(d)

    if divisors:
        # Use the largest divisor (closest to max_workers)
        return max(divisors)

    # No perfect divisor exists in range
    # Find divisor closest to max_workers, even if below min_workers
    best_divisor = 1
    for d in range(2, int(n**0.5) + 1):
        if n % d == 0:
            # d is a divisor
            if d <= max_workers:
                best_divisor = max(best_divisor, d)
            # n/d is also a divisor
            other = n // d
            if other <= max_workers:
                best_divisor = max(best_divisor, other)

    # If we found a decent divisor below max_workers, use it
    if best_divisor >= min_workers:
        return best_divisor

    # Otherwise, just use max_workers (accept uneven distribution)
    return max_workers
