"""Global rate limiter for API calls to optimize throughput.

Based on testing:
- Optimal: 25 parallel workers → 19.73 calls/sec
- Above 25: Throughput degrades significantly
- At 100 workers: Only 3.56 calls/sec (worse!)

This module provides a global semaphore to limit concurrent API calls.
"""

import threading

# Global semaphore to limit concurrent API calls
# Set to 25 based on rate limit testing results
MAX_CONCURRENT_API_CALLS = 25
api_semaphore = threading.Semaphore(MAX_CONCURRENT_API_CALLS)


def get_api_semaphore():
    """Get the global API semaphore for rate limiting."""
    return api_semaphore


def set_max_concurrent_calls(max_calls: int):
    """Update the maximum concurrent API calls limit.

    Args:
        max_calls: New maximum (e.g., 25 for optimal throughput)
    """
    global api_semaphore, MAX_CONCURRENT_API_CALLS
    MAX_CONCURRENT_API_CALLS = max_calls
    api_semaphore = threading.Semaphore(max_calls)
    print(f"[RateLimiter] Max concurrent API calls set to {max_calls}")
