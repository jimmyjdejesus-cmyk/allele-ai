from phylogenic.llm_client import RateLimiter


def test_rate_limiter_can_be_constructed_without_event_loop():
    """Ensure that constructing a RateLimiter synchronously doesn't require an
    existing asyncio event loop (fixes RuntimeError on Python 3.9 CI runners).
    """
    rl = RateLimiter(60, 10000)
    assert rl.requests_per_minute == 60
    assert rl.tokens_per_minute == 10000
