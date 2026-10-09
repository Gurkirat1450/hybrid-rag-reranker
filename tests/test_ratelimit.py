"""Tests for app/ratelimit.py, using an injected fake clock."""

import pytest

from app.ratelimit import SlidingWindowRateLimiter, resolve_client_ip


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def test_allows_up_to_limit_then_blocks():
    clock = FakeClock(0.0)
    rl = SlidingWindowRateLimiter(limit=3, window_seconds=60, clock=clock)
    assert rl.hit("ip", 1) == (True, 0.0)
    assert rl.hit("ip", 1) == (True, 0.0)
    assert rl.hit("ip", 1) == (True, 0.0)
    allowed, retry = rl.hit("ip", 1)
    # oldest event at t=0 expires at 0 + 60 -> retry after 60 - 0 = 60
    assert allowed is False
    assert retry == pytest.approx(60.0)


def test_window_slides():
    clock = FakeClock(0.0)
    rl = SlidingWindowRateLimiter(limit=3, window_seconds=60, clock=clock)
    for _ in range(3):
        rl.hit("ip", 1)
    clock.t = 59.5
    allowed, retry = rl.hit("ip", 1)
    # still inside the window: 0 + 60 - 59.5 = 0.5
    assert allowed is False
    assert retry == pytest.approx(0.5)
    clock.t = 60.0
    # at t=60 the three events from t=0 are out of the window (ts <= 60 - 60)
    assert rl.hit("ip", 1) == (True, 0.0)


def test_costs_and_retry_after():
    clock = FakeClock(0.0)
    rl = SlidingWindowRateLimiter(limit=5, window_seconds=60, clock=clock)
    assert rl.hit("ip", 3)[0] is True  # t=0,  used 3
    clock.t = 10.0
    assert rl.hit("ip", 1)[0] is True  # t=10, used 4
    clock.t = 20.0
    allowed, retry = rl.hit("ip", 3)  # 4 + 3 = 7 > 5, need to free 2
    # freeing the t=0 event (cost 3) is enough -> retry = 0 + 60 - 20 = 40
    assert allowed is False
    assert retry == pytest.approx(40.0)
    clock.t = 60.0
    # t=0 event expired; used = 1 (t=10); 1 + 3 = 4 <= 5
    assert rl.hit("ip", 3)[0] is True


def test_denied_requests_do_not_consume_budget():
    clock = FakeClock(0.0)
    rl = SlidingWindowRateLimiter(limit=2, window_seconds=60, clock=clock)
    rl.hit("ip", 2)
    for _ in range(5):
        assert rl.hit("ip", 1)[0] is False
    clock.t = 60.0
    # only the t=0 event (cost 2) was recorded; it has expired -> full budget again
    assert rl.hit("ip", 2)[0] is True


def test_cost_above_limit_is_always_denied():
    rl = SlidingWindowRateLimiter(limit=2, window_seconds=60, clock=FakeClock(0.0))
    allowed, retry = rl.hit("ip", 3)
    assert allowed is False
    assert retry == pytest.approx(60.0)


def test_keys_are_independent():
    rl = SlidingWindowRateLimiter(limit=1, window_seconds=60, clock=FakeClock(0.0))
    assert rl.hit("a", 1)[0] is True
    assert rl.hit("a", 1)[0] is False
    assert rl.hit("b", 1)[0] is True


def test_client_ip_ignores_forwarded_header_by_default():
    assert resolve_client_ip("10.0.0.1", "1.1.1.1, 2.2.2.2", trust_proxy=False) == "10.0.0.1"


def test_client_ip_uses_last_forwarded_entry_when_trusted():
    # the first entry is client-controlled (spoofable); the last was appended by the nearest proxy
    assert resolve_client_ip("10.0.0.1", "6.6.6.6, 2.2.2.2", trust_proxy=True) == "2.2.2.2"
    assert resolve_client_ip("10.0.0.1", " 3.3.3.3 ", trust_proxy=True) == "3.3.3.3"
    assert resolve_client_ip("10.0.0.1", "4.4.4.4, ", trust_proxy=True) == "4.4.4.4"


def test_client_ip_falls_back_when_header_missing():
    assert resolve_client_ip("10.0.0.1", None, trust_proxy=True) == "10.0.0.1"
    assert resolve_client_ip("10.0.0.1", "", trust_proxy=True) == "10.0.0.1"
    assert resolve_client_ip(None, None, trust_proxy=False) == "unknown"
