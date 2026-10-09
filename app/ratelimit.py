"""In-memory sliding-window rate limiter, keyed by client IP.

Each request records (timestamp, cost). A request is allowed if the sum of costs
recorded in the last ``window_seconds`` plus its own cost stays within ``limit``.
State lives in process memory: it resets on restart and is not shared between
workers, which is fine for a single-process demo.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable

_SWEEP_EVERY = 1000


def resolve_client_ip(client_host: str | None, forwarded_for: str | None, trust_proxy: bool) -> str:
    """Pick the IP to rate-limit on.

    * Default: the TCP peer address (``client_host``).
    * Only when ``trust_proxy`` is True and an X-Forwarded-For value is present, use
      its LAST entry: the one appended by the nearest proxy. The first entries are
      supplied by the client and can be forged.
    """
    if trust_proxy and forwarded_for:
        parts = [p.strip() for p in forwarded_for.split(",") if p.strip()]
        if parts:
            return parts[-1]
    return client_host or "unknown"


class SlidingWindowRateLimiter:
    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limit = limit
        self.window = window_seconds
        self._clock = clock
        self._events: dict[str, deque[tuple[float, int]]] = {}
        self._lock = threading.Lock()
        self._calls = 0

    def _prune(self, q: deque[tuple[float, int]], now: float) -> None:
        cutoff = now - self.window
        while q and q[0][0] <= cutoff:
            q.popleft()

    def _sweep(self, now: float) -> None:
        for key in list(self._events):
            q = self._events[key]
            self._prune(q, now)
            if not q:
                del self._events[key]

    def hit(self, key: str, cost: int = 1) -> tuple[bool, float]:
        """Try to spend ``cost`` units for ``key``.

        Returns (allowed, retry_after_seconds). retry_after is 0.0 when allowed;
        otherwise it is the time until enough old events expire for this request
        to fit (or the full window if ``cost`` exceeds ``limit``).
        """
        now = self._clock()
        with self._lock:
            self._calls += 1
            if self._calls % _SWEEP_EVERY == 0:
                self._sweep(now)

            if cost > self.limit:
                return False, self.window

            q = self._events.setdefault(key, deque())
            self._prune(q, now)
            used = sum(c for _, c in q)
            if used + cost <= self.limit:
                q.append((now, cost))
                return True, 0.0

            need = used + cost - self.limit
            freed = 0
            for ts, c in q:
                freed += c
                if freed >= need:
                    return False, max(0.0, ts + self.window - now)
            return False, self.window  # unreachable: need <= used

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
