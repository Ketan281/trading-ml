"""Small in-process TTL cache for API endpoints."""

import time

_CACHE = {}
_TTL = 60


def cached(key, builder, ttl=_TTL):
    now = time.time()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    value = builder()
    _CACHE[key] = (now, value)
    return value
