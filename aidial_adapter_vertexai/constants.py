import os


def _get_sse_heartbeat_interval() -> float | None:
    if (val := os.getenv("SSE_HEARTBEAT_INTERVAL")) is None:
        return None
    return float(val)


SSE_HEARTBEAT_INTERVAL = _get_sse_heartbeat_interval()
