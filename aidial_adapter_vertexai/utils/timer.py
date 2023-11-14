import time


class Timer:
    start: float

    def __init__(self):
        self.start = time.perf_counter()

    def stop(self) -> float:
        return time.perf_counter() - self.start

    def __str__(self) -> str:
        return f"{self.stop():.3f}s"
