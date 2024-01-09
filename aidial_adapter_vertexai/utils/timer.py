import time
from typing import Callable


class Timer:
    start: float
    format: str
    printer: Callable[[str], None]

    def __init__(
        self,
        format: str = "Elapsed time: {time}",
        printer: Callable[[str], None] = print,
    ):
        self.start = time.perf_counter()
        self.format = format
        self.printer = printer

    def stop(self) -> float:
        return time.perf_counter() - self.start

    def __str__(self) -> str:
        return f"{self.stop():.3f}s"

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        self.printer(self.format.format(time=self))
