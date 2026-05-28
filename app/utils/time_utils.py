import time
from contextlib import contextmanager
from typing import Generator


@contextmanager
def measure_time() -> Generator[dict[str, float], None, None]:
    result: dict[str, float] = {"elapsed_ms": 0.0}
    start = time.perf_counter()
    yield result
    result["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 2)
