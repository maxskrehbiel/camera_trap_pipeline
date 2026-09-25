"""Ordered thread-pool map used to decode images ahead of the model that consumes them."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def ordered_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    workers: int = 4,
    lookahead: int | None = None,
    catch: tuple[type[Exception], ...] = (),
) -> Iterator[tuple[T, R | Exception]]:
    """Apply ``fn`` to each item in a thread pool and yield results in input order.

    JPEG decoding and resizing release the GIL, so threads keep a model fed without the
    cost of processes.

    Args:
        fn: Function to apply.
        items: Inputs, consumed lazily.
        workers: Thread count (at least 1).
        lookahead: Maximum calls in flight; defaults to ``4 * workers``.
        catch: Exception types that are expected for single items (such as a corrupt
            file). They are yielded as the result so the caller can record them; any other
            exception propagates with its traceback.

    Yields:
        ``(item, result)`` pairs in input order, where ``result`` may be a caught exception.
    """
    iterator = iter(items)
    workers = max(1, workers)
    in_flight = max(1, lookahead if lookahead is not None else 4 * workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: deque[tuple[T, Future[R]]] = deque()

        def submit_next() -> bool:
            try:
                item = next(iterator)
            except StopIteration:
                return False
            pending.append((item, pool.submit(fn, item)))
            return True

        for _ in range(in_flight):
            if not submit_next():
                break
        while pending:
            item, future = pending.popleft()
            submit_next()
            try:
                result: R | Exception = future.result()
            except catch as exc:
                result = exc
            yield item, result
