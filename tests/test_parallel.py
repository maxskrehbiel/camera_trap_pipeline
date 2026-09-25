from __future__ import annotations

import time

import pytest

from camera_trap_pipeline.parallel import ordered_map


def test_results_keep_input_order_even_when_later_items_finish_first() -> None:
    def slow_for_small(value: int) -> int:
        time.sleep(0.02 * (5 - value))
        return value * 10

    results = list(ordered_map(slow_for_small, range(5), workers=4))
    assert results == [(i, i * 10) for i in range(5)]


def test_expected_exceptions_are_yielded_and_others_raised() -> None:
    def fail_on_two(value: int) -> int:
        if value == 2:
            raise ValueError("bad item")
        return value

    results = dict(ordered_map(fail_on_two, [1, 2, 3], workers=2, lookahead=1, catch=(ValueError,)))
    assert results[1] == 1
    assert results[3] == 3
    assert isinstance(results[2], ValueError)
    with pytest.raises(ValueError, match="bad item"):
        list(ordered_map(fail_on_two, [1, 2, 3], workers=2))


def test_empty_input_yields_nothing() -> None:
    assert list(ordered_map(str, [], workers=0)) == []
