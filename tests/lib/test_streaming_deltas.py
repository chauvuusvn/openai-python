from __future__ import annotations

from copy import deepcopy
from typing import cast

import pytest

from openai.lib.streaming._deltas import accumulate_delta, find_indexed_entry


@pytest.mark.parametrize("initial", [{}, {"tool_calls": None}, {"tool_calls": []}])
def test_duplicate_indexes_in_initial_list(initial: dict[object, object]) -> None:
    acc = deepcopy(initial)
    accumulate_delta(
        acc,
        {
            "tool_calls": [
                {"index": 0, "id": "call_abc", "type": "function", "function": {"name": "list_files"}},
                {"index": 0, "function": {"arguments": '{"path"'}},
            ]
        },
    )
    accumulate_delta(acc, {"tool_calls": [{"index": 0, "function": {"arguments": ': "."}'}}]})
    assert acc["tool_calls"] == [
        {
            "index": 0,
            "id": "call_abc",
            "type": "function",
            "function": {"name": "list_files", "arguments": '{"path": "."}'},
        }
    ]


@pytest.mark.parametrize(
    "initial,delta,expected",
    [
        ({}, {"value": [1, "a"]}, {"value": [1, "a"]}),
        ({"value": None}, {"value": [1, "a"]}, {"value": [1, "a"]}),
        ({"value": []}, {"value": [1, "a"]}, {"value": [1, "a"]}),
        ({"value": [1]}, {"value": [2, "a"]}, {"value": [1, 2, "a"]}),
        ({"value": []}, {"value": [{"text": "a"}]}, {"value": [{"text": "a"}]}),
        ({}, {"value": [{"text": "a"}]}, {"value": [{"text": "a"}]}),
        ({"value": [1]}, {"value": []}, {"value": [1]}),
        # Full Assistants snapshots omit the delta-only index field.
        (
            {"value": [{"text": "a"}]},
            {"value": [{"index": 0, "text": "b"}]},
            {"value": [{"index": 0, "text": "ab"}]},
        ),
        (
            {"index": 1, "type": "text", "text": {"value": "a"}, "count": 2},
            {"index": 1, "type": "text", "text": {"value": "b"}, "count": 3},
            {"index": 1, "type": "text", "text": {"value": "ab"}, "count": 5},
        ),
        (
            {"function": {"name": "list_", "arguments": "{"}},
            {"function": {"name": "files", "arguments": "}"}},
            {"function": {"name": "list_files", "arguments": "{}"}},
        ),
    ],
)
def test_existing_delta_semantics(
    initial: dict[object, object], delta: dict[object, object], expected: dict[object, object]
) -> None:
    assert accumulate_delta(deepcopy(initial), deepcopy(delta)) == expected


@pytest.mark.parametrize(
    "entry,error,match",
    [
        ("invalid", TypeError, "not a dictionary"),
        ({"text": "invalid"}, RuntimeError, "an `index` key"),
        ({"index": "0"}, TypeError, "not an integer"),
    ],
)
def test_invalid_indexed_delta(entry: object, error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        accumulate_delta({"content": [{"index": 0, "text": "a"}]}, {"content": [entry]})


def test_out_of_order_indexed_delta_accumulation() -> None:
    acc: dict[object, object] = {}
    chunk_index_1: dict[object, object] = {
        "tool_calls": [
            {"index": 1, "id": "call_1", "type": "function", "function": {"name": "func_one", "arguments": '{"a": 1}'}}
        ]
    }
    chunk_index_0: dict[object, object] = {
        "tool_calls": [
            {"index": 0, "id": "call_0", "type": "function", "function": {"name": "func_zero", "arguments": '{"b": 2}'}}
        ]
    }
    accumulate_delta(acc, chunk_index_1)
    accumulate_delta(acc, chunk_index_0)

    assert acc["tool_calls"] == [
        {
            "index": 0,
            "id": "call_0",
            "type": "function",
            "function": {"name": "func_zero", "arguments": '{"b": 2}'},
        },
        {
            "index": 1,
            "id": "call_1",
            "type": "function",
            "function": {"name": "func_one", "arguments": '{"a": 1}'},
        },
    ]


def test_sparse_indexed_delta_accumulation_no_placeholder() -> None:
    acc: dict[object, object] = {}
    chunk_sparse: dict[object, object] = {
        "tool_calls": [
            {"index": 7, "id": "call_7", "type": "function", "function": {"name": "func_sparse", "arguments": '{"val": 7}'}}
        ]
    }
    chunk_sparse_delta: dict[object, object] = {
        "tool_calls": [
            {"index": 7, "function": {"arguments": ', "extra": true}'}}
        ]
    }
    accumulate_delta(acc, chunk_sparse)
    accumulate_delta(acc, chunk_sparse_delta)

    # Must allocate only 1 entry for sparse index 7 without filling placeholder slots
    tool_calls = cast("list[dict[str, object]]", acc["tool_calls"])
    assert len(tool_calls) == 1
    assert tool_calls[0] == {
        "index": 7,
        "id": "call_7",
        "type": "function",
        "function": {"name": "func_sparse", "arguments": '{"val": 7}, "extra": true}'},
    }


def test_large_sparse_indexed_delta_no_memory_bloat() -> None:
    acc: dict[object, object] = {}
    chunk_large: dict[object, object] = {
        "tool_calls": [
            {"index": 100000, "id": "call_large", "type": "function", "function": {"name": "func_large", "arguments": '{"k": 1}'}}
        ]
    }
    accumulate_delta(acc, chunk_large)
    tool_calls = cast("list[dict[str, object]]", acc["tool_calls"])
    assert len(tool_calls) == 1
    assert tool_calls[0]["index"] == 100000


def test_find_indexed_entry_helper() -> None:
    class DummyCall:
        def __init__(self, id: str | None, index: int | None) -> None:
            self.id = id
            self.index = index

    item0 = DummyCall(id="call_0", index=0)
    item7 = DummyCall(id="call_7", index=7)
    calls = [item0, item7]

    assert find_indexed_entry(calls, 0) is item0
    assert find_indexed_entry(calls, 7) is item7
    assert find_indexed_entry(calls, 999, entry_id="call_7") is item7
    assert find_indexed_entry(None, 0) is None
    assert find_indexed_entry([], 0) is None

    # (1) Unknown index with existing entries returns None without falling back to last item
    assert find_indexed_entry(calls, 5) is None
    assert find_indexed_entry(calls, 99) is None

    # (2) ID and index lookups over mapping-shaped dictionary entries
    dict_calls = [
        {"id": "call_d0", "index": 0, "name": "fn0"},
        {"id": "call_d1", "index": 1, "name": "fn1"},
    ]
    assert find_indexed_entry(dict_calls, 1) == {"id": "call_d1", "index": 1, "name": "fn1"}
    assert find_indexed_entry(dict_calls, 999, entry_id="call_d0") == {"id": "call_d0", "index": 0, "name": "fn0"}
    assert find_indexed_entry(dict_calls, 42) is None

    # (3) Legacy unindexed snapshots allow bounded positional fallback
    legacy_calls = [DummyCall(id=None, index=None), DummyCall(id=None, index=None)]
    assert find_indexed_entry(legacy_calls, 0) is legacy_calls[0]
    assert find_indexed_entry(legacy_calls, 1) is legacy_calls[1]
    assert find_indexed_entry(legacy_calls, 2) is None

