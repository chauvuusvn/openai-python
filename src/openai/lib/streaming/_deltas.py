from __future__ import annotations

from typing import TypeVar, Sequence
from ..._utils import is_dict, is_list

_T = TypeVar("_T")


def _has_indexed_entries(value: object) -> bool:
    return is_list(value) and any(is_dict(entry) and "index" in entry for entry in value)


def _get_entry_attr(entry: object, key: str) -> object | None:
    if is_dict(entry):
        return entry.get(key)
    return getattr(entry, key, None)


def find_indexed_entry(
    entries: Sequence[_T] | None,
    logical_index: int,
    entry_id: str | None = None,
) -> _T | None:
    """Safely find an entry in an accumulated sequence by logical index or ID.

    For indexed entries, returns None if unmatched to prevent cross-call corruption.
    Preserves positional access only for legacy unindexed snapshots.
    """
    if not entries:
        return None

    # 1. Match by explicit entry ID (supports both models and mappings)
    if entry_id:
        for entry in entries:
            if _get_entry_attr(entry, "id") == entry_id:
                return entry

    # 2. Match by explicit logical index
    has_explicit_indices = False
    for entry in entries:
        idx = _get_entry_attr(entry, "index")
        if idx is not None:
            has_explicit_indices = True
            if idx == logical_index:
                return entry

    # 3. Positional fallback strictly for legacy unindexed snapshots
    if not has_explicit_indices and 0 <= logical_index < len(entries):
        return entries[logical_index]

    return None


def accumulate_delta(acc: dict[object, object], delta: dict[object, object]) -> dict[object, object]:
    for key, delta_value in delta.items():
        if key not in acc or acc[key] is None:
            if _has_indexed_entries(delta_value):
                # Even the first indexed list must be merged: one chunk can
                # contain several fragments for the same index.
                acc[key] = []
            else:
                acc[key] = delta_value
                continue

        acc_value = acc[key]

        # the `index` property is used in arrays of objects so it should
        # not be accumulated like other values e.g.
        # [{'foo': 'bar', 'index': 0}]
        #
        # the same applies to `type` properties as they're used for
        # discriminated unions
        if key == "index" or key == "type":
            acc[key] = delta_value
            continue

        if isinstance(acc_value, str) and isinstance(delta_value, str):
            acc_value += delta_value
        elif isinstance(acc_value, (int, float)) and isinstance(delta_value, (int, float)):
            acc_value += delta_value
        elif is_dict(acc_value) and is_dict(delta_value):
            acc_value = accumulate_delta(acc_value, delta_value)
        elif is_list(acc_value) and is_list(delta_value):
            # for lists of non-dictionary items we'll only ever get new entries
            # in the array, existing entries will never be changed
            # An empty accumulator cannot tell us whether this is an indexed
            # list. Inspect the delta before taking the append-only path.
            if all(isinstance(x, (str, int, float)) for x in acc_value) and (
                acc_value or not _has_indexed_entries(delta_value)
            ):
                acc_value.extend(delta_value)
                continue

            for delta_entry in delta_value:
                if not is_dict(delta_entry):
                    raise TypeError(f"Unexpected list delta entry is not a dictionary: {delta_entry}")

                try:
                    index = delta_entry["index"]
                except KeyError as exc:
                    raise RuntimeError(f"Expected list delta entry to have an `index` key; {delta_entry}") from exc

                if not isinstance(index, int):
                    raise TypeError(f"Unexpected, list delta entry `index` value is not an integer; {index}")

                matching_idx: int | None = None
                for i, existing in enumerate(acc_value):
                    if is_dict(existing):
                        existing_idx = existing.get("index")
                        if existing_idx == index or (existing_idx is None and i == index):
                            matching_idx = i
                            break

                if matching_idx is None:
                    # Insert in ordered position by logical index without allocating sparse placeholders
                    insert_pos = 0
                    while insert_pos < len(acc_value):
                        existing_entry = acc_value[insert_pos]
                        if is_dict(existing_entry):
                            existing_idx = existing_entry.get("index")
                            if isinstance(existing_idx, int) and existing_idx > index:
                                break
                        insert_pos += 1
                    acc_value.insert(insert_pos, delta_entry)
                else:
                    acc_entry = acc_value[matching_idx]
                    if not is_dict(acc_entry):
                        raise TypeError("not handled yet")

                    acc_value[matching_idx] = accumulate_delta(acc_entry, delta_entry)

        acc[key] = acc_value

    return acc
