from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


def trim_valid_rows(arr):
    if hasattr(arr, "ndim") and int(getattr(arr, "ndim", 0)) == 1:
        arr = [arr]
    rows = [list(map(float, row)) for row in arr]
    if not rows:
        return []
    keep = []
    for row in rows:
        feat = row[1:] if len(row) > 1 else row
        if any(abs(v) > 1e-8 for v in feat):
            keep.append(row)
    return keep


def normalize_k_counts(values: Sequence[float]) -> List[int]:
    out: List[int] = []
    for value in values:
        k = int(round(float(value)))
        if k < 0:
            k = 0
        if k > 2:
            k = 2
        out.append(k)
    return out


def preorder_kcount_parent_indices(k_counts: Sequence[int]) -> List[int]:
    values = normalize_k_counts(k_counts)
    parents = [-1] * len(values)
    stack: List[List[int]] = []
    for idx, value in enumerate(values):
        while stack and stack[-1][1] <= 0:
            stack.pop()
        if stack:
            parents[idx] = int(stack[-1][0])
            stack[-1][1] -= 1
        if value > 0:
            stack.append([idx, value])
    return parents


def parents_to_children(parents: Sequence[int]) -> List[List[int]]:
    parents = [int(v) for v in parents]
    children: List[List[int]] = [[] for _ in range(len(parents))]
    for idx, parent in enumerate(parents):
        if parent >= 0:
            children[parent].append(idx)
    return children


@dataclass
class EventNode:
    incoming_length: int
    degree: int
    parent: int
    original_index: int


def _follow_unary_chain(children: Sequence[Sequence[int]], start_idx: int) -> Tuple[int, int]:
    node_idx = int(start_idx)
    skipped = 0
    while True:
        degree = len(children[node_idx])
        if degree != 1:
            return node_idx, skipped
        node_idx = int(children[node_idx][0])
        skipped += 1


def compress_k_counts_to_branch_skeleton(k_counts: Sequence[int]) -> Dict[str, List[int]]:
    values = normalize_k_counts(k_counts)
    if not values:
        raise ValueError("k_counts must not be empty.")

    parents = preorder_kcount_parent_indices(values)
    children = parents_to_children(parents)

    event_nodes: List[EventNode] = []
    original_to_event: Dict[int, int] = {}

    def visit(original_idx: int, incoming_length: int, parent_event_idx: int) -> int:
        degree = len(children[original_idx])
        if degree < 0 or degree > 2:
            raise ValueError(f"Unsupported node degree {degree} at index {original_idx}.")

        event_idx = len(event_nodes)
        event_nodes.append(
            EventNode(
                incoming_length=int(incoming_length),
                degree=int(degree),
                parent=int(parent_event_idx),
                original_index=int(original_idx),
            )
        )
        original_to_event[int(original_idx)] = int(event_idx)

        for child_idx in children[original_idx]:
            next_event_idx, skipped = _follow_unary_chain(children, int(child_idx))
            visit(next_event_idx, skipped, event_idx)
        return event_idx

    visit(0, 0, -1)

    incoming_lengths = [int(node.incoming_length) for node in event_nodes]
    degrees = [int(node.degree) for node in event_nodes]
    event_parents = [int(node.parent) for node in event_nodes]
    event_original_indices = [int(node.original_index) for node in event_nodes]

    return {
        "incoming_lengths": incoming_lengths,
        "degrees": degrees,
        "event_parents": event_parents,
        "event_original_indices": event_original_indices,
        "dense_parents": parents,
        "dense_k_counts": values,
    }


def parse_branch_skeleton(incoming_lengths: Sequence[int], degrees: Sequence[int]) -> Dict[str, List[int]]:
    lengths = [int(v) for v in incoming_lengths]
    degs = [int(v) for v in degrees]
    if len(lengths) != len(degs):
        raise ValueError("incoming_lengths and degrees must have the same length.")
    if not lengths:
        raise ValueError("Branch skeleton must contain at least one event.")

    event_parents = [-1] * len(degs)

    def parse_at(index: int, parent_event: int, is_root: bool) -> int:
        if index >= len(degs):
            raise ValueError("Skeleton sequence ended before the tree was complete.")
        degree = int(degs[index])
        length = int(lengths[index])
        if degree < 0 or degree > 2:
            raise ValueError(f"Invalid event degree {degree} at event {index}.")
        if not is_root and degree == 1:
            raise ValueError("Non-root event nodes may not have degree 1 in the compressed skeleton.")
        if length < 0:
            raise ValueError(f"Incoming branch length must be >= 0, got {length} at event {index}.")
        event_parents[index] = int(parent_event)
        next_index = index + 1
        for _ in range(degree):
            next_index = parse_at(next_index, index, False)
        return next_index

    end_index = parse_at(0, -1, True)
    if end_index != len(degs):
        raise ValueError("Skeleton sequence contains trailing events after parsing completed.")

    return {
        "incoming_lengths": lengths,
        "degrees": degs,
        "event_parents": event_parents,
    }


def expand_branch_skeleton_to_k_counts(
    incoming_lengths: Sequence[int],
    degrees: Sequence[int],
) -> Dict[str, List[int]]:
    parsed = parse_branch_skeleton(incoming_lengths, degrees)
    lengths = parsed["incoming_lengths"]
    degs = parsed["degrees"]
    event_parents = parsed["event_parents"]
    children = parents_to_children(event_parents)
    dense_k_counts: List[int] = []

    def expand_event(event_idx: int) -> None:
        dense_k_counts.append(int(degs[event_idx]))
        for child_event in children[event_idx]:
            child_event = int(child_event)
            dense_k_counts.extend([1] * int(lengths[child_event]))
            expand_event(child_event)

    expand_event(0)
    dense_parents = preorder_kcount_parent_indices(dense_k_counts)
    return {
        "incoming_lengths": lengths,
        "degrees": degs,
        "event_parents": event_parents,
        "dense_k_counts": dense_k_counts,
        "dense_parents": dense_parents,
    }


def compute_depths(parents: Sequence[int]) -> List[int]:
    depths = [0] * len(parents)
    for idx, parent in enumerate(parents):
        parent = int(parent)
        if parent >= 0:
            depths[idx] = int(depths[parent]) + 1
    return depths
