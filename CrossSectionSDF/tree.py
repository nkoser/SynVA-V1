"""Parse vessel tree from .npy files in pre_order_kcount format.

Column layout of each row (k=39, total 40 columns):
  [0]       k_children   (0=leaf, 1=continuation, 2=bifurcation)
  [1:4]     x, y, z      (centerline position, absolute coordinates)
  [4:12]    cx[0:8]      (B-spline control points, X axis)
  [12:20]   cy[0:8]      (B-spline control points, Y axis)
  [20:28]   cz[0:8]      (B-spline control points, Z axis)
  [28:40]   knots[0:12]  (B-spline knot vector, degree 3)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Segment:
    """Linear chain of nodes forming one vessel branch.

    At bifurcations the last node of the parent segment is duplicated as
    the first node of each child segment (overlap by 1) to ensure
    continuity.
    """
    centers: np.ndarray   # (N, 3) centerline positions
    coeffs: np.ndarray    # (N, 36) spline coefficients per node
    parent_idx: int = -1  # index of parent segment (-1 = root)


def load_tree_segments(path: str, k: int = 39) -> List[Segment]:
    """Load a .npy file and return vessel segments.

    Parameters
    ----------
    path : str
        Path to a .npy file with shape (num_nodes, 1+k).
    k : int
        Number of features per node (default 39 = 3 xyz + 24 ctrl pts + 12 knots).

    Returns
    -------
    List of Segment objects.
    """
    data = np.load(path).astype(np.float64).reshape(-1)
    seq = data.tolist()

    root, _ = _parse_node(seq, k)
    if root is None:
        return []

    segments: List[Segment] = []
    _extract_segments(root, [], [], -1, segments, k)
    return segments


# ── Internal tree node (transient, not exposed) ─────────────────────────

class _Node:
    __slots__ = ("center", "coeffs", "children")

    def __init__(self, center: np.ndarray, coeffs: np.ndarray):
        self.center = center
        self.coeffs = coeffs
        self.children: List["_Node"] = []


def _parse_node(seq: list, k: int) -> Tuple[Optional[_Node], list]:
    """Recursively parse one node and its subtree from a flat sequence."""
    if len(seq) < 1 + k:
        return None, seq

    k_children = int(round(seq.pop(0)))
    k_children = max(0, min(2, k_children))

    center = np.array([seq.pop(0), seq.pop(0), seq.pop(0)])
    coeffs = np.array([seq.pop(0) for _ in range(k - 3)])

    node = _Node(center, coeffs)
    for _ in range(k_children):
        child, seq = _parse_node(seq, k)
        if child is not None:
            node.children.append(child)

    return node, seq


def _extract_segments(
    node: _Node,
    cur_centers: list,
    cur_coeffs: list,
    parent_idx: int,
    segments: List[Segment],
    k: int,
):
    """DFS to split the tree into segments (bifurcation node shared)."""
    cur_centers.append(node.center)
    cur_coeffs.append(node.coeffs)

    n_children = len(node.children)

    if n_children == 0:
        # Leaf → flush segment
        segments.append(Segment(
            centers=np.array(cur_centers),
            coeffs=np.array(cur_coeffs),
            parent_idx=parent_idx,
        ))

    elif n_children == 1:
        # Continuation → extend
        _extract_segments(node.children[0], cur_centers, cur_coeffs,
                          parent_idx, segments, k)

    else:
        # Bifurcation → flush, then start children with overlap
        seg_idx = len(segments)
        segments.append(Segment(
            centers=np.array(cur_centers),
            coeffs=np.array(cur_coeffs),
            parent_idx=parent_idx,
        ))
        for child in node.children:
            _extract_segments(
                child,
                [node.center.copy()],
                [node.coeffs.copy()],
                seg_idx,
                segments,
                k,
            )
