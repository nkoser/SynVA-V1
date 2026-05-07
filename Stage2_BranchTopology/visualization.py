from __future__ import annotations

import os
from typing import Dict, List, Sequence


def _parents_to_children(parents: Sequence[int]) -> List[List[int]]:
    children: List[List[int]] = [[] for _ in range(len(parents))]
    for idx, parent in enumerate(parents):
        parent = int(parent)
        if parent >= 0:
            children[parent].append(int(idx))
    return children


def _assign_event_x(children: Sequence[Sequence[int]]):
    x_positions = [0.0] * len(children)
    next_leaf_x = 0.0

    def visit(node_idx: int) -> float:
        nonlocal next_leaf_x
        kids = [int(v) for v in children[node_idx]]
        if not kids:
            x_positions[node_idx] = float(next_leaf_x)
            next_leaf_x += 1.0
            return x_positions[node_idx]
        xs = [visit(child_idx) for child_idx in kids]
        x_positions[node_idx] = float(sum(xs) / len(xs))
        return x_positions[node_idx]

    if children:
        visit(0)
    return x_positions


def _assign_event_y(parents: Sequence[int], incoming_lengths: Sequence[int]):
    y_positions = [0.0] * len(parents)
    for idx in range(1, len(parents)):
        parent_idx = int(parents[idx])
        branch_steps = int(incoming_lengths[idx]) + 1
        y_positions[idx] = float(y_positions[parent_idx] - branch_steps)
    return y_positions


def plot_branch_topology(
    incoming_lengths: Sequence[int],
    degrees: Sequence[int],
    event_parents: Sequence[int],
    out_path: str,
    title: str = "",
    show_intermediate_dense_nodes: bool = True,
    annotate_lengths: bool = True,
):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if not event_parents:
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    children = _parents_to_children(event_parents)
    x_event = _assign_event_x(children)
    y_event = _assign_event_y(event_parents, incoming_lengths)

    fig, ax = plt.subplots(figsize=(9, 7))

    for idx in range(1, len(event_parents)):
        parent_idx = int(event_parents[idx])
        x0, y0 = x_event[parent_idx], y_event[parent_idx]
        x1, y1 = x_event[idx], y_event[idx]
        ax.plot([x0, x1], [y0, y1], color="#6b7280", linewidth=1.2, alpha=0.85, zorder=1)

        branch_steps = int(incoming_lengths[idx]) + 1
        if show_intermediate_dense_nodes and branch_steps > 1:
            xs = []
            ys = []
            for step in range(1, branch_steps):
                frac = float(step) / float(branch_steps)
                xs.append((1.0 - frac) * x0 + frac * x1)
                ys.append((1.0 - frac) * y0 + frac * y1)
            if xs:
                ax.scatter(xs, ys, s=12, c="#9ca3af", alpha=0.95, zorder=2)

        if annotate_lengths:
            xm = 0.5 * (x0 + x1)
            ym = 0.5 * (y0 + y1)
            ax.text(
                xm,
                ym + 0.12,
                f"{int(incoming_lengths[idx])}",
                fontsize=8,
                ha="center",
                va="bottom",
                color="#374151",
            )

    root_x = x_event[0]
    root_y = y_event[0]
    ax.scatter([root_x], [root_y], s=180, c="#16a34a", marker="*", zorder=4)

    leaf_x = []
    leaf_y = []
    unary_x = []
    unary_y = []
    bif_x = []
    bif_y = []
    for idx, degree in enumerate(degrees):
        degree = int(degree)
        if idx == 0:
            continue
        if degree == 0:
            leaf_x.append(x_event[idx])
            leaf_y.append(y_event[idx])
        elif degree == 1:
            unary_x.append(x_event[idx])
            unary_y.append(y_event[idx])
        else:
            bif_x.append(x_event[idx])
            bif_y.append(y_event[idx])

    if bif_x:
        ax.scatter(bif_x, bif_y, s=55, c="#ea580c", edgecolors="white", linewidths=0.8, zorder=4)
    if unary_x:
        ax.scatter(unary_x, unary_y, s=45, c="#2563eb", edgecolors="white", linewidths=0.8, zorder=4)
    if leaf_x:
        ax.scatter(leaf_x, leaf_y, s=40, c="#111827", edgecolors="white", linewidths=0.8, zorder=4)

    ax.set_xlabel("branch order")
    ax.set_ylabel("dense depth")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_aspect("auto")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

