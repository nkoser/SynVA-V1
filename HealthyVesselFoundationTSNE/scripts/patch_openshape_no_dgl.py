"""Patch OpenShape demo support to avoid the DGL runtime dependency.

OpenShape's public demo support package uses DGL only for farthest-point
sampling in `openshape/pointnet_util.py`. That same file already contains a
pure PyTorch fallback implementation immediately after the DGL call, but the
fallback is unreachable because the DGL call returns first.

This helper removes the top-level `import dgl.geometry` and the early
`return dgl.geometry.farthest_point_sampler(...)` line, so the existing
PyTorch implementation is used instead. This is useful on newer GPUs where
old DGL wheels are hard to match to the installed PyTorch/CUDA stack.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def patch_pointnet_util(repo_path: Path) -> Path:
    target = repo_path / "openshape" / "pointnet_util.py"
    if not target.exists():
        raise FileNotFoundError(f"Could not find {target}")

    text = target.read_text()
    original = text

    lines = []
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped == "import dgl.geometry":
            continue
        if stripped.startswith("return dgl.geometry.farthest_point_sampler("):
            continue
        lines.append(line)
    text = "".join(lines)

    if text == original:
        print(f"No changes needed: {target}")
        return target

    backup = target.with_suffix(target.suffix + ".bak")
    if not backup.exists():
        backup.write_text(original)
    target.write_text(text)
    print(f"Patched: {target}")
    print(f"Backup:  {backup}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "repo_path",
        type=Path,
        help="Path to the cloned openshape-demo-support repository.",
    )
    args = parser.parse_args()
    patch_pointnet_util(args.repo_path.expanduser().resolve())


if __name__ == "__main__":
    main()
