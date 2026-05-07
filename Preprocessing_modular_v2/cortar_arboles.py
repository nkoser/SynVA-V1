import argparse
import os
import sys
from glob import glob

import numpy as np

try:
    import yaml
except Exception as exc:
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tree_functions import deserialize, serialize_pre_order_k, serialize_pre_order_kcount, serialize_pre_order_kdir


class TreeDepthCutter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.paths = cfg.get("paths", {})
        self.params = cfg.get("params", {})

    @classmethod
    def load_config(cls, path):
        config_path = os.path.abspath(path)
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg or {}

    @staticmethod
    def _ensure_dir(path):
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def remove_below_level(node, level, current_level=1):
        if not node:
            return None

        node.left = TreeDepthCutter.remove_below_level(node.left, level, current_level + 1)
        node.right = TreeDepthCutter.remove_below_level(node.right, level, current_level + 1)

        if current_level > level:
            return None

        return node

    def _iter_files(self):
        input_dir = self.paths.get("input_dir")
        pattern = self.params.get("glob", "*.npy")
        if not input_dir:
            raise ValueError("paths.input_dir is required")
        return sorted(glob(os.path.join(input_dir, pattern)))

    def process_file(self, file_path):
        max_depth = int(self.params.get("max_depth", 20))
        k = int(self.params.get("k", 39))
        input_mode = self.params.get("input_mode", "post_order")
        output_mode = self.params.get("output_mode", "pre_order")
        output_dir = self.paths.get("output_dir")
        overwrite = bool(self.params.get("overwrite", False))

        if not output_dir:
            raise ValueError("paths.output_dir is required")

        self._ensure_dir(output_dir)
        out_path = os.path.join(output_dir, os.path.basename(file_path))
        if os.path.exists(out_path) and not overwrite:
            return "skip", out_path, None

        data = np.load(file_path)
        node_dim = k + 1 if input_mode in {"pre_order_kcount", "pre_order_k", "pre_order_kdir", "pre_order_k_lr"} else k
        if data.ndim == 1:
            data = data.reshape((-1, node_dim))
        serial_tree = list(data.flatten())
        tree = deserialize(serial_tree, mode=input_mode, k=k)
        decoded = self.remove_below_level(tree, max_depth)
        if output_mode in {"pre_order_kcount", "pre_order_k"}:
            vec = serialize_pre_order_kcount(decoded, k=k)
            node_dim = k + 1
        elif output_mode in {"pre_order_kdir", "pre_order_k_lr"}:
            vec = serialize_pre_order_kdir(decoded, k=k)
            node_dim = k + 1
        else:
            vec = serialize_pre_order_k(decoded, k=k)
            node_dim = k

        np.save(out_path, np.array(vec, dtype=np.float32).reshape((-1, node_dim)))
        return "ok", out_path, len(vec) / node_dim

    def run(self):
        files = self._iter_files()
        total = 0
        written = 0
        log_every = int(self.params.get("log_every", 1))

        for idx, file_path in enumerate(files, start=1):
            total += 1
            status, out_path, length = self.process_file(file_path)

            if status == "ok":
                written += 1
                if log_every > 0 and (idx % log_every == 0):
                    print(f"[{idx}/{len(files)}] saved {out_path} (len={length})")
            else:
                if log_every > 0 and (idx % log_every == 0):
                    print(f"[{idx}/{len(files)}] skip {out_path}")

        print(f"done: {written}/{total} written")


def main():
    parser = argparse.ArgumentParser(description="Trim tree depth from .npy serialized trees.")
    parser.add_argument("--config", default="cortar_arboles_config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = TreeDepthCutter.load_config(args.config)
    cutter = TreeDepthCutter(cfg)
    cutter.run()


if __name__ == "__main__":
    main()
