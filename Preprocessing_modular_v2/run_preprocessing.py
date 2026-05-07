import argparse
import os
import sys

# Allow running from Preprocessing_modular/ while importing sibling package Preprocessing/.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pipeline import load_config, run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run modular preprocessing pipeline.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--only", default="", help="Comma-separated list of steps to run")
    args = parser.parse_args()

    cfg = load_config(args.config)

    only_steps = None
    if args.only:
        only_steps = [s.strip() for s in args.only.split(",") if s.strip()]

    run_pipeline(cfg, only_steps=only_steps)


if __name__ == "__main__":
    main()
