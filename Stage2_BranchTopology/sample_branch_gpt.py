from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import torch
from transformers import GPT2LMHeadModel, LogitsProcessorList

try:
    import yaml
except Exception as exc:
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from Stage2_BranchTopology.constraints import BranchSkeletonConstraintLogitsProcessor
from Stage2_BranchTopology.representation import compute_depths, expand_branch_skeleton_to_k_counts
from Stage2_BranchTopology.vocab import decode_branch_skeleton
from Stage2_BranchTopology.visualization import plot_branch_topology


def load_config(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    return cfg or {}


def load_metadata(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_value):
    if isinstance(device_value, int):
        device_value = f"cuda:{device_value}"
    if isinstance(device_value, str) and device_value.startswith("cuda") and torch.cuda.is_available():
        print(f"Using device: {torch.cuda.get_device_name(0)}")
        return torch.device(device_value)
    print("CUDA not available. Using CPU.")
    return torch.device("cpu")


def save_sample(output_dir: str, sample_idx: int, tokens: torch.Tensor, payload: Dict[str, object]) -> None:
    prefix = f"sample_{sample_idx:04d}"
    torch.save(tokens.cpu(), os.path.join(output_dir, prefix + ".tok"))
    with open(os.path.join(output_dir, prefix + ".json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample compressed branch-skeleton topology with constrained GPT decoding.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    params = cfg.get("params", {})

    metadata = load_metadata(paths["metadata_path"])
    vocab = metadata["vocab"]
    model_dir = paths["model_dir"]
    output_dir = paths["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    visualize = bool(params.get("visualize", True))
    visualize_dir = params.get("visualize_dir", os.path.join(output_dir, "plots"))
    if visualize:
        os.makedirs(visualize_dir, exist_ok=True)

    seed_all(int(params.get("seed", 12)))
    device = resolve_device(params.get("device", 0))

    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    bos_token_id = int(vocab["bos_token_id"])
    eos_token_id = int(vocab["eos_token_id"])
    pad_token_id = int(vocab["pad_token_id"])
    max_new_tokens = int(params.get("max_new_tokens", 256))
    num_samples = int(params.get("num_samples", 8))
    temperature = float(params.get("temperature", 1.0))
    top_k = int(params.get("top_k", 0))
    top_p = float(params.get("top_p", 1.0))

    logits_processor = LogitsProcessorList([BranchSkeletonConstraintLogitsProcessor(vocab)])

    input_ids = torch.full((num_samples, 1), bos_token_id, dtype=torch.long, device=device)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "logits_processor": logits_processor,
    }
    if top_k > 0:
        generation_kwargs["top_k"] = top_k

    outputs = model.generate(input_ids=input_ids, **generation_kwargs)

    for sample_idx in range(outputs.shape[0]):
        tokens = outputs[sample_idx].detach().cpu().long().view(-1)
        try:
            lengths, degrees = decode_branch_skeleton(tokens.tolist(), vocab)
            expanded = expand_branch_skeleton_to_k_counts(lengths, degrees)
            event_depths = compute_depths(expanded["event_parents"])
            dense_depths = compute_depths(expanded["dense_parents"])
            payload = {
                "valid": True,
                "incoming_lengths": [int(v) for v in lengths],
                "degrees": [int(v) for v in degrees],
                "event_parents": [int(v) for v in expanded["event_parents"]],
                "event_depths": [int(v) for v in event_depths],
                "dense_k_counts": [int(v) for v in expanded["dense_k_counts"]],
                "dense_parents": [int(v) for v in expanded["dense_parents"]],
                "dense_depths": [int(v) for v in dense_depths],
            }
            if visualize:
                plot_branch_topology(
                    lengths,
                    degrees,
                    expanded["event_parents"],
                    out_path=os.path.join(visualize_dir, f"sample_{sample_idx:04d}.png"),
                    title=f"branch topology {sample_idx}",
                    show_intermediate_dense_nodes=bool(params.get("show_intermediate_dense_nodes", True)),
                    annotate_lengths=bool(params.get("annotate_lengths", True)),
                )
        except Exception as exc:
            payload = {
                "valid": False,
                "error": str(exc),
            }
        save_sample(output_dir, sample_idx, tokens, payload)

    print(f"Saved {outputs.shape[0]} samples to: {output_dir}")


if __name__ == "__main__":
    main()
