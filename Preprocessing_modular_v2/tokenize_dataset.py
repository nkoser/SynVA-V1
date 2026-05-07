import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import yaml
except Exception as exc:
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from Stage1.funciones import Args
from Stage1.modelsMultitalk.stage1_vocaset import VQAutoEncoder
from Stage2_New.token_ids import (
    build_global_token_plan,
    local_to_global_interleaved_tokens,
    normalize_reserved_token_ids,
)
try:
    from Stage1.modelsMultitalk.stream_vq import StreamVQAutoEncoder
except Exception:
    StreamVQAutoEncoder = None
try:
    from Stage1.modelsMultitalk.stream_vq_v2 import StreamVQAutoEncoderV2
except Exception:
    StreamVQAutoEncoderV2 = None


K_MODES = {"pre_order_kcount", "pre_order_k", "pre_order_kdir", "pre_order_k_lr"}


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def resolve_device(device_value):
    if isinstance(device_value, int):
        device_value = f"cuda:{device_value}"
    if isinstance(device_value, str) and device_value.startswith("cuda"):
        if torch.cuda.is_available():
            print(f"Using device: {torch.cuda.get_device_name(0)}")
            return torch.device(device_value)
    print("CUDA not available. Using CPU.")
    return torch.device("cpu")


def load_state_dict(model, checkpoint_path, device, strict=True):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    else:
        state = ckpt
    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            cleaned[key[len("module."):]] = value
        else:
            cleaned[key] = value
    model.load_state_dict(cleaned, strict=strict)


def iter_files(input_dir, pattern):
    return sorted(glob.glob(os.path.join(input_dir, pattern)))


def build_model(cfg, device):
    model_cfg = cfg.get("model", {})
    params = cfg.get("params", {})
    k = int(params.get("k", 39))

    args_cfg = Args()
    args_cfg.in_dim = int(model_cfg.get("in_dim", k))
    for key, value in model_cfg.items():
        setattr(args_cfg, key, value)

    quant_mode = getattr(args_cfg, "quantization_mode", "legacy")
    if quant_mode == "stream":
        if StreamVQAutoEncoder is None:
            raise RuntimeError("StreamVQAutoEncoder is not available in this environment.")
        model = StreamVQAutoEncoder(args_cfg).to(device)
    elif quant_mode == "stream_v2":
        if StreamVQAutoEncoderV2 is None:
            raise RuntimeError("StreamVQAutoEncoderV2 is not available in this environment.")
        model = StreamVQAutoEncoderV2(args_cfg).to(device)
    else:
        model = VQAutoEncoder(args_cfg).to(device)
    model.eval()
    return model, args_cfg


def main():
    parser = argparse.ArgumentParser(description="Tokenize tree datasets with a trained Stage1 VQ-VAE.")
    parser.add_argument("--config", default="tokenize_dataset_config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    params = cfg.get("params", {})

    input_dir = paths.get("input_dir")
    output_dir = paths.get("output_dir")
    checkpoint_path = paths.get("model_checkpoint")

    if not input_dir or not output_dir or not checkpoint_path:
        raise ValueError("paths.input_dir, paths.output_dir, and paths.model_checkpoint are required.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(params.get("device", 0))
    pattern = params.get("pattern", "*.npy")
    k = int(params.get("k", 39))
    mode = params.get("mode", "pre_order")
    node_dim = k + 1 if mode in {"pre_order_kcount", "pre_order_k", "pre_order_kdir", "pre_order_k_lr"} else k
    zero_threshold = float(params.get("zero_threshold", 1e-3))
    overwrite = bool(params.get("overwrite", False))
    add_bos_eos = bool(params.get("add_bos_eos", False))
    bos_id = int(params.get("bos_id", 256))
    eos_id = int(params.get("eos_id", 256))
    pad_id = params.get("pad_id", 257)
    pad_id = None if pad_id is None else int(pad_id)
    sep_id = params.get("sep_id", 258)
    sep_id = None if sep_id is None else int(sep_id)
    strict_load = bool(params.get("strict_load", True))
    preserve_input_k_count = bool(params.get("preserve_input_k_count", mode in K_MODES))
    use_global_token_ids = bool(params.get("use_global_token_ids", False))
    global_token_base = params.get("global_token_base", None)
    if global_token_base is not None:
        global_token_base = int(global_token_base)

    model, args_cfg = build_model(cfg, device)
    load_state_dict(model, checkpoint_path, device, strict=strict_load)
    default_tokens_per_row = int(getattr(args_cfg, "face_quan_num", 1))
    if hasattr(model, "get_stream_token_keys"):
        try:
            default_tokens_per_row = int(len(model.get_stream_token_keys(include_k_count=True)))
        except Exception:
            pass

    null_id = params.get("null_id", None)
    if null_id is None:
        quant_mode = str(getattr(args_cfg, "quantization_mode", "legacy"))
        if quant_mode in {"stream", "stream_v2"}:
            max_embed = int(getattr(args_cfg, "n_embed", 0))
            stream_cfg = getattr(args_cfg, "stream_config", None)
            if isinstance(stream_cfg, dict):
                for _, stream_def in stream_cfg.items():
                    try:
                        n_embed_val = stream_def.get("n_embed", 0)
                        if isinstance(n_embed_val, (list, tuple)):
                            for v in n_embed_val:
                                max_embed = max(max_embed, int(v))
                        else:
                            max_embed = max(max_embed, int(n_embed_val))
                    except Exception:
                        continue
            null_id = max_embed + 2
        else:
            null_id = int(args_cfg.n_embed) + 1
    else:
        null_id = int(null_id)

    if null_id in (bos_id, eos_id):
        raise ValueError("null_id cannot match bos_id or eos_id.")

    ordered_slot_sizes = None
    if hasattr(model, "get_stream_slot_vocab_sizes"):
        try:
            k_classes = int(getattr(args_cfg, "k_classes", 0) or 0)
            if k_classes <= 0:
                k_classes = None
            slot_sizes = model.get_stream_slot_vocab_sizes(include_k_count=True, k_classes=k_classes)
            slot_keys = (
                list(model.get_stream_token_keys(include_k_count=True))
                if hasattr(model, "get_stream_token_keys")
                else list(slot_sizes.keys())
            )
            ordered_slot_sizes = [int(slot_sizes[k]) for k in slot_keys if k in slot_sizes]
        except Exception:
            ordered_slot_sizes = None

    global_token_plan = None
    if use_global_token_ids:
        reserved_token_ids = normalize_reserved_token_ids(bos_id, eos_id, pad_id, sep_id)
        if null_id in reserved_token_ids:
            new_null_id = max(reserved_token_ids) + 1
            print(
                f"null_id={null_id} collides with reserved ids {reserved_token_ids}; "
                f"using null_id={new_null_id} instead."
            )
            null_id = int(new_null_id)
        reserved_token_ids = normalize_reserved_token_ids(reserved_token_ids, null_id)
        if not ordered_slot_sizes:
            raise ValueError("use_global_token_ids requires stream slot vocab sizes.")
        global_token_plan = build_global_token_plan(
            ordered_slot_sizes,
            reserved_token_ids=reserved_token_ids,
            start_id=global_token_base,
        )
        print(
            "Global token IDs enabled | "
            f"start_id={global_token_plan['start_id']} | "
            f"slot_offsets={global_token_plan['offsets']} | "
            f"vocab_size={global_token_plan['vocab_size']}"
        )

    if preserve_input_k_count and mode in K_MODES:
        print("Tokenization preserves input k_count values for stage-2 tokens.")

    files = iter_files(input_dir, pattern)
    written = 0
    skipped = 0

    for file_path in files:
        name = os.path.splitext(os.path.basename(file_path))[0]
        out_path = output_dir / f"{name}.tok"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        data = np.load(file_path)
        if data.ndim == 1:
            data = data.reshape((-1, node_dim))
        if data.shape[1] != node_dim:
            raise ValueError(f"{file_path} has {data.shape[1]} features, expected {node_dim}.")

        if mode in {"pre_order_kcount", "pre_order_k", "pre_order_kdir", "pre_order_k_lr"}:
            data_attrs = data[:, 1:]
        else:
            data_attrs = data
        zero_mask = np.all(np.abs(data_attrs) <= zero_threshold, axis=1)
        tokens_per_row = int(default_tokens_per_row)
        if zero_mask.all():
            tokens = torch.full((data.shape[0] * tokens_per_row,), null_id, dtype=torch.long)
        else:
            tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                _quant, indices = model.get_quant(tensor)
            if isinstance(indices, dict):
                if preserve_input_k_count and mode in K_MODES and "k_count" in indices:
                    k_count = torch.as_tensor(
                        np.rint(data[:, 0]),
                        dtype=torch.long,
                        device=tensor.device,
                    ).view(tensor.shape[0], -1)
                    k_classes = int(getattr(args_cfg, "k_classes", 0) or 0)
                    if k_classes > 0:
                        k_count = k_count.clamp(min=0, max=k_classes - 1)
                    indices = dict(indices)
                    indices["k_count"] = k_count
                keys = []
                if hasattr(model, "get_stream_token_keys"):
                    try:
                        keys = [name for name in model.get_stream_token_keys(include_k_count=True) if name in indices]
                    except Exception:
                        keys = []
                if not keys:
                    order = ["k_count", "position", "control_points", "knots"]
                    keys = [name for name in order if name in indices]
                extra_keys = [name for name in indices.keys() if name not in keys]
                if extra_keys:
                    keys.extend(sorted(extra_keys))
                if not keys:
                    raise RuntimeError("No quantization indices found for tokenization.")
                factor_indices = [indices[name].reshape(tensor.shape[0], -1) for name in keys]
                stacked = torch.stack(factor_indices, dim=-1)  # B, L, F
                tokens = stacked.reshape(-1).detach().cpu().long()
                tokens_per_row = stacked.shape[-1]
            elif isinstance(indices, (list, tuple)):
                factor_indices = [idx.view(tensor.shape[0], -1) for idx in indices]
                stacked = torch.stack(factor_indices, dim=-1)  # B, L, F
                tokens = stacked.reshape(-1).detach().cpu().long()
                tokens_per_row = stacked.shape[-1]
            else:
                tokens = indices.view(-1).detach().cpu().long()
            if tokens.numel() % data.shape[0] != 0:
                raise ValueError(
                    f"{file_path} produced {tokens.numel()} tokens for {data.shape[0]} rows. "
                    "Check quant_factor or preprocessing."
                )
            expected_tokens_per_row = int(default_tokens_per_row)
            if tokens_per_row != expected_tokens_per_row and not isinstance(indices, (list, tuple)):
                print(
                    f"Warning: tokens per row ({tokens_per_row}) != expected ({expected_tokens_per_row})."
                )
            if zero_mask.any():
                mask = torch.from_numpy(zero_mask).repeat_interleave(tokens_per_row)
                tokens[mask] = null_id

        if add_bos_eos:
            tokens = torch.cat([torch.tensor([bos_id]), tokens, torch.tensor([eos_id])])
        if global_token_plan is not None:
            tokens = local_to_global_interleaved_tokens(
                tokens,
                eos_token=eos_id,
                slot_vocab_sizes=global_token_plan["slot_vocab_sizes"],
                token_offsets=global_token_plan["offsets"],
                passthrough_token_ids=[null_id],
            )

        torch.save(tokens, out_path)
        written += 1

    print(f"Done. Written: {written} | Skipped: {skipped}")


if __name__ == "__main__":
    main()
