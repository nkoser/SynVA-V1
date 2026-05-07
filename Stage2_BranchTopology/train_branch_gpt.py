from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, get_scheduler
from tqdm import tqdm

try:
    import yaml
except Exception as exc:
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

try:
    import wandb
except Exception:
    wandb = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from Stage1.base.utilities import AverageMeter
from Stage2_BranchTopology.dataset import TokenDataset, collate_tokens


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


def create_attention_mask(batch: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    return (batch != int(pad_token_id)).long()


def create_labels(batch: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    labels = batch.clone()
    labels[labels == int(pad_token_id)] = -100
    return labels


def build_loader(folder_path: str, batch_size: int, pad_token_id: int, shuffle: bool, limit=None):
    dataset = TokenDataset(folder_path, limit=limit)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        collate_fn=lambda batch: collate_tokens(batch, pad_token_id=int(pad_token_id)),
    )
    return dataset, loader


def run_epoch(loader, model, optimizer, scheduler, device, pad_token_id, train: bool):
    meter = AverageMeter()
    if train:
        model.train()
    else:
        model.eval()

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc="train" if train else "val", leave=False):
            batch = batch.to(device)
            attention_mask = create_attention_mask(batch, pad_token_id).to(device)
            labels = create_labels(batch, pad_token_id).to(device)
            outputs = model(input_ids=batch, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            meter.update(float(loss.item()), int(batch.shape[0]))
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            del outputs, loss, batch, attention_mask, labels
            gc.collect()
    return float(meter.avg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GPT-2 on branch-skeleton topology tokens.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    params = cfg.get("params", {})
    model_cfg = cfg.get("model", {})
    wandb_cfg = cfg.get("wandb", {})

    metadata = load_metadata(paths["metadata_path"])
    vocab = metadata["vocab"]
    pad_token_id = int(vocab["pad_token_id"])
    bos_token_id = int(vocab["bos_token_id"])
    eos_token_id = int(vocab["eos_token_id"])
    vocab_size = int(vocab["vocab_size"])

    train_dir = paths.get("train_dir") or metadata["paths"]["train_token_dir"]
    val_dir = paths.get("val_dir") or metadata["paths"].get("val_token_dir")
    output_dir = paths.get("output_dir")
    best_model_dir = paths.get("best_model_dir", os.path.join(output_dir, "best-gpt2"))
    if not train_dir or not output_dir:
        raise ValueError("paths.output_dir and a train token directory are required.")

    seed = int(params.get("seed", 12))
    seed_all(seed)
    device = resolve_device(params.get("device", 0))

    batch_size = int(params.get("batch_size", 16))
    train_dataset, train_loader = build_loader(
        train_dir,
        batch_size=batch_size,
        pad_token_id=pad_token_id,
        shuffle=bool(params.get("shuffle", True)),
        limit=params.get("train_limit"),
    )
    val_dataset = None
    val_loader = None
    if val_dir:
        val_dataset, val_loader = build_loader(
            val_dir,
            batch_size=batch_size,
            pad_token_id=pad_token_id,
            shuffle=False,
            limit=params.get("val_limit"),
        )

    max_train_len = max((len(seq) for seq in train_dataset), default=0)
    max_val_len = max((len(seq) for seq in val_dataset), default=0) if val_dataset is not None else 0
    max_positions = max(int(params.get("max_positions", 0)), max_train_len, max_val_len)
    if max_positions <= 0:
        raise ValueError("Unable to infer a valid max_positions value.")

    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=vocab_size,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            n_positions=max_positions,
            n_ctx=max_positions,
            n_embd=int(model_cfg.get("n_embd", 256)),
            n_layer=int(model_cfg.get("n_layer", 4)),
            n_head=int(model_cfg.get("n_head", 4)),
            resid_pdrop=float(model_cfg.get("resid_pdrop", 0.1)),
            attn_pdrop=float(model_cfg.get("attn_pdrop", 0.1)),
            embd_pdrop=float(model_cfg.get("embd_pdrop", 0.1)),
        )
    )
    model.to(device)

    lr = float(params.get("lr", 1e-4))
    epochs = int(params.get("epochs", 200))
    warmup_steps = int(params.get("warmup_steps", 0))
    weight_decay = float(params.get("weight_decay", 0.01))
    scheduler_name = str(params.get("scheduler", "linear"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_scheduler(
        scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(1, len(train_loader) * epochs),
    )

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config_used.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    history_path = os.path.join(output_dir, "history.csv")
    with open(history_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_ppl",
                "val_loss",
                "val_ppl",
                "best_metric",
                "improved",
            ]
        )

    wandb_enabled = bool(wandb_cfg.get("enabled", False))
    if wandb_enabled:
        if wandb is None:
            raise RuntimeError("wandb is enabled but not installed.")
        wandb.init(
            project=wandb_cfg.get("project", "VesselGPT_BranchTopology"),
            entity=wandb_cfg.get("entity"),
            mode=wandb_cfg.get("mode", "online"),
            name=wandb_cfg.get("run_name"),
            config=cfg,
        )

    best_metric_name = str(params.get("best_metric", "val")).lower()
    best_metric = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(train_loader, model, optimizer, scheduler, device, pad_token_id, train=True)
        val_loss = None
        if val_loader is not None:
            val_loss = run_epoch(val_loader, model, optimizer, scheduler, device, pad_token_id, train=False)

        metric = train_loss if (best_metric_name == "train" or val_loss is None) else val_loss
        improved = metric < best_metric
        if improved:
            best_metric = metric
            os.makedirs(best_model_dir, exist_ok=True)
            model.save_pretrained(best_model_dir)
            with open(os.path.join(best_model_dir, "branch_topology_metadata.json"), "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2)

        train_ppl = math.exp(min(train_loss, 20.0))
        val_ppl = math.exp(min(val_loss, 20.0)) if val_loss is not None else None
        msg = f"epoch={epoch}/{epochs} train_loss={train_loss:.6f} train_ppl={train_ppl:.3f}"
        if val_loss is not None:
            msg += f" val_loss={val_loss:.6f} val_ppl={val_ppl:.3f}"
        if improved:
            msg += " best=1"
        print(msg)
        with open(history_path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    int(epoch),
                    float(train_loss),
                    float(train_ppl),
                    "" if val_loss is None else float(val_loss),
                    "" if val_ppl is None else float(val_ppl),
                    float(best_metric),
                    int(improved),
                ]
            )

        if wandb_enabled:
            payload = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_ppl": train_ppl,
                "best_metric": best_metric,
            }
            if val_loss is not None:
                payload["val_loss"] = val_loss
                payload["val_ppl"] = val_ppl
            wandb.log(payload)

    if wandb_enabled:
        wandb.finish()

    print(f"Best model saved to: {best_model_dir}")


if __name__ == "__main__":
    main()
