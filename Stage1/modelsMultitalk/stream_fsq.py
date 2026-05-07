"""
Stream-based FSQ Auto-Encoder — replaces VQ with Finite Scalar Quantization.

Architecture matches StreamVQAutoEncoderV2 (separate encoder/decoder per stream)
but uses FSQuantizer instead of VectorQuantizer.

Key advantages:
- 100% codebook utilization (no collapse)
- No commitment/codebook loss needed
- Simpler training (only reconstruction + k-classification loss)
- 1 token per stream, 3 tokens total per node (position, control_points, knots)

Input format: [k_count (1), position (3), control_points (24), knots (12)] = 40 dims
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from Stage1.modelsMultitalk.lib.fsq_quantizer import FSQuantizer
from Stage1.modelsMultitalk.lib.base_models import Transformer, LinearEmbedding, PositionalEncoding
from Stage1.base import BaseModel


# ---------------------------------------------------------------------------
# Encoder / Decoder building blocks (same as stream_vq_v2)
# ---------------------------------------------------------------------------

class StreamEncoder(nn.Module):
    """Small transformer encoder for a single stream's input dimensions."""

    def __init__(self, input_dim, hidden_dim, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, key_padding_mask=None):
        h = self.input_proj(x)
        h = self.transformer(h, src_key_padding_mask=key_padding_mask)
        return h


class StreamDecoder(nn.Module):
    """Small transformer decoder for a single stream's output dimensions."""

    def __init__(self, embed_dim, hidden_dim, output_dim, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, key_padding_mask=None):
        h = self.input_proj(x)
        h = self.transformer(h, src_key_padding_mask=key_padding_mask)
        return self.output_proj(h)


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class StreamFSQAutoEncoder(BaseModel):
    """
    Stream-based Auto-Encoder with Finite Scalar Quantization.

    Per stream:
        Encoder → Linear → FSQ (low-dim quantization) → Decoder

    FSQ levels define both the quantization dimensionality and codebook size:
        position:       [8, 6, 5]       → 240 codes,  3 FSQ dims
        control_points: [8, 5, 5, 5]    → 1000 codes, 4 FSQ dims
        knots:          [5, 5, 5]       → 125 codes,  3 FSQ dims
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_k_head = True

        # Parse stream config
        self.streams = self._parse_streams(args)
        self.stream_order = list(self.streams.keys())

        dropout = float(getattr(args, "dropout", 0.1))
        enc_layers = int(getattr(args, "enc_layers", 2))
        dec_layers = int(getattr(args, "dec_layers", 2))
        enc_heads = int(getattr(args, "enc_heads", 4))
        dec_heads = int(getattr(args, "dec_heads", 4))

        # K-count classification head (no quantization, same as V2)
        k_hidden = 128
        self.k_encoder = nn.Sequential(
            nn.Linear(args.in_dim, k_hidden),
            nn.LayerNorm(k_hidden),
            nn.GELU(),
            nn.Linear(k_hidden, k_hidden),
            nn.GELU(),
            nn.Linear(k_hidden, args.k_classes),
        )

        # Per-stream components
        self.stream_encoders = nn.ModuleDict()
        self.stream_to_fsq = nn.ModuleDict()      # hidden_dim → FSQ dim
        self.stream_quantizers = nn.ModuleDict()   # FSQuantizer
        self.stream_decoders = nn.ModuleDict()

        for name, cfg in self.streams.items():
            input_dim = cfg["output_dim"]
            hidden_dim = cfg["hidden_dim"]
            fsq_levels = cfg["fsq_levels"]
            fsq_dim = len(fsq_levels)

            self.stream_encoders[name] = StreamEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=enc_layers,
                num_heads=enc_heads,
                dropout=dropout,
            )

            # Project from hidden_dim → fsq_dim (very low: 3-5)
            self.stream_to_fsq[name] = nn.Linear(hidden_dim, fsq_dim)

            # FSQ quantizer
            self.stream_quantizers[name] = FSQuantizer(fsq_levels)

            # Decoder: fsq_dim → hidden_dim → output_dim
            self.stream_decoders[name] = StreamDecoder(
                embed_dim=fsq_dim,
                hidden_dim=hidden_dim,
                output_dim=input_dim,
                num_layers=dec_layers,
                num_heads=dec_heads,
                dropout=dropout,
            )

        # Print model summary
        total_codes = {n: cfg["n_codes"] for n, cfg in self.streams.items()}
        print(f"StreamFSQAutoEncoder: streams={list(self.streams.keys())}")
        for name, cfg in self.streams.items():
            print(
                f"  {name}: input_dim={cfg['output_dim']}, hidden_dim={cfg['hidden_dim']}, "
                f"fsq_levels={cfg['fsq_levels']}, n_codes={cfg['n_codes']}, fsq_dim={len(cfg['fsq_levels'])}"
            )

    # ------------------------------------------------------------------
    # Config parsing
    # ------------------------------------------------------------------

    def _parse_streams(self, args):
        stream_config = getattr(args, "stream_config", None)

        if stream_config is None:
            # Sensible defaults
            return {
                "position": {
                    "input_range": (1, 4),
                    "output_dim": 3,
                    "hidden_dim": 96,
                    "fsq_levels": [8, 6, 5],  # 240 codes
                    "n_codes": 240,
                },
                "control_points": {
                    "input_range": (4, 28),
                    "output_dim": 24,
                    "hidden_dim": 192,
                    "fsq_levels": [8, 5, 5, 5],  # 1000 codes
                    "n_codes": 1000,
                },
                "knots": {
                    "input_range": (28, 40),
                    "output_dim": 12,
                    "hidden_dim": 96,
                    "fsq_levels": [5, 5, 5],  # 125 codes
                    "n_codes": 125,
                },
            }

        streams = {}
        for name, sdef in stream_config.items():
            input_range = sdef["input_range"]
            if isinstance(input_range, list):
                input_range = tuple(input_range)
            fsq_levels = [int(l) for l in sdef["fsq_levels"]]
            from functools import reduce
            n_codes = reduce(lambda a, b: a * b, fsq_levels)
            streams[name] = {
                "input_range": input_range,
                "output_dim": input_range[1] - input_range[0],
                "hidden_dim": int(sdef["hidden_dim"]),
                "fsq_levels": fsq_levels,
                "n_codes": n_codes,
            }
        return streams

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_input(self, x):
        stream_inputs = {}
        for name in self.stream_order:
            start, end = self.streams[name]["input_range"]
            stream_inputs[name] = x[:, :, start:end]
        return stream_inputs, x

    def _build_key_padding_mask(self, x=None, attn_mask=None):
        mask_from_input = None
        if x is not None:
            mask_from_input = torch.all(torch.abs(x) <= 1e-6, dim=-1)

        mask_from_attn = None
        if attn_mask is not None:
            pair_mask = attn_mask
            if pair_mask.dim() == 4:
                pair_mask = pair_mask[:, 0]
            if pair_mask.dim() == 3:
                valid_strength = pair_mask.float().sum(dim=-1)
                mask_from_attn = valid_strength <= 1.0

        if mask_from_input is None:
            return mask_from_attn
        if mask_from_attn is None:
            return mask_from_input
        return mask_from_input | mask_from_attn

    # ------------------------------------------------------------------
    # Token key / vocab helpers (for extraction compatibility)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_stream_token_keys(self, include_k_count=True):
        keys = list(self.stream_order)
        if include_k_count:
            return ["k_count"] + keys
        return keys

    @torch.no_grad()
    def get_stream_slot_vocab_sizes(self, include_k_count=True, k_classes=None):
        slots = {}
        if include_k_count:
            if k_classes is None:
                k_classes = int(getattr(self.args, "k_classes", 3))
            slots["k_count"] = int(max(1, k_classes))
        for name in self.stream_order:
            slots[name] = self.streams[name]["n_codes"]
        return slots

    # ------------------------------------------------------------------
    # Encode / Decode / Forward
    # ------------------------------------------------------------------

    def encode(self, x, x_a=None, attn_mask=None, quant_valid_mask=None):
        stream_inputs, k_input = self._split_input(x)
        key_padding_mask = self._build_key_padding_mask(x=x, attn_mask=attn_mask)

        if quant_valid_mask is None:
            if key_padding_mask is None:
                quant_valid_mask = torch.any(torch.abs(x) > 1e-6, dim=-1)
            else:
                quant_valid_mask = ~key_padding_mask

        k_logits = self.k_encoder(k_input)

        quant_dict = {}
        loss_dict = {}
        info_dict = {}

        for name in self.stream_order:
            stream_x = stream_inputs[name]

            # Encode
            h = self.stream_encoders[name](stream_x, key_padding_mask=key_padding_mask)

            # Project to FSQ dimension
            h_fsq = self.stream_to_fsq[name](h)  # [B, L, fsq_dim]

            # Quantize via FSQ
            quant, loss, info = self.stream_quantizers[name](h_fsq, valid_mask=quant_valid_mask)

            quant_dict[name] = quant   # [B, fsq_dim, L]
            loss_dict[name] = loss     # scalar 0 for FSQ
            info_dict[name] = info     # (perplexity, None, indices)

        return quant_dict, loss_dict, info_dict, k_logits

    def decode(self, quant_dict, return_features=False, attn_mask=None):
        outputs = []
        key_padding_mask = self._build_key_padding_mask(attn_mask=attn_mask)

        for name in self.stream_order:
            q = quant_dict[name].permute(0, 2, 1)  # [B, L, fsq_dim]
            out = self.stream_decoders[name](q, key_padding_mask=key_padding_mask)
            outputs.append(out)

        output = torch.cat(outputs, dim=-1)  # [B, L, 39]
        if return_features:
            return output, None
        return output

    def forward(self, x, attn_mask=None, quant_valid_mask=None):
        quant_dict, loss_dict, info_dict, k_logits = self.encode(
            x, attn_mask=attn_mask, quant_valid_mask=quant_valid_mask,
        )
        dec = self.decode(quant_dict, attn_mask=attn_mask)

        k_pred = torch.argmax(k_logits, dim=-1, keepdim=True).float()
        output = torch.cat([k_pred, dec], dim=-1)  # [B, L, 40]

        # FSQ loss is always 0, but we return it for interface compatibility
        emb_loss = sum(loss_dict.values()) / max(1, len(loss_dict))

        return output, emb_loss, info_dict, k_logits

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_quant(self, x, x_a=None, attn_mask=None):
        """Get quantized indices for all streams."""
        quant_dict, loss_dict, info_dict, k_logits = self.encode(x, attn_mask=attn_mask)

        indices = {}
        for name, info in info_dict.items():
            indices[name] = info[2]  # min_encoding_indices → [N, 1]

        k_pred = torch.argmax(k_logits, dim=-1)
        indices["k_count"] = k_pred

        return quant_dict, indices

    @torch.no_grad()
    def entry_to_feature(self, indices, zshape):
        """Convert indices back to quantized features."""
        quant_dict = {}
        B, L = int(zshape[0]), int(zshape[1])

        for name in self.stream_order:
            idx = indices[name].long().reshape(-1)
            z_q = self.stream_quantizers[name].get_codebook_entry(idx, shape=None)
            fsq_dim = len(self.streams[name]["fsq_levels"])
            z_q = z_q.reshape(B, L, fsq_dim).permute(0, 2, 1)
            quant_dict[name] = z_q

        return quant_dict

    @torch.no_grad()
    def decode_to_img(self, indices, zshape):
        """Decode from indices to full output."""
        quant_dict = self.entry_to_feature(indices, zshape)
        dec = self.decode(quant_dict)

        k_count = indices["k_count"]
        if k_count.dim() == 1:
            k_count = k_count.unsqueeze(-1)
        elif k_count.dim() == 2 and k_count.shape[-1] != 1:
            k_count = k_count.unsqueeze(-1)
        output = torch.cat([k_count.float(), dec], dim=-1)
        return output
