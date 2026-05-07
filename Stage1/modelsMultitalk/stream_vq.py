"""
Stream-based Vector Quantization for semantic feature grouping.

Each semantic group (position, control points, knots) gets its own:
- Projection head from shared encoder
- Vector Quantizer with appropriate codebook size
- Decoder head to reconstruct that group

K-count is handled via classification (no VQ).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from Stage1.modelsMultitalk.lib.quantizer import VectorQuantizer
from Stage1.modelsMultitalk.lib.base_models import Transformer, LinearEmbedding, PositionalEncoding
from Stage1.base import BaseModel


class StreamVQAutoEncoder(BaseModel):
    """
    Stream-based VQ-VAE with semantic feature grouping.
    
    Streams:
    - position: dims 0-2 (3 values) - Centerline xyz
    - control_points: dims 3-26 (24 values) - B-Spline control points
    - knots: dims 27-38 (12 values) - B-Spline knot vector
    - k_count: dim 39 (1 value) - Classification (no VQ)
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_k_head = True  # Stream mode always uses k-head classification
        self.vq_beta = float(getattr(args, "vq_beta", 0.25))
        
        # Parse stream config
        self.streams = self._parse_streams(args)
        
        # Shared encoder
        self.encoder = StreamTransformerEncoder(args)
        
        # K-head for classification (no VQ)
        self.k_head = nn.Linear(args.hidden_size, args.k_classes)
        
        # Per-stream components
        self.stream_proj_enc = nn.ModuleDict()  # Encoder projections
        self.stream_to_embed = nn.ModuleDict()  # Projection to embed_dim
        self.stream_quantizers = nn.ModuleDict()  # Per-stream list of VQ modules (RVQ levels)
        self.stream_proj_dec = nn.ModuleDict()  # Decoder projections
        self.stream_output = nn.ModuleDict()  # Final output layers

        self.stream_level_counts = {}
        self.stream_token_keys = []
        
        for name, cfg in self.streams.items():
            hidden_dim = cfg['hidden_dim']
            n_embed_levels = cfg['n_embed_levels']
            embed_dim = cfg['embed_dim']
            output_dim = cfg['output_dim']
            levels = int(cfg.get("levels", len(n_embed_levels)))
            self.stream_level_counts[name] = levels
            
            # Encoder projection: hidden_size -> hidden_dim
            self.stream_proj_enc[name] = nn.Sequential(
                nn.Linear(args.hidden_size, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            
            # Projection to embed_dim for VQ
            self.stream_to_embed[name] = nn.Linear(hidden_dim, embed_dim)
            
            # Vector Quantizers (one per RVQ level)
            self.stream_quantizers[name] = nn.ModuleList(
                [VectorQuantizer(int(n_e), embed_dim, beta=self.vq_beta) for n_e in n_embed_levels]
            )
            
            # Decoder projection: embed_dim -> hidden_dim
            self.stream_proj_dec[name] = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            
            # Output layer: hidden_dim -> output_dim
            self.stream_output[name] = nn.Linear(hidden_dim, output_dim)

            for level_idx in range(levels):
                self.stream_token_keys.append(self._stream_level_key(name, level_idx))
        
        # Shared decoder transformer (optional, for cross-stream context)
        self.use_shared_decoder = getattr(args, 'stream_shared_decoder', True)
        if self.use_shared_decoder:
            # Compute total embed dim for decoder input
            total_embed_dim = sum(cfg['embed_dim'] for cfg in self.streams.values())
            self.decoder_proj_in = nn.Linear(total_embed_dim, args.hidden_size)
            self.decoder = StreamTransformerDecoder(args)
            self.decoder_proj_out = nn.Linear(args.hidden_size, args.in_dim - 1)  # -1 for k_count

    def _stream_level_key(self, stream_name, level_idx):
        levels = int(self.stream_level_counts.get(stream_name, 1))
        if levels <= 1:
            return stream_name
        return f"{stream_name}_l{int(level_idx)}"

    @torch.no_grad()
    def get_stream_token_keys(self, include_k_count=True):
        keys = list(self.stream_token_keys)
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
        for name in self.streams.keys():
            quantizers = self.stream_quantizers[name]
            for level_idx, quantizer in enumerate(quantizers):
                key = self._stream_level_key(name, level_idx)
                slots[key] = int(getattr(quantizer, "n_e", 0))
        return slots

    def _align_quant_valid_mask(self, quant_valid_mask, target_len, batch_size):
        """Resize [B, L] valid mask to match encoder token length."""
        if quant_valid_mask is None:
            return None
        if quant_valid_mask.dim() != 2:
            raise ValueError(f"quant_valid_mask must have shape [B, L], got {quant_valid_mask.shape}")
        if quant_valid_mask.shape[0] != batch_size:
            raise ValueError(
                f"quant_valid_mask batch size ({quant_valid_mask.shape[0]}) does not match input batch ({batch_size})"
            )
        mask = quant_valid_mask.bool()
        current_len = int(mask.shape[1])
        if current_len == int(target_len):
            return mask
        mask_f = mask.float().unsqueeze(1)
        if current_len > int(target_len):
            resized = F.adaptive_max_pool1d(mask_f, output_size=int(target_len))
        else:
            resized = F.interpolate(mask_f, size=int(target_len), mode="nearest")
        return resized.squeeze(1) > 0.5
    
    def _parse_streams(self, args):
        """Parse stream configuration from args."""
        stream_config = getattr(args, 'stream_config', None)
        
        if stream_config is None:
            # Default configuration
            return {
                'position': {
                    'input_range': (0, 3),
                    'output_dim': 3,
                    'hidden_dim': 64,
                    'n_embed_levels': [1024],
                    'embed_dim': 32,
                    'levels': 1,
                },
                'control_points': {
                    'input_range': (3, 27),
                    'output_dim': 24,
                    'hidden_dim': 256,
                    'n_embed_levels': [2048],
                    'embed_dim': 64,
                    'levels': 1,
                },
                'knots': {
                    'input_range': (27, 39),
                    'output_dim': 12,
                    'hidden_dim': 128,
                    'n_embed_levels': [512],
                    'embed_dim': 32,
                    'levels': 1,
                },
            }
        
        # Parse from config (dict format from yaml)
        streams = {}
        for name, stream_def in stream_config.items():
            input_range = stream_def['input_range']
            if isinstance(input_range, list):
                input_range = tuple(input_range)
            n_embed_cfg = stream_def.get('n_embed', 256)
            levels_cfg = stream_def.get('levels', stream_def.get('rvq_levels', None))
            if isinstance(n_embed_cfg, (list, tuple)):
                n_embed_levels = [int(v) for v in n_embed_cfg]
                levels = len(n_embed_levels) if levels_cfg is None else int(levels_cfg)
                if levels != len(n_embed_levels):
                    raise ValueError(
                        f"stream_config.{name}: levels={levels} does not match len(n_embed)={len(n_embed_levels)}"
                    )
            else:
                levels = int(levels_cfg) if levels_cfg is not None else 1
                n_embed_levels = [int(n_embed_cfg)] * max(1, levels)
            streams[name] = {
                'input_range': input_range,
                'output_dim': input_range[1] - input_range[0],
                'hidden_dim': stream_def['hidden_dim'],
                'n_embed_levels': n_embed_levels,
                'embed_dim': stream_def['embed_dim'],
                'levels': len(n_embed_levels),
            }
        return streams

    def encode(self, x, x_a=None, attn_mask=None, quant_valid_mask=None):
        """
        Encode input through shared encoder, then project to each stream.
        
        Args:
            x: [B, L, 40] input features
            attn_mask: [B, L, L] attention mask
            
        Returns:
            quant_dict: Dict of quantized features per stream {name: [B, embed_dim, L]}
            loss_dict: Dict of VQ losses per stream
            info_dict: Dict of VQ info (perplexity, indices) per stream
            k_logits: [B, L, k_classes] classification logits
        """
        # Shared encoder
        h = self.encoder(x, attn_mask=attn_mask)  # [B, L, hidden_size]
        if quant_valid_mask is None:
            quant_valid_mask = torch.any(torch.abs(x) > 1e-6, dim=-1)
        quant_valid_mask = self._align_quant_valid_mask(
            quant_valid_mask,
            target_len=h.shape[1],
            batch_size=x.shape[0],
        )
        
        # K-count classification (from full hidden state)
        k_logits = self.k_head(h)  # [B, L, k_classes]
        
        quant_dict = {}
        loss_dict = {}
        info_dict = {}
        
        for name, cfg in self.streams.items():
            # Project to stream-specific space
            h_stream = self.stream_proj_enc[name](h)  # [B, L, hidden_dim]
            
            # Project to embed_dim for VQ
            h_embed = self.stream_to_embed[name](h_stream)  # [B, L, embed_dim]

            residual = h_embed
            quant_sum = None
            stream_loss = torch.zeros((), device=h_embed.device, dtype=h_embed.dtype)

            quantizers = self.stream_quantizers[name]
            for level_idx, quantizer in enumerate(quantizers):
                quant_level, loss_level, info_level = quantizer(residual, valid_mask=quant_valid_mask)
                quant_sum = quant_level if quant_sum is None else (quant_sum + quant_level)
                stream_loss = stream_loss + loss_level
                residual = residual - quant_level.permute(0, 2, 1).contiguous()
                info_dict[self._stream_level_key(name, level_idx)] = info_level

            quant_dict[name] = quant_sum
            loss_dict[name] = stream_loss / max(1, len(quantizers))
        
        return quant_dict, loss_dict, info_dict, k_logits

    def decode(self, quant_dict, return_features=False, attn_mask=None):
        """
        Decode quantized features back to original space.
        
        Args:
            quant_dict: Dict of quantized features per stream {name: [B, embed_dim, L]}
            
        Returns:
            output: [B, L, 39] reconstructed features (without k_count)
        """
        if self.use_shared_decoder:
            # Concatenate all streams and decode together
            quant_list = []
            for name in self.streams.keys():
                q = quant_dict[name]  # [B, embed_dim, L]
                quant_list.append(q)
            
            quant_cat = torch.cat(quant_list, dim=1)  # [B, total_embed_dim, L]
            quant_cat = quant_cat.permute(0, 2, 1)  # [B, L, total_embed_dim]
            
            # Project to hidden size and decode
            h = self.decoder_proj_in(quant_cat)  # [B, L, hidden_size]
            h = self.decoder(h, attn_mask=attn_mask)  # [B, L, hidden_size]
            
            if return_features:
                output = self.decoder_proj_out(h)  # [B, L, 39]
                return output, h
            
            output = self.decoder_proj_out(h)  # [B, L, 39]
            return output
        
        else:
            # Decode each stream separately
            outputs = {}
            for name, cfg in self.streams.items():
                q = quant_dict[name]  # [B, embed_dim, L]
                q = q.permute(0, 2, 1)  # [B, L, embed_dim]
                
                h = self.stream_proj_dec[name](q)  # [B, L, hidden_dim]
                out = self.stream_output[name](h)  # [B, L, output_dim]
                outputs[name] = out
            
            # Concatenate in order
            output = torch.cat([
                outputs['position'],       # [B, L, 3]
                outputs['control_points'], # [B, L, 24]
                outputs['knots'],          # [B, L, 12]
            ], dim=-1)  # [B, L, 39]
            
            if return_features:
                # Return hidden from last stream as features
                return output, h
            
            return output

    def forward(self, x, attn_mask=None, quant_valid_mask=None):
        """
        Full forward pass.
        
        Args:
            x: [B, L, 40] input - format: [k_count, position, control_points, knots]
            attn_mask: [B, L, L] attention mask
            
        Returns:
            output: [B, L, 40] reconstructed output - same format as input
            emb_loss: Combined VQ loss
            info: Dict of VQ info per stream
            k_logits: [B, L, k_classes] classification logits
        """
        # Encode
        quant_dict, loss_dict, info_dict, k_logits = self.encode(
            x, attn_mask=attn_mask, quant_valid_mask=quant_valid_mask
        )
        
        # Decode (returns [B, L, 39] - position, control_points, knots)
        dec = self.decode(quant_dict, attn_mask=attn_mask)
        
        # Reconstruct k_count from classification
        k_pred = torch.argmax(k_logits, dim=-1, keepdim=True).float()  # [B, L, 1]
        
        # Combine: [k_count, position, control_points, knots] - same as input format!
        output = torch.cat([k_pred, dec], dim=-1)  # [B, L, 40]
        
        # Combined loss (weighted average)
        emb_loss = sum(loss_dict.values()) / len(loss_dict)
        
        return output, emb_loss, info_dict, k_logits

    @torch.no_grad()
    def get_quant(self, x, x_a=None, attn_mask=None):
        """Get quantized indices for all streams."""
        quant_dict, loss_dict, info_dict, k_logits = self.encode(x, attn_mask=attn_mask)
        
        # Extract indices from each stream
        indices = {}
        for name, info in info_dict.items():
            indices[name] = info[2]  # min_encoding_indices
        
        # Add k_count prediction
        k_pred = torch.argmax(k_logits, dim=-1)
        indices['k_count'] = k_pred
        
        return quant_dict, indices

    @torch.no_grad()
    def entry_to_feature(self, indices, zshape):
        """
        Convert indices back to features.
        
        Args:
            indices: Dict of indices per stream
            zshape: Shape hint (B, L, embed_dim)
        """
        quant_dict = {}
        
        batch_size = int(zshape[0])
        seq_len = int(zshape[1])

        for name in self.streams.keys():
            quant_sum = None
            for level_idx, quantizer in enumerate(self.stream_quantizers[name]):
                key = self._stream_level_key(name, level_idx)
                if key not in indices and level_idx == 0 and name in indices:
                    key = name
                if key not in indices:
                    raise KeyError(f"Missing stream index key '{key}' for stream '{name}'.")
                idx = indices[key].long()
                if idx.dim() == 1:
                    idx = idx.reshape(batch_size, -1)
                elif idx.dim() != 2:
                    raise ValueError(f"indices['{key}'] must be shape [B, L] or [L], got {idx.shape}")
                if idx.shape[0] != batch_size:
                    raise ValueError(
                        f"indices['{key}'] batch size {idx.shape[0]} does not match expected {batch_size}"
                    )
                if idx.shape[1] != seq_len:
                    seq_len = int(idx.shape[1])
                quant_level = quantizer.get_codebook_entry(idx.reshape(-1), shape=None)
                quant_level = quant_level.reshape(batch_size, seq_len, -1).permute(0, 2, 1)
                quant_sum = quant_level if quant_sum is None else (quant_sum + quant_level)
            quant_dict[name] = quant_sum
        
        return quant_dict

    @torch.no_grad()
    def decode_to_img(self, indices, zshape):
        """Decode from indices to full output."""
        quant_dict = self.entry_to_feature(indices, zshape)
        dec = self.decode(quant_dict)
        
        # Add k_count
        k_count = indices['k_count'].unsqueeze(-1).float()
        output = torch.cat([k_count, dec], dim=-1)
        
        return output


class StreamTransformerEncoder(nn.Module):
    """Encoder for Stream VQ-VAE."""
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        size = args.in_dim
        dim = args.hidden_size
        
        self.vertice_mapping = nn.Sequential(
            nn.Linear(size, dim),
            nn.LeakyReLU(args.neg, True)
        )
        
        if args.quant_factor == 0:
            layers = [nn.Sequential(
                nn.Conv1d(dim, dim, 5, stride=1, padding=2, padding_mode='replicate'),
                nn.LeakyReLU(args.neg, True),
                nn.InstanceNorm1d(dim, affine=args.INaffine)
            )]
        else:
            layers = [nn.Sequential(
                nn.Conv1d(dim, dim, 5, stride=2, padding=2, padding_mode='replicate'),
                nn.LeakyReLU(args.neg, True),
                nn.InstanceNorm1d(dim, affine=args.INaffine)
            )]
            for _ in range(1, args.quant_factor):
                layers += [nn.Sequential(
                    nn.Conv1d(dim, dim, 5, stride=1, padding=2, padding_mode='replicate'),
                    nn.LeakyReLU(args.neg, True),
                    nn.InstanceNorm1d(dim, affine=args.INaffine),
                    nn.MaxPool1d(2)
                )]
        
        self.squasher = nn.Sequential(*layers)
        
        dropout = getattr(args, 'dropout', 0.1)
        self.encoder_transformer = Transformer(
            in_size=args.hidden_size,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            intermediate_size=args.intermediate_size,
            dropout=dropout
        )
        
        self.encoder_pos_embedding = PositionalEncoding(args.hidden_size)
        self.encoder_linear_embedding = LinearEmbedding(args.hidden_size, args.hidden_size)
    
    def forward(self, inputs, attn_mask=None):
        if attn_mask is not None:
            if attn_mask.dim() == 4:
                mask_4d = attn_mask
            elif attn_mask.dim() == 3:
                mask_4d = attn_mask.unsqueeze(1)
            else:
                raise ValueError(f"Unsupported attn_mask shape: {attn_mask.shape}")
            if self.args.quant_factor > 0:
                reduction = 2 ** self.args.quant_factor
                mask_4d = F.max_pool2d(mask_4d, kernel_size=reduction, stride=reduction)
            mask_info = {'max_mask': mask_4d.shape[-1], 'mask_index': -1, 'mask': mask_4d}
        else:
            mask_info = {'max_mask': None, 'mask_index': -1, 'mask': None}
        
        inputs = self.vertice_mapping(inputs)
        inputs = self.squasher(inputs.permute(0, 2, 1)).permute(0, 2, 1)
        
        encoder_features = self.encoder_linear_embedding(inputs)
        encoder_features = self.encoder_pos_embedding(encoder_features)
        encoder_features = self.encoder_transformer((encoder_features, mask_info))
        
        return encoder_features


class StreamTransformerDecoder(nn.Module):
    """Decoder for Stream VQ-VAE."""
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        size = args.hidden_size
        dim = args.hidden_size
        
        dropout = getattr(args, 'dropout', 0.1)
        self.decoder_transformer = Transformer(
            in_size=args.hidden_size,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            intermediate_size=args.intermediate_size,
            dropout=dropout
        )
        
        self.decoder_pos_embedding = PositionalEncoding(args.hidden_size)
        self.decoder_linear_embedding = LinearEmbedding(args.hidden_size, args.hidden_size)
    
    def forward(self, inputs, attn_mask=None):
        if attn_mask is not None:
            if attn_mask.dim() == 4:
                mask_4d = attn_mask
            elif attn_mask.dim() == 3:
                mask_4d = attn_mask.unsqueeze(1)
            else:
                raise ValueError(f"Unsupported attn_mask shape: {attn_mask.shape}")
            mask_info = {'max_mask': mask_4d.shape[-1], 'mask_index': -1, 'mask': mask_4d}
        else:
            mask_info = {'max_mask': None, 'mask_index': -1, 'mask': None}
        
        decoder_features = self.decoder_linear_embedding(inputs)
        decoder_features = self.decoder_pos_embedding(decoder_features)
        decoder_features = self.decoder_transformer((decoder_features, mask_info))
        
        return decoder_features
