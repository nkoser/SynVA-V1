"""
Stream-based Vector Quantization V2 - with TRUE semantic separation.

Each semantic group (position, control points, knots) gets its own:
- Encoder that processes ONLY that group's input dimensions
- Vector Quantizer with appropriate codebook size  
- Decoder that reconstructs ONLY that group's output dimensions

K-count is handled via classification (no VQ).

Input format: [k_count (1), position (3), control_points (24), knots (12)] = 40 dims
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from Stage1.modelsMultitalk.lib.quantizer import VectorQuantizer
from Stage1.modelsMultitalk.lib.base_models import Transformer, LinearEmbedding, PositionalEncoding
from Stage1.base import BaseModel


class StreamEncoder(nn.Module):
    """Small encoder for a single stream's input dimensions."""
    
    def __init__(self, input_dim, hidden_dim, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Small transformer for sequence modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x, key_padding_mask=None):
        """
        Args:
            x: [B, L, input_dim]
            key_padding_mask: [B, L], True values are ignored by attention
        Returns:
            h: [B, L, hidden_dim]
        """
        h = self.input_proj(x)

        h = self.transformer(h, src_key_padding_mask=key_padding_mask)
        return h


class StreamDecoder(nn.Module):
    """Small decoder for a single stream's output dimensions."""
    
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
        """
        Args:
            x: [B, L, embed_dim]
            key_padding_mask: [B, L], True values are ignored by attention
        Returns:
            out: [B, L, output_dim]
        """
        h = self.input_proj(x)
        h = self.transformer(h, src_key_padding_mask=key_padding_mask)
        out = self.output_proj(h)
        return out


class StreamVQAutoEncoderV2(BaseModel):
    """
    Stream-based VQ-VAE V2 with TRUE semantic separation.
    
    Each stream has its own encoder/decoder that processes only its dimensions.
    
    Input format: [k_count (1), position (3), control_points (24), knots (12)] = 40 dims
    Stream indices (after removing k_count at index 0):
    - position: indices 1:4 (3 values)
    - control_points: indices 4:28 (24 values)
    - knots: indices 28:40 (12 values)
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_k_head = True
        self.vq_beta = float(getattr(args, "vq_beta", 0.25))
        
        # Parse stream config
        self.streams = self._parse_streams(args)
        self.stream_order = ['position', 'control_points', 'knots']
        
        dropout = getattr(args, 'dropout', 0.1)
        
        # K-count classification head
        # Uses a small network to classify based on ALL input features
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
        self.stream_to_embed = nn.ModuleDict()
        self.stream_quantizers = nn.ModuleDict()
        self.stream_decoders = nn.ModuleDict()
        
        for name, cfg in self.streams.items():
            input_dim = cfg['output_dim']  # input_dim = output_dim for each stream
            hidden_dim = cfg['hidden_dim']
            n_embed = cfg['n_embed']
            embed_dim = cfg['embed_dim']
            
            # Stream-specific encoder
            self.stream_encoders[name] = StreamEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=2,
                num_heads=4,
                dropout=dropout,
            )
            
            # Projection to embed_dim for VQ
            self.stream_to_embed[name] = nn.Linear(hidden_dim, embed_dim)
            
            # Vector Quantizer
            self.stream_quantizers[name] = VectorQuantizer(n_embed, embed_dim, beta=self.vq_beta)
            
            # Stream-specific decoder
            self.stream_decoders[name] = StreamDecoder(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                output_dim=input_dim,
                num_layers=2,
                num_heads=4,
                dropout=dropout,
            )
        
        # Optional: Cross-stream context layer
        self.use_cross_stream = getattr(args, 'stream_cross_attention', False)
        if self.use_cross_stream:
            total_embed = sum(cfg['embed_dim'] for cfg in self.streams.values())
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=total_embed,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )
    
    def _parse_streams(self, args):
        """Parse stream configuration from args."""
        stream_config = getattr(args, 'stream_config', None)
        
        if stream_config is None:
            # Default configuration - indices AFTER k_count (so starting from 1)
            # But we extract without k_count, so position starts at index 0 in the sliced tensor
            return {
                'position': {
                    'input_range': (1, 4),  # indices in full 40-dim input
                    'output_dim': 3,
                    'hidden_dim': 64,
                    'n_embed': 512,
                    'embed_dim': 32,
                },
                'control_points': {
                    'input_range': (4, 28),
                    'output_dim': 24,
                    'hidden_dim': 128,
                    'n_embed': 1024,
                    'embed_dim': 64,
                },
                'knots': {
                    'input_range': (28, 40),
                    'output_dim': 12,
                    'hidden_dim': 64,
                    'n_embed': 256,
                    'embed_dim': 32,
                },
            }
        
        # Parse from config (dict format from yaml)
        streams = {}
        for name, stream_def in stream_config.items():
            input_range = stream_def['input_range']
            if isinstance(input_range, list):
                input_range = tuple(input_range)
            streams[name] = {
                'input_range': input_range,
                'output_dim': input_range[1] - input_range[0],
                'hidden_dim': stream_def['hidden_dim'],
                'n_embed': stream_def['n_embed'],
                'embed_dim': stream_def['embed_dim'],
            }
        return streams

    def _split_input(self, x):
        """
        Split input tensor into stream-specific tensors.
        
        Args:
            x: [B, L, 40] - full input with k_count at index 0
            
        Returns:
            stream_inputs: Dict of {stream_name: [B, L, stream_dim]}
            k_input: [B, L, 40] - full input for k classification
        """
        stream_inputs = {}
        for name in self.stream_order:
            cfg = self.streams[name]
            start, end = cfg['input_range']
            stream_inputs[name] = x[:, :, start:end]
        
        return stream_inputs, x

    def _build_key_padding_mask(self, x=None, attn_mask=None):
        """Build [B, L] padding mask where True means padded/ignored."""
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
                # With diagonal-safe masks, padded rows typically have only self-attention active.
                mask_from_attn = valid_strength <= 1.0

        if mask_from_input is None:
            return mask_from_attn
        if mask_from_attn is None:
            return mask_from_input
        return mask_from_input | mask_from_attn

    def encode(self, x, x_a=None, attn_mask=None, quant_valid_mask=None):
        """
        Encode each stream separately.
        
        Args:
            x: [B, L, 40] input features
            attn_mask: [B, L, L] or [B, 1, L, L] attention mask
            
        Returns:
            quant_dict: Dict of quantized features per stream {name: [B, embed_dim, L]}
            loss_dict: Dict of VQ losses per stream
            info_dict: Dict of VQ info (perplexity, indices) per stream
            k_logits: [B, L, k_classes] classification logits
        """
        stream_inputs, k_input = self._split_input(x)
        key_padding_mask = self._build_key_padding_mask(x=x, attn_mask=attn_mask)
        if quant_valid_mask is None:
            if key_padding_mask is None:
                quant_valid_mask = torch.any(torch.abs(x) > 1e-6, dim=-1)
            else:
                quant_valid_mask = ~key_padding_mask
        
        # K-count classification
        k_logits = self.k_encoder(k_input)  # [B, L, k_classes]
        
        quant_dict = {}
        loss_dict = {}
        info_dict = {}
        
        for name in self.stream_order:
            stream_x = stream_inputs[name]  # [B, L, stream_dim]
            
            # Encode this stream
            h = self.stream_encoders[name](stream_x, key_padding_mask=key_padding_mask)  # [B, L, hidden_dim]
            
            # Project to embed_dim
            h_embed = self.stream_to_embed[name](h)  # [B, L, embed_dim]
            
            # Quantize
            quant, loss, info = self.stream_quantizers[name](h_embed, valid_mask=quant_valid_mask)  # quant: [B, embed_dim, L]
            
            quant_dict[name] = quant
            loss_dict[name] = loss
            info_dict[name] = info
        
        return quant_dict, loss_dict, info_dict, k_logits

    def decode(self, quant_dict, return_features=False, attn_mask=None):
        """
        Decode each stream separately.
        
        Args:
            quant_dict: Dict of quantized features per stream {name: [B, embed_dim, L]}
            
        Returns:
            output: [B, L, 39] reconstructed features (position + control_points + knots)
        """
        outputs = []
        key_padding_mask = self._build_key_padding_mask(attn_mask=attn_mask)
        
        for name in self.stream_order:
            q = quant_dict[name]  # [B, embed_dim, L]
            q = q.permute(0, 2, 1)  # [B, L, embed_dim]
            
            # Decode this stream
            out = self.stream_decoders[name](q, key_padding_mask=key_padding_mask)  # [B, L, output_dim]
            outputs.append(out)
        
        # Concatenate in order: position, control_points, knots
        output = torch.cat(outputs, dim=-1)  # [B, L, 39]
        
        if return_features:
            return output, None
        
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
        # Encode each stream
        quant_dict, loss_dict, info_dict, k_logits = self.encode(
            x, attn_mask=attn_mask, quant_valid_mask=quant_valid_mask
        )
        
        # Decode each stream
        dec = self.decode(quant_dict, attn_mask=attn_mask)  # [B, L, 39]
        
        # Reconstruct k_count from classification
        k_pred = torch.argmax(k_logits, dim=-1, keepdim=True).float()  # [B, L, 1]
        
        # Combine: [k_count, position, control_points, knots]
        output = torch.cat([k_pred, dec], dim=-1)  # [B, L, 40]
        
        # Combined loss (weighted average or sum)
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
        Convert indices back to quantized features.
        
        Args:
            indices: Dict of indices per stream {name: [B*L] or [B, L]}
            zshape: Shape hint (B, L, embed_dim)
        """
        quant_dict = {}
        B, L = zshape[0], zshape[1]
        
        for name in self.stream_order:
            idx = indices[name].long()
            quant = self.stream_quantizers[name].get_codebook_entry(
                idx.reshape(-1), shape=None
            )
            embed_dim = self.streams[name]['embed_dim']
            quant = quant.reshape(B, L, embed_dim)  # [B, L, embed_dim]
            quant = quant.permute(0, 2, 1)  # [B, embed_dim, L]
            quant_dict[name] = quant
        
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
