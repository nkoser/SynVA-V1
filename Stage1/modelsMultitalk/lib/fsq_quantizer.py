"""
Finite Scalar Quantization (FSQ) — drop-in replacement for VectorQuantizer.

Reference: Mentzer et al., "Finite Scalar Quantization: VQ-VAE Made Simple", ICLR 2024.

Key properties vs VQ:
- No learned codebook (implicit codebook from level combinations)
- No commitment loss or codebook loss needed
- 100% codebook utilization guaranteed (no collapse)
- Each encoder dimension is independently rounded to discrete levels

Usage:
    fsq = FSQuantizer(levels=[8, 6, 5])   # 240 codes, 3 dims
    z_q, loss, (perplexity, None, indices) = fsq(z)
    # z: [B, L, 3],  z_q: [B, 3, L],  indices: [B*L, 1]
"""

import math
from functools import reduce

import torch
import torch.nn as nn


def _round_ste(z: torch.Tensor) -> torch.Tensor:
    """Round with straight-through estimator."""
    return z + (torch.round(z) - z).detach()


class FSQuantizer(nn.Module):
    """
    Finite Scalar Quantizer.

    Args:
        levels: list of ints, e.g. [8, 6, 5].
            Each entry is the number of discrete levels for that dimension.
            Total codebook size = prod(levels).
            The quantization range for dim i is
                {-floor((L_i-1)/2), ..., +floor((L_i-1)/2)}  for odd L_i
            or  {-(L_i-1)/2, ..., +(L_i-1)/2}                for even L_i

    Interface is compatible with VectorQuantizer:
        forward(z, valid_mask=None)
            z: [B, L, dim]
            returns: (z_q [B, dim, L], loss, (perplexity, None, indices [N,1]))
    """

    def __init__(self, levels):
        super().__init__()
        if not levels or any(int(l) < 2 for l in levels):
            raise ValueError(f"FSQ levels must be a list of ints >= 2, got {levels}")

        _levels = torch.tensor([int(l) for l in levels], dtype=torch.int64)
        self.register_buffer("_levels", _levels)

        self.dim = len(levels)
        self.n_codes = int(reduce(lambda a, b: a * b, levels))
        self.n_e = self.n_codes  # alias for compatibility with VQ code

        # Pre-compute mixed-radix basis for index encoding
        # basis[i] = prod(levels[i+1:])
        basis = torch.ones(self.dim, dtype=torch.int64)
        for i in range(self.dim - 2, -1, -1):
            basis[i] = basis[i + 1] * _levels[i + 1]
        self.register_buffer("_basis", basis)

        # Half-widths per dimension (for mapping to integer range)
        # Using [0, L-1] representation: simpler and avoids even/odd level issues
        self.register_buffer(
            "_level_max", (_levels.float() - 1.0).clamp(min=1.0)
        )

    def extra_repr(self):
        return (
            f"levels={self._levels.tolist()}, "
            f"n_codes={self.n_codes}, dim={self.dim}"
        )

    # ------------------------------------------------------------------
    # Core quantization
    # ------------------------------------------------------------------

    def _bound(self, z: torch.Tensor) -> torch.Tensor:
        """Bound continuous z to [0, L_i-1] range per dimension using tanh."""
        # tanh: R → (-1,1), shift+scale to (0, L_i-1)
        return (torch.tanh(z) + 1.0) / 2.0 * self._level_max

    def _quantize(self, z: torch.Tensor) -> torch.Tensor:
        """
        Quantize z ∈ R^d to integer codes in {0, 1, ..., L_i-1} per dim with STE.
        """
        z_bounded = self._bound(z)           # continuous in (0, L_i-1)
        z_hat = _round_ste(z_bounded)         # integer valued, gradients via STE
        z_hat = z_hat.clamp(min=0.0)          # safety clamp lower
        # Upper clamp per dimension
        z_hat = torch.min(z_hat, self._level_max)
        return z_hat

    def _normalize(self, z_hat: torch.Tensor) -> torch.Tensor:
        """Normalize integer codes [0, L_i-1] to [-1, +1] for decoder input."""
        return z_hat / self._level_max * 2.0 - 1.0

    # ------------------------------------------------------------------
    # Index conversion
    # ------------------------------------------------------------------

    def codes_to_indices(self, z_hat: torch.Tensor) -> torch.Tensor:
        """
        Convert quantized integer codes → flat index.

        z_hat: [..., dim] integer-valued tensor in [0, L_i-1] per dim
        Returns: [...] long tensor with values in [0, n_codes)
        """
        return (z_hat.long() * self._basis).sum(dim=-1)

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Convert flat indices → quantized integer codes in [0, L_i-1] per dim.

        indices: [...] long tensor
        Returns: [..., dim] float tensor
        """
        codes = torch.zeros(
            *indices.shape, self.dim,
            device=indices.device, dtype=torch.long,
        )
        remaining = indices.clone()
        for i in range(self.dim):
            codes[..., i] = remaining // self._basis[i]
            remaining = remaining % self._basis[i]
        return codes.float()

    # ------------------------------------------------------------------
    # Forward (VectorQuantizer-compatible interface)
    # ------------------------------------------------------------------

    def forward(self, z, valid_mask=None):
        """
        Quantize input z.

        Args:
            z: [B, L, dim] continuous encoder output
            valid_mask: [B, L] bool tensor (True = valid, False = padded)

        Returns: (z_q, loss, info)
            z_q: [B, dim, L] quantized codes (permuted for VQ compat)
            loss: scalar 0 (FSQ has no quantization loss)
            info: (perplexity, None, min_encoding_indices [N, 1])
        """
        assert z.shape[-1] == self.dim, (
            f"FSQ expects last dim = {self.dim}, got {z.shape[-1]}"
        )

        # Quantize
        z_hat = self._quantize(z)                        # [B, L, dim], int in [0, L_i-1]
        indices = self.codes_to_indices(z_hat)            # [B, L]

        # No quantization loss for FSQ
        loss = torch.zeros((), device=z.device, dtype=z.dtype)

        # Perplexity (code utilization metric)
        perplexity = self._compute_perplexity(indices, valid_mask)

        # Flatten indices to [N, 1] for VQ-compatible info tuple
        min_encoding_indices = indices.reshape(-1, 1)

        # Normalize z_hat to [-1, 1] for decoder input
        z_hat_norm = self._normalize(z_hat)  # [-1, 1]

        # Permute to [B, dim, L] for VQ compatibility
        z_q = z_hat_norm.permute(0, 2, 1).contiguous()

        return z_q, loss, (perplexity, None, min_encoding_indices)

    def _compute_perplexity(self, indices, valid_mask=None):
        """Compute perplexity from index distribution."""
        idx_flat = indices.reshape(-1)

        if valid_mask is not None:
            valid_flat = valid_mask.reshape(-1).bool()
            if valid_flat.numel() == idx_flat.numel() and valid_flat.any():
                idx_flat = idx_flat[valid_flat]

        if idx_flat.numel() == 0:
            return torch.zeros((), device=indices.device)

        # One-hot encoding for perplexity
        onehot = torch.zeros(idx_flat.shape[0], self.n_codes, device=indices.device)
        onehot.scatter_(1, idx_flat.unsqueeze(1).clamp(0, self.n_codes - 1), 1)
        avg_probs = onehot.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return perplexity

    # ------------------------------------------------------------------
    # Codebook entry retrieval (for compatibility)
    # ------------------------------------------------------------------

    def get_codebook_entry(self, indices, shape=None):
        """
        Look up codes from indices (VQ-compatible interface).

        Args:
            indices: [N] flat indices
            shape: ignored (for VQ compat)

        Returns:
            z_q: [N, dim] normalized codes in [-1, 1]
        """
        z_hat = self.indices_to_codes(indices)  # [N, dim], int in [0, L-1]
        z_q = self._normalize(z_hat)            # [-1, 1]
        return z_q
