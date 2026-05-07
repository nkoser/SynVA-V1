"""
Autoregressive Tree Flow Matching — Level-by-Level Generation.

Key Idea:
    Instead of denoising ALL nodes simultaneously, we grow the tree
    level-by-level in BFS order:

    1. Level 0 (root): Generate root node geometry via ODE integration
    2. Level 1: Generate all depth-1 nodes, conditioned on clean root
    3. Level 2: Generate all depth-2 nodes, conditioned on clean L0+L1
    ...
    D. Level D: Generate deepest nodes, conditioned on all ancestors

    Within each level, nodes are denoised IN PARALLEL via standard
    Flow Matching (OT-CFM).  But across levels, generation is SEQUENTIAL.

    This creates a natural coarse-to-fine hierarchy:
    - Parent structure is fully resolved before children are generated
    - The model can use clean parent features as strong conditioning
    - Siblings at the same level attend to each other for coordination

Architecture:
    Standard Transformer (same as v8), but with a LEVEL-CAUSAL attention mask:
    - Nodes at depth < frontier_level: fully visible (clean conditioning)
    - Nodes at depth == frontier_level: visible to each other (being generated)
    - Nodes at depth > frontier_level: invisible (not yet generated)

Training:
    For each sample in a batch:
    1. Sample a random frontier_level L ~ Uniform(0, max_depth)
    2. Nodes depth < L: features = clean x_0 (teacher forcing)
    3. Nodes depth == L: features = noisy x_t at sampled time t
    4. Nodes depth > L: masked out entirely
    5. Loss = velocity MSE only on nodes at depth == L

    This trains the model to predict velocity for a specific level,
    given clean ancestors, which is exactly the inference setting.

Convention (OT-CFM):
    x_t = (1-t)*noise + t*x_0,  velocity target = x_0 - noise
    t=0 → noise, t=1 → clean data
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Time embedding (same as v8)
# ──────────────────────────────────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float().view(-1, 1)
        half = self.d_model // 2
        freq = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half - 1, 1))
        ).view(1, -1)
        args = t * freq
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.d_model:
            emb = F.pad(emb, (0, self.d_model - emb.shape[-1]))
        elif emb.shape[-1] > self.d_model:
            emb = emb[:, :self.d_model]
        return self.proj(emb)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for BFS node order."""
    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2 + d_model % 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


# ──────────────────────────────────────────────────────────────────────────────
# Level-causal attention mask
# ──────────────────────────────────────────────────────────────────────────────

def build_level_causal_mask(
    depths: torch.Tensor,      # [B, N]
    frontier_level: torch.Tensor,  # [B] — which level is being generated
    node_mask: torch.Tensor,   # [B, N] bool
) -> torch.Tensor:
    """
    Build attention mask for level-causal autoregressive generation.

    For each sample b:
        - Nodes with depth <= frontier_level[b]: CAN be attended to
        - Nodes with depth > frontier_level[b]: CANNOT be attended to (future)
        - Padded nodes: CANNOT be attended to

    Returns: [B, N, N] bool mask — True = allowed, False = blocked
    """
    B, N = depths.shape
    device = depths.device

    fl = frontier_level.view(B, 1)  # [B, 1]

    # Which nodes are "visible" (ancestors + current level)
    visible = (depths <= fl) & node_mask  # [B, N]

    # Attention mask: query at position i can attend to key at position j
    # if j is visible
    # [B, N, N]: for each query position, which keys are visible
    mask = visible.unsqueeze(1).expand(B, N, N)  # broadcast query dim

    return mask


# ──────────────────────────────────────────────────────────────────────────────
# Level indicator embedding
# ──────────────────────────────────────────────────────────────────────────────

class LevelStateEmbedding(nn.Module):
    """
    Embedding that tells the model the state of each node:
        0 = ancestor (clean, already generated)
        1 = frontier (being generated, noisy)
        2 = future (not yet generated, masked)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(3, d_model)

    def forward(self, node_state: torch.Tensor) -> torch.Tensor:
        """node_state: [B, N] with values in {0, 1, 2}"""
        return self.embed(node_state.clamp(0, 2))


# ──────────────────────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────────────────────

class AutoregressiveTreeFlowNet(nn.Module):
    """
    Transformer-based velocity predictor with level-causal attention.

    Same capacity as v8 Transformer, but trained and evaluated with
    level-by-level autoregressive conditioning.
    """

    def __init__(
        self,
        geom_dim: int = 39,
        k_classes: int = 3,
        max_depth: int = 128,
        d_model: int = 384,
        n_heads: int = 8,
        n_layers: int = 10,
        d_ff: int = 1536,
        max_nodes: int = 256,
        dropout: float = 0.1,
        self_conditioning: bool = False,
    ):
        super().__init__()
        self.geom_dim = int(geom_dim)
        self.d_model = int(d_model)
        self.max_nodes = int(max_nodes)
        self.n_layers = int(n_layers)
        self.self_conditioning = bool(self_conditioning)

        # ── Structural embeddings ─────────────────────────────────────
        self.k_embed = nn.Embedding(int(k_classes), self.d_model)
        self.depth_embed = nn.Embedding(int(max_depth) + 1, self.d_model)
        self.child_slot_embed = nn.Embedding(3, self.d_model)
        self.struct_proj = nn.Sequential(
            nn.Linear(self.d_model * 3, self.d_model),
            nn.GELU(),
        )

        # ── Node state embedding (ancestor/frontier/future) ──────────
        self.state_embed = LevelStateEmbedding(self.d_model)

        # ── Self-conditioning projection ──────────────────────────────
        if self.self_conditioning:
            self.self_cond_proj = nn.Sequential(
                nn.Linear(self.geom_dim, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.d_model),
                nn.GELU(),
            )

        # ── Geometry projection ───────────────────────────────────────
        self.geom_proj = nn.Sequential(
            nn.Linear(self.geom_dim + 1, self.d_model),  # +1 for node_mask
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )

        # ── Time embedding ────────────────────────────────────────────
        self.time_embed = SinusoidalTimeEmbedding(self.d_model)

        # ── Input combination ─────────────────────────────────────────
        # struct + geom + state → d_model
        self.input_proj = nn.Linear(self.d_model * 3, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.input_drop = nn.Dropout(dropout)
        self.pos_enc = PositionalEncoding(self.d_model, max_len=max_nodes, dropout=dropout)

        # ── Transformer encoder ───────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(d_ff),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.n_layers,
        )

        # ── Output head ───────────────────────────────────────────────
        self.output_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.geom_dim),
        )

    def forward(
        self,
        k_counts: torch.Tensor,       # [B, N]
        depths: torch.Tensor,          # [B, N]
        child_slots: torch.Tensor,     # [B, N]
        x_geom: torch.Tensor,         # [B, N, geom_dim] — mixed clean/noisy
        t: torch.Tensor,              # [B] time for frontier nodes
        node_mask: torch.Tensor,       # [B, N] bool
        node_state: torch.Tensor,      # [B, N] — 0=ancestor, 1=frontier, 2=future
        frontier_level: torch.Tensor,  # [B] — depth level being generated
        x_self_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict velocity for FRONTIER nodes.

        x_geom contains:
            - Clean features for ancestor nodes (depth < frontier_level)
            - Noisy features at time t for frontier nodes (depth == frontier_level)
            - Zeros for future nodes (depth > frontier_level)

        Returns [B, N, geom_dim] velocity (only meaningful at frontier positions)
        """
        B, N, G = x_geom.shape
        device = x_geom.device

        # ── Structural features ───────────────────────────────────────
        k_e = self.k_embed(k_counts.clamp(0, self.k_embed.num_embeddings - 1))
        d_e = self.depth_embed(depths.clamp(0, self.depth_embed.num_embeddings - 1))
        c_e = self.child_slot_embed(child_slots.clamp(0, 2))
        struct = self.struct_proj(torch.cat([k_e, d_e, c_e], dim=-1))

        # ── Geometry features ─────────────────────────────────────────
        geom_features = torch.cat(
            [x_geom, node_mask.float().unsqueeze(-1)], dim=-1
        )
        geom_h = self.geom_proj(geom_features)

        # Self-conditioning
        if self.self_conditioning and x_self_cond is not None:
            sc_h = self.self_cond_proj(x_self_cond.float())
            geom_h = geom_h + sc_h

        # ── Node state embedding ──────────────────────────────────────
        state_h = self.state_embed(node_state)

        # ── Combine ───────────────────────────────────────────────────
        h = self.input_proj(torch.cat([struct, geom_h, state_h], dim=-1))
        h = h + self.time_embed(t).unsqueeze(1)
        h = self.input_norm(h)
        h = self.input_drop(h)
        h = self.pos_enc(h)

        # ── Level-causal attention mask ───────────────────────────────
        level_mask = build_level_causal_mask(depths, frontier_level, node_mask)
        # Convert to float mask for Transformer: True→0, False→-inf
        n_heads = self.transformer.layers[0].self_attn.num_heads
        attn_mask = torch.zeros(B, N, N, device=device)
        attn_mask[~level_mask] = float("-inf")
        attn_mask = (
            attn_mask.unsqueeze(1)
            .expand(-1, n_heads, -1, -1)
            .reshape(B * n_heads, N, N)
        )

        h = self.transformer(h, mask=attn_mask)

        # ── Output velocity ───────────────────────────────────────────
        v = self.output_head(h)

        # Only meaningful at frontier positions, zero elsewhere
        frontier_mask = (node_state == 1).float().unsqueeze(-1)
        v = v * frontier_mask

        return v

    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def sample(
        self,
        k_counts: torch.Tensor,       # [B, N]
        depths: torch.Tensor,          # [B, N]
        child_slots: torch.Tensor,     # [B, N]
        node_mask: torch.Tensor,       # [B, N]
        n_steps: int = 50,
        clamp_value: float | None = None,
        velocity_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate tree geometry level-by-level.

        For each depth level L (0, 1, 2, ...):
            1. Set ancestor nodes (depth < L) to their clean generated features
            2. Initialize frontier nodes (depth == L) with noise
            3. Run n_steps ODE integration to denoise frontier
            4. Store generated features
            5. Move to next level

        Returns [B, N, geom_dim] generated geometry.
        """
        device = k_counts.device
        B, N = k_counts.shape

        # Output buffer — stores generated clean geometry
        x_clean = torch.zeros(B, N, self.geom_dim, device=device)

        # Find all unique depth levels across batch
        max_d = depths[node_mask].max().item() if node_mask.any() else 0

        for level in range(int(max_d) + 1):
            # Which nodes are at this level?
            is_frontier = (depths == level) & node_mask  # [B, N]

            # Skip if no nodes at this level in any batch
            if not is_frontier.any():
                continue

            # Build node_state: 0=ancestor, 1=frontier, 2=future
            node_state = torch.full((B, N), 2, dtype=torch.long, device=device)
            node_state[depths < level] = 0
            node_state[is_frontier] = 1
            node_state[~node_mask] = 2

            frontier_level = torch.full((B,), level, dtype=torch.long, device=device)
            frontier_mask = is_frontier.float().unsqueeze(-1)  # [B, N, 1]

            # Initialize frontier nodes with noise
            noise = torch.randn(B, N, self.geom_dim, device=device) * frontier_mask

            # Assemble x_geom: clean ancestors + noisy frontier + zeros future
            x_geom = x_clean.clone()  # ancestors are already clean
            x_geom = x_geom * (node_state == 0).float().unsqueeze(-1)  # keep ancestors
            x_geom = x_geom + noise  # add noise at frontier

            # Self-conditioning state
            x_hat = torch.zeros(B, N, self.geom_dim, device=device) if self.self_conditioning else None

            # ODE integration for frontier level
            t_points = [i / n_steps for i in range(n_steps + 1)]

            for step in range(n_steps):
                t_val = t_points[step]
                t_next = t_points[step + 1]
                dt = t_next - t_val
                t_batch = torch.full((B,), t_val, device=device)

                # Self-conditioning: first pass
                if self.self_conditioning:
                    v_sc = self.forward(
                        k_counts, depths, child_slots, x_geom, t_batch,
                        node_mask, node_state, frontier_level, x_self_cond=x_hat,
                    )
                    remaining = max(1.0 - t_val, 1e-4)
                    x_hat = (x_geom + remaining * v_sc) * frontier_mask
                    if clamp_value:
                        x_hat = x_hat.clamp(-clamp_value, clamp_value)

                v = self.forward(
                    k_counts, depths, child_slots, x_geom, t_batch,
                    node_mask, node_state, frontier_level, x_self_cond=x_hat,
                )

                # Euler step — only update frontier nodes
                x_frontier = x_geom * frontier_mask + (1 - frontier_mask) * x_geom
                x_frontier = x_frontier + v * dt * velocity_scale
                if clamp_value:
                    x_frontier = x_frontier.clamp(-clamp_value, clamp_value)

                # Reassemble: keep ancestors clean, update frontier
                x_geom = x_clean * (node_state == 0).float().unsqueeze(-1) + \
                          x_frontier * frontier_mask

            # Store generated features for this level
            x_clean = x_clean + x_geom * frontier_mask

        return x_clean
