"""
Tree-GNN Flow Matching velocity predictor for vessel tree geometry.

Key difference from Transformer-based model:
    Instead of flattening the tree to a sequence and using positional encoding +
    full self-attention, this model uses MESSAGE PASSING on the actual tree graph.

    Each message-passing layer computes directional messages:
        0 = Parent → Child  (top-down anatomical context)
        1 = Child → Parent  (bottom-up aggregation)
        2 = Sibling ↔ Sibling  (lateral symmetry, bifurcation angle)

    After K rounds of message passing, each node's hidden state incorporates
    K-hop tree context — the information flows along the tree edges, not through
    arbitrary attention over a flat sequence.

Convention (OT-CFM linear path):
    x_t = (1 - t) * noise  +  t * x_0     (t ∈ [0, 1])
    velocity  v = x_0 - noise              (constant along the straight path)
    t = 0 → pure noise,  t = 1 → clean data
"""

import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Time embedding (same as Transformer variant)
# ──────────────────────────────────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """Map scalar t ∈ [0,1] to a d_model-dimensional embedding."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] → [B, d_model]"""
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
            emb = emb[:, : self.d_model]
        return self.proj(emb)


# ──────────────────────────────────────────────────────────────────────────────
# Graph construction from tree topology
# ──────────────────────────────────────────────────────────────────────────────

def build_tree_edges(parents, k_counts, node_mask):
    """
    Convert parent indices to bidirectional edge lists with edge types.

    Args:
        parents:    [B, N] int64 — parent index per node (-1 for root)
        k_counts:   [B, N] int64 — child count (0/1/2)
        node_mask:  [B, N] bool  — valid node mask

    Returns:
        edge_index: [2, E] long  — source/destination indices (global: b*N + local)
        edge_type:  [E] long     — 0=parent→child, 1=child→parent, 2=sibling
        batch_ids:  [total_nodes] long — batch assignment per node (for scatter)
    """
    B, N = parents.shape
    device = parents.device

    src_list, dst_list, etype_list = [], [], []

    for b in range(B):
        mask = node_mask[b]
        n = mask.sum().item()
        offset = b * N
        par = parents[b].cpu().tolist()

        children_of = defaultdict(list)

        for i in range(1, n):
            p = par[i]
            if p < 0:
                continue
            # Parent → Child (direction 0)
            src_list.append(offset + p)
            dst_list.append(offset + i)
            etype_list.append(0)
            # Child → Parent (direction 1)
            src_list.append(offset + i)
            dst_list.append(offset + p)
            etype_list.append(1)
            children_of[p].append(i)

        # Sibling edges (at bifurcations)
        for p, kids in children_of.items():
            if len(kids) == 2:
                src_list.append(offset + kids[0])
                dst_list.append(offset + kids[1])
                etype_list.append(2)
                src_list.append(offset + kids[1])
                dst_list.append(offset + kids[0])
                etype_list.append(2)

    if len(src_list) == 0:
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
        edge_type = torch.zeros(0, dtype=torch.long, device=device)
    else:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=device)
        edge_type = torch.tensor(etype_list, dtype=torch.long, device=device)

    return edge_index, edge_type


# ──────────────────────────────────────────────────────────────────────────────
# Tree Message Passing Layer
# ──────────────────────────────────────────────────────────────────────────────

class TreeMessagePassingLayer(nn.Module):
    """
    One round of asymmetric tree message passing.

    Three edge types with SEPARATE MLPs:
        0 = Parent → Child  (top-down: propagate trunk geometry)
        1 = Child → Parent  (bottom-up: aggregate subtree info)
        2 = Sibling ↔ Sibling  (lateral: bifurcation symmetry)

    Update: h_new = LayerNorm(h + Update(h, AggMsg)) + FFN residual
    """

    def __init__(self, d_model: int, n_edge_types: int = 3, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_edge_types = n_edge_types

        # Separate message MLPs per edge type
        self.msg_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            for _ in range(n_edge_types)
        ])

        # Aggregated message → node update
        self.update_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # [self_feat || agg_msg]
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)

        # FFN (same as Transformer)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, h, edge_index, edge_type):
        """
        Args:
            h:          [total_nodes, d_model]
            edge_index: [2, E]
            edge_type:  [E]

        Returns:
            h:          [total_nodes, d_model]
        """
        src, dst = edge_index
        total_nodes = h.shape[0]

        # Compute messages per edge type and scatter-add to destinations
        agg_msg = torch.zeros_like(h)
        for etype in range(self.n_edge_types):
            mask = edge_type == etype
            if not mask.any():
                continue
            s_idx = src[mask]
            d_idx = dst[mask]
            pair = torch.cat([h[s_idx], h[d_idx]], dim=-1)  # [E_type, 2d]
            msg = self.msg_mlps[etype](pair)                # [E_type, d]
            agg_msg.scatter_add_(
                0,
                d_idx.unsqueeze(-1).expand_as(msg),
                msg,
            )

        # Update with residual
        combined = torch.cat([h, agg_msg], dim=-1)  # [N, 2d]
        h = self.norm1(h + self.drop1(self.update_mlp(combined)))

        # FFN with residual
        h = self.norm2(h + self.drop2(self.ffn(h)))

        return h


# ──────────────────────────────────────────────────────────────────────────────
# Optional: Global attention layer (for long-range symmetry)
# ──────────────────────────────────────────────────────────────────────────────

class GlobalAttentionLayer(nn.Module):
    """
    Standard multi-head self-attention over all nodes within each tree.
    Used sparingly (e.g. every 2-3 GNN layers) to capture long-range
    correlations that exceed the GNN receptive field.
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, h_batched, node_mask):
        """
        Args:
            h_batched:  [B, N, d_model]
            node_mask:  [B, N] bool

        Returns:
            h_batched:  [B, N, d_model]
        """
        attn_out, _ = self.attn(
            h_batched, h_batched, h_batched,
            key_padding_mask=~node_mask,
        )
        h_batched = self.norm(h_batched + self.drop(attn_out))
        return h_batched


# ──────────────────────────────────────────────────────────────────────────────
# Tree Flow Net: GNN-based velocity predictor
# ──────────────────────────────────────────────────────────────────────────────

class TreeFlowNet(nn.Module):
    """
    Graph Neural Network velocity predictor for OT-CFM on tree-structured data.

    Instead of a Transformer operating on a flat DFS sequence, this model:
    1. Embeds each node's noisy geometry + topology + time into d_model
    2. Runs K rounds of tree-aware message passing (parent→child, child→parent,
       sibling↔sibling edges, each with separate MLPs)
    3. Optionally interleaves sparse global attention layers for long-range
    4. Outputs per-node velocity prediction

    The inductive bias: information flows ALONG tree edges, not arbitrary attention.
    """

    def __init__(
        self,
        geom_dim: int = 39,
        k_classes: int = 3,
        max_depth: int = 128,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 1024,    # unused (GNN uses 4×d_model internally)
        max_nodes: int = 256,
        dropout: float = 0.1,
        self_conditioning: bool = False,
        global_attn_every: int = 0,
        depth_warp_alpha: float = 0.0,
        # Unused args for compatibility with Transformer config loading:
        tree_attn_hops: int = 0,
        input_clamp_value: float | None = None,
        cfg_dropout: float = 0.0,
    ):
        super().__init__()
        self.geom_dim = int(geom_dim)
        self.d_model = int(d_model)
        self.max_nodes = int(max_nodes)
        self.n_layers = int(n_layers)
        self.self_conditioning = bool(self_conditioning)
        self.global_attn_every = int(global_attn_every)
        self.depth_warp_alpha = float(depth_warp_alpha)
        self.input_clamp_value = (
            None if input_clamp_value is None else float(input_clamp_value)
        )

        # ── Structural embeddings (topology conditioning) ─────────────
        self.k_embed = nn.Embedding(int(k_classes), self.d_model)
        self.depth_embed = nn.Embedding(int(max_depth) + 1, self.d_model)
        self.child_slot_embed = nn.Embedding(3, self.d_model)
        self.struct_proj = nn.Sequential(
            nn.Linear(self.d_model * 3, self.d_model),
            nn.GELU(),
        )

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

        # ── Input projection (struct + geom → d_model) ───────────────
        self.input_proj = nn.Linear(self.d_model * 2, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.input_drop = nn.Dropout(dropout)

        # ── Tree Message Passing layers ──────────────────────────────
        self.gnn_layers = nn.ModuleList([
            TreeMessagePassingLayer(self.d_model, n_edge_types=3, dropout=dropout)
            for _ in range(self.n_layers)
        ])

        # ── Optional global attention layers ─────────────────────────
        self.global_attn_layers = nn.ModuleDict()
        if self.global_attn_every > 0:
            for i in range(self.n_layers):
                if (i + 1) % self.global_attn_every == 0:
                    self.global_attn_layers[str(i)] = GlobalAttentionLayer(
                        self.d_model, int(n_heads), dropout,
                    )

        # ── Output head → velocity [geom_dim] ────────────────────────
        self.output_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.geom_dim),
        )

    # ──────────────────────────────────────────────────────────────────
    def warp_time(self, t: torch.Tensor, depths: torch.Tensor,
                  node_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Depth-warped time:  t_eff[b, n] = t[b] ^ (1 + alpha * depth[b,n] / d_max)

        Shallow nodes (depth ≈ 0) get t_eff ≈ t  — denoised first (coarse).
        Deep nodes get t_eff < t — denoised later (fine details).

        Returns [B, N] per-node effective time.
        """
        if self.depth_warp_alpha == 0.0:
            return t.unsqueeze(1).expand(-1, depths.shape[1])  # [B, N]

        # Compute max depth per tree for proper normalization
        if node_mask is not None:
            # Set masked nodes to 0 for max computation
            masked_depths = depths.float() * node_mask.float()
            d_max = masked_depths.max(dim=1, keepdim=True).values.clamp(min=1.0)  # [B, 1]
        else:
            d_max = depths.float().max(dim=1, keepdim=True).values.clamp(min=1.0)

        # Exponent: 1 + alpha * d/d_max ∈ [1, 1+alpha]
        exponent = 1.0 + self.depth_warp_alpha * depths.float() / d_max  # [B, N]
        # t_eff = t^exponent; clamp t to avoid log(0)
        t_clamped = t.unsqueeze(1).clamp(min=1e-6)  # [B, 1]
        t_eff = t_clamped.pow(exponent)  # [B, N]
        return t_eff

    def forward(
        self,
        k_counts: torch.Tensor,       # [B, N]
        depths: torch.Tensor,          # [B, N]
        child_slots: torch.Tensor,     # [B, N]
        x_t: torch.Tensor,            # [B, N, geom_dim]  noisy geometry
        t: torch.Tensor,              # [B]               time ∈ [0, 1]
        node_mask: torch.Tensor | None = None,  # [B, N] bool
        parents: torch.Tensor | None = None,     # [B, N]  (REQUIRED!)
        x_self_cond: torch.Tensor | None = None,  # [B, N, geom_dim]
        force_uncond: bool = False,    # unused (for interface compat)
    ) -> torch.Tensor:
        """Predict velocity v_θ(x_t, t | tree).  Returns [B, N, geom_dim]."""
        B, N, G = x_t.shape
        device = x_t.device

        if node_mask is None:
            node_mask = torch.ones(B, N, dtype=torch.bool, device=device)
        if parents is None:
            raise ValueError("TreeFlowNet requires `parents` tensor!")

        x = x_t.float()
        if self.input_clamp_value is not None:
            x = x.clamp(-self.input_clamp_value, self.input_clamp_value)

        # ── Build tree graph ──────────────────────────────────────────
        edge_index, edge_type = build_tree_edges(parents, k_counts, node_mask)

        # ── Structural features ───────────────────────────────────────
        k_e = self.k_embed(k_counts.clamp(0, self.k_embed.num_embeddings - 1))
        d_e = self.depth_embed(depths.clamp(0, self.depth_embed.num_embeddings - 1))
        c_e = self.child_slot_embed(child_slots.clamp(0, 2))
        struct = self.struct_proj(torch.cat([k_e, d_e, c_e], dim=-1))  # [B,N,D]

        # ── Geometry features ─────────────────────────────────────────
        geom_features = torch.cat(
            [x, node_mask.float().unsqueeze(-1)], dim=-1,
        )  # [B, N, G+1]
        geom_h = self.geom_proj(geom_features)  # [B, N, D]

        # Self-conditioning
        if self.self_conditioning and x_self_cond is not None:
            sc_h = self.self_cond_proj(x_self_cond.float())
            geom_h = geom_h + sc_h

        # ── Combine + time (depth-warped if alpha > 0) ─────────────
        h = self.input_proj(torch.cat([struct, geom_h], dim=-1))  # [B,N,D]
        if self.depth_warp_alpha > 0.0:
            # Per-node time embedding: [B, N] → each node gets its own time
            t_eff = self.warp_time(t, depths, node_mask)  # [B, N]
            t_eff_flat = t_eff.reshape(B * N)  # [B*N]
            t_emb = self.time_embed(t_eff_flat)  # [B*N, D]
            h = h + t_emb.reshape(B, N, -1)
        else:
            h = h + self.time_embed(t).unsqueeze(1)
        h = self.input_norm(h)
        h = self.input_drop(h)

        # ── Flatten for GNN: [B, N, D] → [B*N, D] ───────────────────
        h_flat = h.reshape(B * N, -1)

        # ── Message passing layers ────────────────────────────────────
        for i, gnn_layer in enumerate(self.gnn_layers):
            h_flat = gnn_layer(h_flat, edge_index, edge_type)

            # Optional global attention (operates on batched [B, N, D])
            if str(i) in self.global_attn_layers:
                h_batched = h_flat.reshape(B, N, -1)
                h_batched = self.global_attn_layers[str(i)](h_batched, node_mask)
                h_flat = h_batched.reshape(B * N, -1)

        # ── Reshape + output ──────────────────────────────────────────
        h = h_flat.reshape(B, N, -1)
        v = self.output_head(h)  # [B, N, geom_dim]
        v = v * node_mask.float().unsqueeze(-1)
        return v

    # ──────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _predict_velocity(
        self,
        k_counts, depths, child_slots, x, t_batch,
        node_mask, parents, x_hat, mask_f,
        clamp_value, velocity_scale_per_group,
        force_uncond=False,
    ):
        """Helper: predict velocity (with optional self-conditioning update)."""
        if self.self_conditioning:
            v = self.forward(
                k_counts, depths, child_slots, x, t_batch,
                node_mask=node_mask, parents=parents,
                x_self_cond=x_hat,
            )
            # For self-cond x_hat estimate: use per-node warped remaining time
            if self.depth_warp_alpha > 0.0:
                t_eff = self.warp_time(t_batch, depths, node_mask)  # [B, N]
                remaining = (1.0 - t_eff).clamp(min=1e-4).unsqueeze(-1)  # [B, N, 1]
            else:
                t_val = t_batch[0].item()
                remaining = max(1.0 - t_val, 1e-4)
            x_hat_new = (x + remaining * v) * mask_f
            sc_limit = clamp_value if clamp_value is not None else self.input_clamp_value
            if sc_limit is not None:
                x_hat_new = x_hat_new.clamp(-sc_limit, sc_limit)
        else:
            v = self.forward(
                k_counts, depths, child_slots, x, t_batch,
                node_mask=node_mask, parents=parents,
            )
            x_hat_new = None

        # Per-feature-group velocity scaling
        if velocity_scale_per_group is not None:
            vs_tensor = torch.ones(self.geom_dim, device=x.device)
            for (a, b), s in velocity_scale_per_group.items():
                vs_tensor[a:b] = s
            v = v * vs_tensor.view(1, 1, self.geom_dim)

        return v, x_hat_new

    @torch.no_grad()
    def sample(
        self,
        k_counts: torch.Tensor,
        depths: torch.Tensor,
        child_slots: torch.Tensor,
        node_mask: torch.Tensor | None = None,
        parents: torch.Tensor | None = None,
        n_steps: int = 50,
        clamp_value: float | None = None,
        velocity_scale: float = 1.0,
        velocity_scale_per_group: dict | None = None,
        solver: str = "euler",
        time_schedule: str = "linear",
        guidance_scale: float = 1.0,  # unused, kept for interface compat
    ) -> torch.Tensor:
        """
        Generate geometry by integrating the learned velocity field
        from t=0 (noise) to t=1 (data).

        Returns: [B, N, geom_dim] predicted clean local geometry.
        """
        device = k_counts.device
        B, N = k_counts.shape

        if node_mask is None:
            node_mask = torch.ones(B, N, dtype=torch.bool, device=device)
        if parents is None:
            raise ValueError("TreeFlowNet.sample() requires `parents`!")

        mask_f = node_mask.float().unsqueeze(-1)

        # Start from Gaussian noise
        x = torch.randn(B, N, self.geom_dim, device=device) * mask_f

        # Self-conditioning state
        x_hat = torch.zeros_like(x) if self.self_conditioning else None

        # Build time schedule
        if time_schedule == "quadratic":
            t_points = [(i / n_steps) ** 2 for i in range(n_steps + 1)]
        else:
            t_points = [i / n_steps for i in range(n_steps + 1)]

        for step in range(n_steps):
            t_val = t_points[step]
            t_next = t_points[step + 1]
            t_batch = torch.full((B,), t_val, device=device)

            # Compute per-node dt for depth-warped time
            if self.depth_warp_alpha > 0.0:
                t_eff_now = self.warp_time(t_batch, depths, node_mask)   # [B, N]
                t_batch_next = torch.full((B,), t_next, device=device)
                t_eff_next = self.warp_time(t_batch_next, depths, node_mask)  # [B, N]
                dt_per_node = (t_eff_next - t_eff_now).unsqueeze(-1)    # [B, N, 1]
            else:
                dt_per_node = t_next - t_val  # scalar

            v, x_hat_new = self._predict_velocity(
                k_counts, depths, child_slots, x, t_batch,
                node_mask, parents, x_hat, mask_f,
                clamp_value, velocity_scale_per_group,
            )
            if x_hat_new is not None:
                x_hat = x_hat_new

            if solver == "heun" and step < n_steps - 1:
                x_tilde = (x + v * dt_per_node * velocity_scale) * mask_f
                limit = clamp_value if clamp_value is not None else self.input_clamp_value
                if limit is not None:
                    x_tilde = x_tilde.clamp(-limit, limit)

                t_batch_next = torch.full((B,), t_next, device=device)
                v2, x_hat_new2 = self._predict_velocity(
                    k_counts, depths, child_slots, x_tilde, t_batch_next,
                    node_mask, parents, x_hat, mask_f,
                    clamp_value, velocity_scale_per_group,
                )
                if x_hat_new2 is not None:
                    x_hat = x_hat_new2
                x = x + 0.5 * (v + v2) * dt_per_node * velocity_scale
            else:
                x = x + v * dt_per_node * velocity_scale

            limit = clamp_value if clamp_value is not None else self.input_clamp_value
            if limit is not None:
                x = x.clamp(-limit, limit)
            x = x * mask_f

        return x
