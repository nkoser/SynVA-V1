"""
Flow Matching velocity predictor for tree-structured vessel geometry.

Architecture: Topology-conditioned Transformer operating on continuous [N, 39]
geometry features with full bidirectional attention at every denoising step.

The key advantage over MaskGIT / GPT-2:
    At EVERY step the model sees ALL nodes simultaneously — exactly like the
    FSQ encoder.  There is no tokenization bottleneck and no left-to-right
    generation handicap.

Convention (OT-CFM linear path):
    x_t = (1 - t) * noise  +  t * x_0     (t ∈ [0, 1])
    velocity  v = x_0 - noise              (constant along the straight path)
    t = 0 → pure noise,  t = 1 → clean data
"""

import math

import torch
import torch.nn as nn

from Stage2_MaskGIT.maskgit_model import PositionalEncoding, build_tree_attention_mask


# ──────────────────────────────────────────────────────────────────────────────
# Time embedding
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
        t = t.float().view(-1, 1)  # [B, 1]
        half = self.d_model // 2
        freq = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half - 1, 1))
        ).view(1, -1)
        args = t * freq
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.d_model:
            emb = nn.functional.pad(emb, (0, self.d_model - emb.shape[-1]))
        elif emb.shape[-1] > self.d_model:
            emb = emb[:, :self.d_model]
        return self.proj(emb)


# ──────────────────────────────────────────────────────────────────────────────
# Velocity predictor
# ──────────────────────────────────────────────────────────────────────────────

class FlowMatchingVelocityModel(nn.Module):
    """
    Predicts velocity v_θ(x_t, t | topology) for OT-CFM.

    Input per node: noisy geometry [39] + structural embeddings + time t.
    Output per node: predicted velocity [39].
    """

    def __init__(
        self,
        geom_dim: int = 39,
        k_classes: int = 3,
        max_depth: int = 128,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 1024,
        max_nodes: int = 256,
        dropout: float = 0.1,
        tree_attn_hops: int = 0,
        input_clamp_value: float | None = None,
        self_conditioning: bool = False,
        cfg_dropout: float = 0.0,
        depth_in_geometry: bool = False,
    ):
        super().__init__()
        self.geom_dim = int(geom_dim)
        self.d_model = int(d_model)
        self.max_nodes = int(max_nodes)
        self.tree_attn_hops = int(tree_attn_hops)
        self.self_conditioning = bool(self_conditioning)
        self.cfg_dropout = float(cfg_dropout)
        self.depth_in_geometry = bool(depth_in_geometry)
        self._max_depth_val = float(max_depth)
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
        # Projects the previous estimate x̂_1 into d_model, added to geom
        if self.self_conditioning:
            self.self_cond_proj = nn.Sequential(
                nn.Linear(self.geom_dim, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.d_model),
                nn.GELU(),
            )

        # ── Geometry projection ───────────────────────────────────────
        # +1 for node_mask indicator, +1 for continuous depth if enabled
        geom_input_dim = self.geom_dim + 1 + (1 if self.depth_in_geometry else 0)
        self.geom_proj = nn.Sequential(
            nn.Linear(geom_input_dim, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )

        # ── Time embedding ────────────────────────────────────────────
        self.time_embed = SinusoidalTimeEmbedding(self.d_model)

        # ── Combine struct + geom → d_model ──────────────────────────
        self.input_proj = nn.Linear(self.d_model * 2, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.input_drop = nn.Dropout(float(dropout))
        self.pos_enc = PositionalEncoding(
            self.d_model, max_len=self.max_nodes, dropout=float(dropout)
        )

        # ── Transformer (full bidirectional) ─────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(d_ff),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=int(n_layers)
        )

        # ── Output head → velocity [39] ──────────────────────────────
        self.output_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.geom_dim),
        )

    # ──────────────────────────────────────────────────────────────────
    def forward(
        self,
        k_counts: torch.Tensor,      # [B, N]
        depths: torch.Tensor,         # [B, N]
        child_slots: torch.Tensor,    # [B, N]
        x_t: torch.Tensor,            # [B, N, 39]  noisy geometry
        t: torch.Tensor,              # [B]          time ∈ [0, 1]
        node_mask: torch.Tensor | None = None,  # [B, N] bool
        parents: torch.Tensor | None = None,    # [B, N]
        x_self_cond: torch.Tensor | None = None,  # [B, N, 39] previous x̂_1 estimate
        force_uncond: bool = False,    # Force unconditional (zero topology) for CFG
    ) -> torch.Tensor:
        """Predict velocity v_θ(x_t, t | topo).  Returns [B, N, 39]."""
        B, N, _ = x_t.shape
        device = x_t.device

        if node_mask is None:
            node_mask = torch.ones(B, N, dtype=torch.bool, device=device)

        x = x_t.float()
        if self.input_clamp_value is not None:
            x = x.clamp(-self.input_clamp_value, self.input_clamp_value)

        # Structural features
        k_e = self.k_embed(k_counts.clamp(0, self.k_embed.num_embeddings - 1))
        d_e = self.depth_embed(depths.clamp(0, self.depth_embed.num_embeddings - 1))
        c_e = self.child_slot_embed(child_slots.clamp(0, 2))
        struct = self.struct_proj(torch.cat([k_e, d_e, c_e], dim=-1))  # [B, N, D]

        # Classifier-Free Guidance: randomly zero out topology during training
        if self.training and self.cfg_dropout > 0:
            drop = (torch.rand(B, device=device) < self.cfg_dropout).float()  # [B]
            struct = struct * (1.0 - drop.view(B, 1, 1))  # zero out entire sample
        # Force unconditional for CFG inference
        if force_uncond:
            struct = torch.zeros_like(struct)

        # Geometry features  (noisy geom + node_mask indicator [+ depth])
        geom_features = torch.cat(
            [x, node_mask.float().unsqueeze(-1)], dim=-1
        )  # [B, N, 40]
        if self.depth_in_geometry:
            # Continuous normalized depth ∈ [0, 1]
            depth_cont = (depths.float() / max(self._max_depth_val, 1.0)).unsqueeze(-1)
            geom_features = torch.cat([geom_features, depth_cont], dim=-1)  # [B, N, 41]
        geom_h = self.geom_proj(geom_features)  # [B, N, D]

        # Self-conditioning: add projected previous estimate
        if self.self_conditioning and x_self_cond is not None:
            sc_h = self.self_cond_proj(x_self_cond.float())  # [B, N, D]
            geom_h = geom_h + sc_h

        # Combine + time conditioning
        h = self.input_proj(torch.cat([struct, geom_h], dim=-1))  # [B, N, D]
        # Support both per-sample t [B] and per-node t [B, N] (Wavefront FM)
        if t.dim() == 1:
            h = h + self.time_embed(t).unsqueeze(1)  # broadcast [B, 1, D]
        else:
            # Per-node times: embed each node's time independently
            t_flat = t.reshape(-1)              # [B*N]
            t_emb = self.time_embed(t_flat)     # [B*N, D]
            h = h + t_emb.reshape(B, N, -1)    # [B, N, D]
        h = self.input_norm(h)
        h = self.input_drop(h)
        h = self.pos_enc(h)

        # Transformer with optional tree attention
        if self.tree_attn_hops > 0 and parents is not None:
            tree_mask = build_tree_attention_mask(
                parents, node_mask, n_hops=self.tree_attn_hops
            )
            n_heads = self.transformer.layers[0].self_attn.num_heads
            attn_mask = torch.zeros(B, N, N, device=device)
            attn_mask[~tree_mask] = float("-inf")
            attn_mask = (
                attn_mask.unsqueeze(1)
                .expand(-1, n_heads, -1, -1)
                .reshape(B * n_heads, N, N)
            )
            h = self.transformer(h, mask=attn_mask)
        else:
            h = self.transformer(h, src_key_padding_mask=~node_mask)

        v = self.output_head(h)  # [B, N, 39]
        # Zero out padded positions
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
                force_uncond=force_uncond,
            )
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
                force_uncond=force_uncond,
            )
            x_hat_new = None

        # Per-feature-group velocity scaling
        if velocity_scale_per_group is not None:
            G = self.geom_dim
            vs_tensor = torch.ones(G, device=x.device)
            for (a, b), s in velocity_scale_per_group.items():
                vs_tensor[a:b] = s
            v = v * vs_tensor.view(1, 1, G)

        return v, x_hat_new

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
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate geometry by integrating the learned velocity field
        from t=0 (noise) to t=1 (data).

        solver: 'euler' (1st order) or 'heun' (2nd order, 2× NFE).
        time_schedule: 'linear' (uniform dt) or 'quadratic' (more steps near t=1).
        guidance_scale: CFG weight (1.0 = no guidance, >1 = sharpen conditional).
            Requires model trained with cfg_dropout > 0.

        With self-conditioning: at each step, predict v once to get x̂_1,
        then predict v again with x̂_1 as self-conditioning input.

        Returns: [B, N, 39] predicted clean local geometry.
        """
        device = k_counts.device
        B, N = k_counts.shape

        if node_mask is None:
            node_mask = torch.ones(B, N, dtype=torch.bool, device=device)

        mask_f = node_mask.float().unsqueeze(-1)

        # Start from Gaussian noise
        x = torch.randn(B, N, self.geom_dim, device=device) * mask_f

        # Self-conditioning state: previous x̂_1 estimate
        x_hat = torch.zeros_like(x) if self.self_conditioning else None

        # Build time schedule
        if time_schedule == "quadratic":
            # More steps concentrated near t=1 (where details are decided)
            t_points = [(i / n_steps) ** 2 for i in range(n_steps + 1)]
        else:  # linear
            t_points = [i / n_steps for i in range(n_steps + 1)]

        for step in range(n_steps):
            t_val = t_points[step]
            t_next = t_points[step + 1]
            dt = t_next - t_val
            t_batch = torch.full((B,), t_val, device=device)

            # Predict v at current point
            v, x_hat_new = self._predict_velocity(
                k_counts, depths, child_slots, x, t_batch,
                node_mask, parents, x_hat, mask_f,
                clamp_value, velocity_scale_per_group,
            )
            if x_hat_new is not None:
                x_hat = x_hat_new

            # Classifier-Free Guidance: v_guided = v_uncond + w*(v_cond - v_uncond)
            if guidance_scale != 1.0 and self.cfg_dropout > 0:
                # Unconditional pass: same inputs but force_uncond=True
                v_uncond, _ = self._predict_velocity(
                    k_counts, depths, child_slots, x, t_batch,
                    node_mask, parents, x_hat, mask_f,
                    clamp_value, velocity_scale_per_group,
                    force_uncond=True,
                )
                v = v_uncond + guidance_scale * (v - v_uncond)

            if solver == "heun" and step < n_steps - 1:
                # Heun: x̃ = x + dt * v1, then v2 = f(x̃, t_next)
                x_tilde = (x + v * dt * velocity_scale) * mask_f
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

                # Heun update: average of v1 and v2
                x = x + 0.5 * (v + v2) * dt * velocity_scale
            else:
                # Euler update
                x = x + v * dt * velocity_scale

            limit = clamp_value if clamp_value is not None else self.input_clamp_value
            if limit is not None:
                x = x.clamp(-limit, limit)
            x = x * mask_f

        return x
    # ──────────────────────────────────────────────────────────────────
    def sample_guided(
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
        guidance_scale: float = 1.0,
        # ── Physio guidance params ──
        physio_fn=None,
        physio_kwargs: dict | None = None,
        guidance_strength: float = 0.1,
        guidance_t_min: float = 0.3,
        guidance_t_max: float = 0.95,
        guidance_schedule: str = "linear",
        guidance_grad_clip: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate geometry with Physio-Guided Sampling.

        Same ODE integration as sample(), but at each step the velocity is
        corrected by subtracting  λ(t) · ∇_{x_t} L_physio(x̂_1).

        This steers the trajectory toward physiologically plausible outputs
        WITHOUT retraining the model.

        Args:
            physio_fn: callable(x_hat, parents, k_counts, node_mask, depths, **kwargs)
                       → dict with 'physio_total' key (scalar loss).
            physio_kwargs: extra kwargs for physio_fn (weights etc.).
            guidance_strength: base λ for the gradient correction.
            guidance_t_min: start applying guidance at this t (0=noise).
            guidance_t_max: stop guidance at this t (1=clean).
            guidance_schedule: 'linear' | 'cosine' | 'constant'
                linear:  λ(t) ramps linearly within [t_min, t_max]
                cosine:  λ(t) follows cosine schedule (smooth ramp)
                constant: λ(t) = guidance_strength in active range
            guidance_grad_clip: max norm for the physio gradient (stability).

        Returns: [B, N, 39] predicted clean local geometry.
        """
        import math as _math

        device = k_counts.device
        B, N = k_counts.shape
        physio_kwargs = physio_kwargs or {}

        if node_mask is None:
            node_mask = torch.ones(B, N, dtype=torch.bool, device=device)
        mask_f = node_mask.float().unsqueeze(-1)

        # Start from Gaussian noise
        x = torch.randn(B, N, self.geom_dim, device=device) * mask_f
        x_hat = torch.zeros_like(x) if self.self_conditioning else None

        # Build time schedule
        if time_schedule == "quadratic":
            t_points = [(i / n_steps) ** 2 for i in range(n_steps + 1)]
        else:
            t_points = [i / n_steps for i in range(n_steps + 1)]

        def _guidance_lambda(t_val):
            """Compute time-dependent guidance weight."""
            if physio_fn is None:
                return 0.0
            if t_val < guidance_t_min or t_val > guidance_t_max:
                return 0.0
            if guidance_schedule == "constant":
                return guidance_strength
            frac = (t_val - guidance_t_min) / max(guidance_t_max - guidance_t_min, 1e-6)
            if guidance_schedule == "cosine":
                # Bell-shaped: peaks at midpoint
                return guidance_strength * _math.sin(frac * _math.pi)
            else:  # linear ramp-up
                return guidance_strength * frac

        for step in range(n_steps):
            t_val = t_points[step]
            t_next = t_points[step + 1]
            dt = t_next - t_val
            t_batch = torch.full((B,), t_val, device=device)
            lam = _guidance_lambda(t_val)

            # ── Standard velocity prediction (no grad) ──
            with torch.no_grad():
                v, x_hat_new = self._predict_velocity(
                    k_counts, depths, child_slots, x, t_batch,
                    node_mask, parents, x_hat, mask_f,
                    clamp_value, velocity_scale_per_group,
                )
                if x_hat_new is not None:
                    x_hat = x_hat_new

                # CFG
                if guidance_scale != 1.0 and self.cfg_dropout > 0:
                    v_uncond, _ = self._predict_velocity(
                        k_counts, depths, child_slots, x, t_batch,
                        node_mask, parents, x_hat, mask_f,
                        clamp_value, velocity_scale_per_group,
                        force_uncond=True,
                    )
                    v = v_uncond + guidance_scale * (v - v_uncond)

            # ── Physio gradient guidance ──
            if lam > 0 and physio_fn is not None:
                # Estimate x̂_1 from current x_t and predicted velocity
                remaining = max(1.0 - t_val, 1e-4)
                x_for_grad = x.detach().requires_grad_(True)
                x_hat_1 = (x_for_grad + remaining * v.detach()) * mask_f

                # Compute physio loss on estimated clean data
                p_loss = physio_fn(
                    x_hat_1, parents, k_counts, node_mask,
                    depths=depths, **physio_kwargs,
                )
                loss_val = p_loss["physio_total"]

                # Backprop to get gradient w.r.t. x_t
                grad = torch.autograd.grad(loss_val, x_for_grad)[0]  # [B, N, 39]

                # Clip gradient for stability
                if guidance_grad_clip > 0:
                    grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    grad = grad * (guidance_grad_clip / grad_norm).clamp(max=1.0)

                # Zero out padded
                grad = grad * mask_f

                # Correct velocity: steer away from high physio loss
                v = v - lam * grad

            # ── Euler step ──
            x = x + v * dt * velocity_scale

            limit = clamp_value if clamp_value is not None else self.input_clamp_value
            if limit is not None:
                x = x.clamp(-limit, limit)
            x = x * mask_f

        return x