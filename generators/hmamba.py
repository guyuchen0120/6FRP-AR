"""
Hierarchical Mamba (HMamba) FRP Generator.

Design:
  - Outer Mamba: scans the full 128-bit sequence to capture global hierarchy
    (prefix -> subnet -> interface dependencies).
  - Inner Mamba: groups samples by AS within a batch and scans across samples
    of the same AS at each bit position. This captures local patterns shared
    by FRPs of the same AS (e.g., common nibble patterns, length distributions).
  - Cross-fusion: a learnable gate fuses outer (per-bit global) and inner
    (per-bit AS-local) features, modulated by the AS embedding.

Training: MLM (random layer masked, predict masked bits) — fully compatible
with the existing UnifiedFRPGenerator interface (compute_mlm_loss /
generate_staged), so train.py / infer.py do NOT need changes.

Pure-PyTorch selective scan (no CUDA extension required) — runs on CPU.
"""
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

MASK_ID = 2  # bit vocab: {0, 1, MASK=2}


# ============================================================================
# Pure-PyTorch selective scan (CPU-friendly; replace with mamba-ssm on GPU)
# ============================================================================

def selective_scan(u: torch.Tensor,
                   delta: torch.Tensor,
                   A: torch.Tensor,
                   B: torch.Tensor,
                   C: torch.Tensor) -> torch.Tensor:
    """
    Pure-PyTorch selective scan.

    Args:
        u:     (B, L, D)        input
        delta: (B, L, D)        step size (input-dependent)
        A:     (D, N)           state matrix (log-parameterized outside)
        B:     (B, L, N)        input projection (input-dependent)
        C:     (B, L, N)        output projection (input-dependent)

    Returns:
        y:     (B, L, D)
    """
    B_size, L, D = u.shape
    N = A.shape[-1]

    # Discretize: dA = exp(delta * A)   shape (B, L, D, N)
    dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
    # dB = delta * B   shape (B, L, D, N)
    dB = delta.unsqueeze(-1) * B.unsqueeze(2)

    h = torch.zeros(B_size, D, N, device=u.device, dtype=u.dtype)
    ys = []
    for t in range(L):
        # h: (B, D, N); dA[:, t]: (B, D, N); dB[:, t]: (B, D, N); u[:, t]: (B, D)
        h = dA[:, t] * h + dB[:, t] * u[:, t].unsqueeze(-1)
        # y: (B, D) = sum_N (h * C[:, t].unsqueeze(1))
        y = (h * C[:, t].unsqueeze(1)).sum(-1)
        ys.append(y)
    return torch.stack(ys, dim=1)


# ============================================================================
# Mamba block
# ============================================================================

class MambaBlock(nn.Module):
    """
    Single Mamba block: in_proj -> conv1d -> SSM -> gate -> out_proj.

    Args:
        d_model: model dimension
        d_state: SSM hidden state size N (default 16)
        d_conv:  local convolution kernel width (default 4)
        expand:  inner expansion factor (default 2)
    """
    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            padding=d_conv - 1, groups=self.d_inner)

        # x_proj produces (delta, B, C) per-step (input-dependent)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # A: (d_inner, d_state), parameterized as log(-A) for stability
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(self.d_inner, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))  # skip term

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) -> (B, L, d_model)"""
        residual = x
        x = self.norm(x)

        xz = self.in_proj(x)                       # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)              # each (B, L, d_inner)

        # Local depthwise conv (causal-padded then truncated)
        L = x_in.size(1)
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)                    # (B, L, d_inner)

        # Selective parameters
        x_dbl = self.x_proj(x_conv)                # (B, L, 2N+1)
        dt, B_ssm, C_ssm = x_dbl.split(
            [1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))          # (B, L, d_inner)

        A = -torch.exp(self.A_log)                 # (d_inner, d_state)

        # SSM
        y = selective_scan(x_conv, dt, A, B_ssm, C_ssm)  # (B, L, d_inner)
        y = y + x_conv * self.D                          # skip

        # Gate
        y = y * F.silu(z)
        return self.out_proj(y) + residual


# ============================================================================
# Hierarchical Mamba layer: outer (per-sample, length=128) + inner (per-AS-group)
# ============================================================================

class HierarchicalMambaLayer(nn.Module):
    """
    One HMamba layer = Outer Mamba (along bit axis)
                     + Inner Mamba (along same-AS samples axis)
                     + AS-conditioned gating fusion.

    Inner Mamba mechanism:
      - Group samples in batch by as_id.
      - For each (as_id group, bit_position) pair, build a sequence of length
        `group_size` and run a Mamba scan along the sample axis.
      - Equivalent to: at bit position p, the inner Mamba reads features from
        all FRPs of the same AS in current batch (acts like an AS-prototype).
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.outer = MambaBlock(d_model, d_state=d_state, d_conv=d_conv)
        self.inner = MambaBlock(d_model, d_state=d_state, d_conv=d_conv)
        # Gate combines outer / inner using AS embedding signal
        self.gate = nn.Linear(d_model, 2)
        self.fuse_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, as_ids: torch.Tensor,
                as_cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:       (B, L, d_model) per-sample bit features
            as_ids:  (B,) long tensor — AS id of each sample
            as_cond: (B, d_model)   — AS conditioning vector for gating

        Returns:
            (B, L, d_model)
        """
        # ---- Outer scan: along bit axis (length=128) ----
        out_outer = self.outer(x)                            # (B, L, D)

        # ---- Inner scan: along same-AS samples axis ----
        out_inner = self._scan_within_as(x, as_ids)          # (B, L, D)

        # ---- Gate fusion (per-sample, conditioned on AS) ----
        g = F.softmax(self.gate(as_cond), dim=-1)            # (B, 2)
        g_outer = g[:, 0:1].unsqueeze(-1)                    # (B, 1, 1)
        g_inner = g[:, 1:2].unsqueeze(-1)                    # (B, 1, 1)
        fused = g_outer * out_outer + g_inner * out_inner
        return self.fuse_norm(fused + x)

    def _scan_within_as(self, x: torch.Tensor,
                        as_ids: torch.Tensor) -> torch.Tensor:
        """
        Group samples by as_id; within each group, run Mamba scan along the
        sample axis at each bit position simultaneously.

        Implementation:
          For each group of size G:
            x_group shape: (G, L, D)
            -> reshape to (1, G, L*D) won't work; instead we treat L as "batch"
               and G as "sequence length" for the inner Mamba.
            -> permute: (L, G, D) and run Mamba on it (treat L as batch).
        """
        B, L, D = x.shape
        out = torch.zeros_like(x)
        unique_as = torch.unique(as_ids)
        for a in unique_as.tolist():
            idx = (as_ids == a).nonzero(as_tuple=True)[0]    # (G,)
            G = idx.size(0)
            if G == 1:
                # Single-sample group: inner == outer no-op (use x itself
                # passed through inner block once with length=1).
                # Treat as 1-step scan: just zero out and let outer handle it.
                out[idx] = x[idx]
                continue
            grp = x[idx]                                     # (G, L, D)
            # Treat L as batch, G as sequence length.
            grp_t = grp.transpose(0, 1).contiguous()         # (L, G, D)
            scanned = self.inner(grp_t)                      # (L, G, D)
            scanned = scanned.transpose(0, 1).contiguous()   # (G, L, D)
            out[idx] = scanned
        return out


# ============================================================================
# Condition encoder (same interface as UnifiedFRPGenerator's)
# ============================================================================

class ConditionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.as_embed = nn.Embedding(cfg.num_as, cfg.as_embed_dim)
        self.org_embed = nn.Embedding(cfg.num_org, cfg.org_embed_dim)
        self.cat1_embed = nn.Embedding(cfg.num_cat1, cfg.cat1_embed_dim)
        self.cat2_embed = nn.Embedding(cfg.num_cat2, cfg.cat2_embed_dim)
        self.proto_embed = nn.Embedding(cfg.num_proto, cfg.proto_embed_dim)
        self.geo_embed = nn.Embedding(cfg.num_geo, cfg.geo_embed_dim)

        total_meta = (cfg.as_embed_dim + cfg.org_embed_dim +
                      cfg.cat1_embed_dim + cfg.cat2_embed_dim +
                      cfg.proto_embed_dim + cfg.geo_embed_dim +
                      cfg.rtt_feat_dim)
        self.meta_proj = nn.Linear(total_meta, cfg.d_model)
        self.meta_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        h = torch.cat([
            self.as_embed(meta["as_id"]),
            self.org_embed(meta["org_id"]),
            self.cat1_embed(meta["cat1_id"]),
            self.cat2_embed(meta["cat2_id"]),
            self.proto_embed(meta["proto_id"]),
            self.geo_embed(meta["geo_id"]),
            meta["rtt_stats"],
        ], dim=-1)
        return self.meta_norm(self.meta_proj(h))


# ============================================================================
# Hierarchical Mamba FRP Generator
# ============================================================================

class HMambaFRPGenerator(nn.Module):
    """
    Drop-in replacement for UnifiedFRPGenerator with HMamba backbone.

    Same forward / compute_mlm_loss / generate_staged interface as the
    Transformer version, so train.py / infer.py work without any change.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.bit_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.layer_embed = nn.Embedding(3, cfg.d_model)

        self.condition_encoder = ConditionEncoder(cfg)

        # Hierarchical Mamba stack
        d_state = getattr(cfg, "ssm_d_state", 16)
        d_conv = getattr(cfg, "ssm_d_conv", 4)
        self.layers = nn.ModuleList([
            HierarchicalMambaLayer(cfg.d_model, d_state=d_state, d_conv=d_conv)
            for _ in range(cfg.num_layers)
        ])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 2)

        # Layer-of-bit map (coarse / medium / fine)
        layer_of_bit = torch.zeros(cfg.seq_len, dtype=torch.long)
        layer_of_bit[:cfg.coarse_end] = 0
        layer_of_bit[cfg.coarse_end:cfg.medium_end] = 1
        layer_of_bit[cfg.medium_end:] = 2
        self.register_buffer("layer_of_bit", layer_of_bit)

    # ---------------- forward ----------------

    def forward(self, bit_seq: torch.Tensor,
                meta: Dict[str, torch.Tensor],
                seed_bits: Optional[torch.Tensor] = None,
                seed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        bit_seq: (B, 128)
        meta:    dict of metadata tensors (must contain 'as_id')
        Returns: logits (B, 128, 2)
        """
        B = bit_seq.size(0)
        device = bit_seq.device

        pos = torch.arange(self.cfg.seq_len, device=device)
        x = (self.bit_embed(bit_seq)
             + self.pos_embed(pos).unsqueeze(0)
             + self.layer_embed(self.layer_of_bit).unsqueeze(0))

        cond = self.condition_encoder(meta)            # (B, D)
        x = x + cond.unsqueeze(1)                      # broadcast condition

        as_ids = meta["as_id"]
        for layer in self.layers:
            x = layer(x, as_ids, cond)

        x = self.final_norm(x)
        return self.head(x)

    # ---------------- MLM training ----------------

    def compute_mlm_loss(self, bit_target: torch.Tensor,
                         meta: Dict[str, torch.Tensor],
                         prefix_lens: torch.Tensor,
                         seed_bits: Optional[torch.Tensor] = None,
                         seed_mask: Optional[torch.Tensor] = None,
                         bit_weight: Optional[torch.Tensor] = None,
                         mask_stage: Optional[str] = None
                         ) -> Dict[str, torch.Tensor]:
        B, L = bit_target.shape
        device = bit_target.device

        if mask_stage is None:
            r = torch.rand(1).item()
            if r < 1 / 3:
                mask_stage = "coarse"
            elif r < 2 / 3:
                mask_stage = "medium"
            else:
                mask_stage = "fine"

        if mask_stage == "coarse":
            mask_start, mask_end = 0, self.cfg.coarse_end
        elif mask_stage == "medium":
            mask_start, mask_end = self.cfg.coarse_end, self.cfg.medium_end
        else:
            mask_start, mask_end = self.cfg.medium_end, self.cfg.seq_len

        bit_input = bit_target.clone()
        bit_input[:, mask_start:mask_end] = MASK_ID

        logits = self.forward(bit_input, meta, seed_bits, seed_mask)

        pos_range = torch.arange(L, device=device).unsqueeze(0)
        in_layer = (pos_range >= mask_start) & (pos_range < mask_end)
        in_prefix = pos_range < prefix_lens.unsqueeze(1)
        loss_mask = (in_layer & in_prefix).float()

        per_bit_loss = F.cross_entropy(
            logits.reshape(-1, 2), bit_target.long().reshape(-1),
            reduction="none").reshape(B, L)

        if bit_weight is not None:
            per_bit_loss = per_bit_loss * bit_weight.unsqueeze(0)

        denom = loss_mask.sum().clamp(min=1.0)
        loss = (per_bit_loss * loss_mask).sum() / denom

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct = ((pred == bit_target) & (loss_mask > 0)).float().sum()
            total = loss_mask.sum().clamp(min=1.0)
            bit_acc = (correct / total).item()

        return {
            "loss": loss,
            "bit_acc": bit_acc,
            "num_masked_bits": int(loss_mask.sum().item()),
            "mask_stage": mask_stage,
        }

    # ---------------- staged inference ----------------

    @torch.no_grad()
    def generate_staged(self, meta: Dict[str, torch.Tensor],
                        seed_bits: Optional[torch.Tensor] = None,
                        seed_mask: Optional[torch.Tensor] = None,
                        target_prefix_lens: Optional[torch.Tensor] = None,
                        bgp_constraint_bits: Optional[torch.Tensor] = None,
                        bgp_constraint_len: Optional[torch.Tensor] = None,
                        temperature: float = 1.0,
                        num_samples: int = 1,
                        per_bit_temperature: Optional[torch.Tensor] = None,
                        ) -> torch.Tensor:
        cfg = self.cfg
        device = meta["as_id"].device
        B = meta["as_id"].size(0)

        def expand(t):
            if t is None:
                return None
            return t.repeat_interleave(num_samples, dim=0)

        meta_b = {k: expand(v) for k, v in meta.items()}
        bgp_bits_b = expand(bgp_constraint_bits) if bgp_constraint_bits is not None else None
        bgp_len_b = expand(bgp_constraint_len) if bgp_constraint_len is not None else None
        plen_b = expand(target_prefix_lens) if target_prefix_lens is not None else None

        NB = B * num_samples
        bits = torch.full((NB, cfg.seq_len), MASK_ID, dtype=torch.long, device=device)

        if bgp_bits_b is not None and bgp_len_b is not None:
            for b in range(NB):
                Lc = int(bgp_len_b[b].item())
                if Lc > 0:
                    bits[b, :Lc] = bgp_bits_b[b, :Lc].long()

        # 3-stage filling
        bits = self._fill_layer(bits, meta_b, 0, cfg.coarse_end,
                                temperature, plen_b, bgp_len_b,
                                per_bit_temperature)
        bits = self._fill_layer(bits, meta_b, cfg.coarse_end, cfg.medium_end,
                                temperature, plen_b, bgp_len_b,
                                per_bit_temperature)
        bits = self._fill_layer(bits, meta_b, cfg.medium_end, cfg.seq_len,
                                temperature, plen_b, bgp_len_b,
                                per_bit_temperature)
        return bits

    def _fill_layer(self, bits, meta, start, end, temperature,
                    plen=None, bgp_len=None, per_bit_temperature=None):
        logits = self.forward(bits, meta)            # (B, L, 2)

        if per_bit_temperature is not None:
            pbt = per_bit_temperature.to(logits.device)
            t = pbt.unsqueeze(0).unsqueeze(-1).clamp(min=1e-6)
            probs = F.softmax(logits / t, dim=-1)
        else:
            probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)

        B = bits.size(0)
        for b in range(B):
            for pos in range(start, end):
                if bits[b, pos].item() != MASK_ID:
                    continue
                if plen is not None and pos >= int(plen[b].item()):
                    bits[b, pos] = 0
                    continue
                if bgp_len is not None and pos < int(bgp_len[b].item()):
                    continue
                p = probs[b, pos]
                bits[b, pos] = torch.multinomial(p, 1).item()
        return bits


# ============================================================================
# Entropy weight loader (compatible with bit_entropy_weights.json)
# ============================================================================

def load_entropy_weights(path: str, seq_len: int = 128) -> Optional[torch.Tensor]:
    """
    Load bit-entropy weights from prepared_new/bit_entropy_weights.json.
    Returns a tensor of shape (seq_len,) with mean-1 normalization.

    The file may have several formats — we try a few keys.
    """
    import json
    try:
        with open(path) as f:
            data = json.load(f)
        ent = None
        if isinstance(data, dict):
            if "global" in data and isinstance(data["global"], dict):
                ent = data["global"].get("entropy")
            if ent is None:
                ent = data.get("bit_entropy_per_position")
            if ent is None:
                ent = data.get("entropy")
        if ent is None or len(ent) < seq_len:
            return None
        t = torch.tensor([max(0.0, float(e)) for e in ent[:seq_len]],
                         dtype=torch.float32)
        mean = t.mean().clamp(min=1e-6)
        return t / mean
    except Exception:
        return None
