"""
Unified Transformer FRP Generator.

Design:
  - Single Transformer encoder (Pre-LN, standard residual) processes full 128-bit
    IPv6 address with MASK tokens for unknown positions.
  - Layer-aware position embedding: pos_embed(i) + layer_embed(layer_of(i)).
  - Metadata + few-shot seed FRPs are fused into a condition prefix.
  - Training: MLM-style. During each step we mask one of the three layer segments
    (coarse / medium / fine) and predict the masked bits.
  - Inference: staged generation across three layers, all sharing the same model.
    Optional BGP-prefix constraint fixes bits[0:L] during generation.

Motivated by experimental findings:
  - Exp1: Three natural layers (bits 0-48, 48-64, 64-128) confirmed by entropy.
  - Exp2: Per-layer specialized architectures provide no measurable benefit;
          a unified model is simpler and equally effective.
  - Exp3: Standard residual is as good as attention residual and more stable.
"""
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


MASK_ID = 2   # bit vocab: {0, 1, MASK=2}


# ============================================================================
# Condition encoder
# ============================================================================

class ConditionEncoder(nn.Module):
    """
    Encodes metadata (AS id, org, category, protocol, geo, RTT stats) into a
    fixed-size vector. Optionally fuses few-shot seed FRPs.
    """
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
                      cfg.proto_embed_dim + cfg.geo_embed_dim + cfg.rtt_feat_dim)
        self.meta_proj = nn.Linear(total_meta, cfg.d_model)
        self.meta_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, meta: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            meta: dict with tensors
              as_id (B,), org_id (B,), cat1_id (B,), cat2_id (B,),
              proto_id (B,), geo_id (B,), rtt_stats (B, 3)

        Returns:
            condition: (B, d_model)
        """
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
# Main unified generator
# ============================================================================

class UnifiedFRPGenerator(nn.Module):
    """
    Masked-LM Transformer for FRP bit generation.

    Input:
      bit_seq: (B, 128) long tensor, values in {0, 1, MASK_ID=2}
      condition: (B, d_model) metadata vector
      optional seed_bits: (B, K, 128) few-shot seed FRPs (0/1 only)

    Output:
      logits: (B, 128, 2) — per-position 0/1 logits (ignores MASK class)
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Bit embedding (0, 1, MASK)
        self.bit_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # Position embedding
        self.pos_embed = nn.Embedding(cfg.seq_len, cfg.d_model)
        # Layer embedding (3 layers: coarse, medium, fine)
        self.layer_embed = nn.Embedding(3, cfg.d_model)

        # Condition encoder
        self.condition_encoder = ConditionEncoder(cfg)

        # Few-shot seed encoder (small Transformer, shared weights)
        self.seed_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                cfg.d_model, cfg.num_heads, cfg.ff_dim,
                dropout=cfg.dropout, batch_first=True, activation="gelu",
                norm_first=True),
            num_layers=cfg.seed_encoder_layers,
        )
        self.seed_pool_proj = nn.Linear(cfg.d_model, cfg.d_model)

        # Main Transformer (Pre-LN, standard residual — Exp3 confirmed sufficient)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.final_norm = nn.LayerNorm(cfg.d_model)

        # Output head: per-position logits over {0, 1}
        self.head = nn.Linear(cfg.d_model, 2)

        # Precompute layer-of-bit mapping
        layer_of_bit = torch.zeros(cfg.seq_len, dtype=torch.long)
        layer_of_bit[:cfg.coarse_end] = 0
        layer_of_bit[cfg.coarse_end:cfg.medium_end] = 1
        layer_of_bit[cfg.medium_end:] = 2
        self.register_buffer("layer_of_bit", layer_of_bit)

    # ---------- forward ----------
    def forward(self, bit_seq: torch.Tensor,
                meta: Dict[str, torch.Tensor],
                seed_bits: Optional[torch.Tensor] = None,
                seed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            bit_seq: (B, 128) with values in {0, 1, MASK_ID}
            meta: dict of metadata tensors
            seed_bits: (B, K, 128) optional few-shot seeds (0/1)
            seed_mask: (B, K) 1 for real seed, 0 for padding

        Returns:
            logits: (B, 128, 2)
        """
        B = bit_seq.size(0)
        device = bit_seq.device

        # Bit + position + layer embeddings
        pos = torch.arange(self.cfg.seq_len, device=device)
        x = (self.bit_embed(bit_seq)
             + self.pos_embed(pos).unsqueeze(0)
             + self.layer_embed(self.layer_of_bit).unsqueeze(0))   # (B, 128, D)

        # Condition vector (metadata + optional seeds)
        cond = self.condition_encoder(meta)                          # (B, D)
        if seed_bits is not None:
            cond = cond + self._encode_seeds(seed_bits, seed_mask)

        # Inject condition by adding to every position (shift)
        x = x + cond.unsqueeze(1)

        # Main encoder
        h = self.encoder(x)
        h = self.final_norm(h)

        # Per-position bit logits
        return self.head(h)

    def _encode_seeds(self, seed_bits: torch.Tensor,
                      seed_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Encode K seed FRPs into a single condition vector.

        Args:
            seed_bits: (B, K, 128) long
            seed_mask: (B, K) float (1 = real seed)
        Returns:
            (B, d_model)
        """
        B, K, L = seed_bits.shape
        flat = seed_bits.reshape(B * K, L)
        pos = torch.arange(L, device=seed_bits.device)
        emb = (self.bit_embed(flat)
               + self.pos_embed(pos).unsqueeze(0)
               + self.layer_embed(self.layer_of_bit).unsqueeze(0))
        seed_enc = self.seed_encoder(emb)               # (B*K, L, D)
        # Pool each seed to a vector
        seed_vec = seed_enc.mean(dim=1).view(B, K, -1)  # (B, K, D)
        if seed_mask is not None:
            mask = seed_mask.unsqueeze(-1).float()
            weighted = (seed_vec * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-6)
            pooled = weighted / denom                   # (B, D)
        else:
            pooled = seed_vec.mean(dim=1)
        return self.seed_pool_proj(pooled)

    # ---------- training helpers ----------
    def compute_mlm_loss(self, bit_target: torch.Tensor,
                         meta: Dict[str, torch.Tensor],
                         prefix_lens: torch.Tensor,
                         seed_bits: Optional[torch.Tensor] = None,
                         seed_mask: Optional[torch.Tensor] = None,
                         bit_weight: Optional[torch.Tensor] = None,
                         mask_stage: Optional[str] = None
                         ) -> Dict[str, torch.Tensor]:
        """
        Compute masked-LM loss. Randomly masks one layer segment per sample.

        Args:
            bit_target: (B, 128) 0/1 real FRP bits (padded with 0 past prefix_len)
            meta: metadata dict
            prefix_lens: (B,) actual prefix length
            seed_bits, seed_mask: optional few-shot
            bit_weight: (128,) entropy-based per-position weight (or None)
            mask_stage: "coarse" | "medium" | "fine" | None (random)

        Returns:
            dict with 'loss', 'bit_acc', 'num_masked_bits'
        """
        B, L = bit_target.shape
        device = bit_target.device

        # Decide which layer to mask
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

        # Build masked input
        bit_input = bit_target.clone()
        bit_input[:, mask_start:mask_end] = MASK_ID

        # Forward
        logits = self.forward(bit_input, meta, seed_bits, seed_mask)
        # Only compute loss on masked region and within prefix_len
        # Build per-position mask
        pos_range = torch.arange(L, device=device).unsqueeze(0)    # (1, L)
        in_layer = (pos_range >= mask_start) & (pos_range < mask_end)
        in_prefix = pos_range < prefix_lens.unsqueeze(1)            # (B, L)
        loss_mask = (in_layer & in_prefix).float()                  # (B, L)

        per_bit_loss = F.cross_entropy(
            logits.reshape(-1, 2), bit_target.long().reshape(-1), reduction="none"
        ).reshape(B, L)

        if bit_weight is not None:
            per_bit_loss = per_bit_loss * bit_weight.unsqueeze(0)

        denom = loss_mask.sum().clamp(min=1.0)
        loss = (per_bit_loss * loss_mask).sum() / denom

        # Bit accuracy on masked positions
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

    # ---------- staged inference ----------
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
        """
        Three-stage generation:
          Stage 1: mask all → predict coarse (bits [0:coarse_end))
          Stage 2: fix coarse, mask medium+fine → predict medium
          Stage 3: fix coarse+medium, mask fine → predict fine

        If bgp_constraint_bits / bgp_constraint_len are provided, those bits are
        forced to the constraint values (used by Mode B: Target discovery).

        Args:
            meta: (B, ...) metadata
            seed_bits: (B, K, 128) optional
            target_prefix_lens: (B,) desired output prefix length (default 128)
            bgp_constraint_bits: (B, 128) 0/1 tensor (only first L bits used)
            bgp_constraint_len: (B,) constraint length
            temperature: sampling temperature (scalar, used as base)
            num_samples: repeat generation per input
            per_bit_temperature: (128,) optional per-position temperature derived
                from entropy analysis. If provided, each bit position uses its own
                temperature = per_bit_temperature[pos] * temperature_scale_factor.
                Low-entropy bits get low temperature (conservative/locked),
                high-entropy bits get higher temperature (more variation).

        Returns:
            bits: (B * num_samples, 128) tensor of 0/1 generated bits
        """
        cfg = self.cfg
        device = meta["as_id"].device
        B = meta["as_id"].size(0)

        # Expand batch for num_samples
        def expand(t):
            if t is None:
                return None
            return t.repeat_interleave(num_samples, dim=0)

        meta_b = {k: expand(v) for k, v in meta.items()}
        seed_bits_b = expand(seed_bits) if seed_bits is not None else None
        seed_mask_b = expand(seed_mask) if seed_mask is not None else None
        bgp_bits_b = expand(bgp_constraint_bits) if bgp_constraint_bits is not None else None
        bgp_len_b = expand(bgp_constraint_len) if bgp_constraint_len is not None else None
        plen_b = expand(target_prefix_lens) if target_prefix_lens is not None else None

        NB = B * num_samples
        bits = torch.full((NB, cfg.seq_len), MASK_ID, dtype=torch.long, device=device)

        # If BGP constraint provided, fix those bits
        if bgp_bits_b is not None and bgp_len_b is not None:
            for b in range(NB):
                L = int(bgp_len_b[b].item())
                if L > 0:
                    bits[b, :L] = bgp_bits_b[b, :L].long()

        # Stage 1: coarse
        bits = self._fill_layer(bits, meta_b, seed_bits_b, seed_mask_b,
                                 0, cfg.coarse_end, temperature,
                                 plen_b, bgp_len_b, per_bit_temperature)
        # Stage 2: medium
        bits = self._fill_layer(bits, meta_b, seed_bits_b, seed_mask_b,
                                 cfg.coarse_end, cfg.medium_end, temperature,
                                 plen_b, bgp_len_b, per_bit_temperature)
        # Stage 3: fine
        bits = self._fill_layer(bits, meta_b, seed_bits_b, seed_mask_b,
                                 cfg.medium_end, cfg.seq_len, temperature,
                                 plen_b, bgp_len_b, per_bit_temperature)

        return bits

    def _fill_layer(self, bits, meta, seed_bits, seed_mask,
                    start, end, temperature, plen=None, bgp_len=None,
                    per_bit_temperature=None):
        """
        Ensure positions [start, end) are set. MASK positions are sampled
        from the model; already-set positions (including BGP constraints) stay.

        If per_bit_temperature is provided (shape (128,)), each position uses
        its own temperature for sampling. This enables entropy-guided generation:
        low-entropy bits (e.g., high-order network prefix bits 0-11) get very low
        temperature (nearly deterministic), while high-entropy bits get normal or
        elevated temperature.
        """
        logits = self.forward(bits, meta, seed_bits, seed_mask)  # (B, L, 2)

        # If per-bit temperature is provided, compute per-position probs
        if per_bit_temperature is not None:
            # per_bit_temperature: (128,) tensor on same device
            # Reshape for broadcasting: (1, L, 1)
            pbt = per_bit_temperature.to(logits.device)
            temp_expanded = pbt.unsqueeze(0).unsqueeze(-1).clamp(min=1e-6)  # (1, 128, 1)
            probs = F.softmax(logits / temp_expanded, dim=-1)  # (B, L, 2)
        else:
            probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)

        B = bits.size(0)
        for b in range(B):
            for pos in range(start, end):
                if bits[b, pos].item() != MASK_ID:
                    continue
                # If outside target prefix length, set to 0
                if plen is not None and pos >= int(plen[b].item()):
                    bits[b, pos] = 0
                    continue
                # If BGP constraint covers this position (< bgp_len), skip (already set)
                if bgp_len is not None and pos < int(bgp_len[b].item()):
                    continue
                # Sample from per-position probability
                p = probs[b, pos]
                sampled = torch.multinomial(p, 1).item()
                bits[b, pos] = sampled
        return bits


def load_entropy_weights(path: str, seq_len: int = 128) -> Optional[torch.Tensor]:
    """
    Load bit-entropy weights from exp1_result.json.
    Returns a tensor of shape (seq_len,) with normalized weights (mean = 1).
    """
    import json
    try:
        with open(path) as f:
            data = json.load(f)
        ent = data["bit_entropy_per_position"]
        t = torch.tensor([max(0.0, e) for e in ent], dtype=torch.float32)
        # Normalize: mean = 1 so total loss magnitude is comparable to uniform weighting
        mean = t.mean().clamp(min=1e-6)
        return (t / mean)
    except Exception:
        return None
