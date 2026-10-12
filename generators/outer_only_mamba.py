# outer_only_mamba.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from hmamba import MambaBlock, ConditionEncoder, selective_scan, MASK_ID

class OuterOnlyMambaLayer(nn.Module):
    """
    仅包含外层扫描的Mamba层（没有内层扫描和门控融合）
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.outer = MambaBlock(d_model, d_state=d_state, d_conv=d_conv)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, as_ids: torch.Tensor,
                as_cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:       (B, L, d_model) per-sample bit features
            as_ids:  (B,) long tensor — AS id of each sample (not used here)
            as_cond: (B, d_model)   — AS conditioning vector (not used here)

        Returns:
            (B, L, d_model)
        """
        # 仅执行外层扫描
        out_outer = self.outer(x)  # (B, L, D)
        return self.norm(out_outer + x)  # 残差连接


class OuterOnlyMambaFRPGenerator(nn.Module):
    """
    仅包含外层扫描的FRP生成器
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.bit_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.layer_embed = nn.Embedding(3, cfg.d_model)

        self.condition_encoder = ConditionEncoder(cfg)

        # 仅外层Mamba堆栈
        d_state = getattr(cfg, "ssm_d_state", 16)
        d_conv = getattr(cfg, "ssm_d_conv", 4)
        self.layers = nn.ModuleList([
            OuterOnlyMambaLayer(cfg.d_model, d_state=d_state, d_conv=d_conv)
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

    def forward(self, bit_seq: torch.Tensor,
                meta: Dict[str, torch.Tensor],
                seed_bits: Optional[torch.Tensor] = None,
                seed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        bit_seq: (B, 128)
        meta:    dict of metadata tensors
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