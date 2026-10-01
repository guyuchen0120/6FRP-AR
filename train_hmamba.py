#!/usr/bin/env python3
"""
HMamba FRP Generator — full training script (CPU / GPU compatible).

Features:
  - Auto device selection (cuda → cpu fallback)
  - Optional mixed-precision (AMP) on GPU
  - AS-grouped batch sampler so the inner Mamba sees same-AS sequences
  - Entropy-weighted MLM loss (compatible with bit_entropy_weights.json)
  - Train/val split, 3-stage val (coarse / medium / fine)
  - Resume from checkpoint, periodic checkpointing, JSON training log
  - Best-by-val-loss checkpoint saved separately

Usage examples:

  # CPU
  python3 train_hmamba.py \
      --frps    output/prepared_new/seeds_prepared.txt \
      --meta    output/prepared_new/meta.json \
      --vocabs  output/prepared_new/vocabs.json \
      --entropy output/prepared_new/bit_entropy_weights.json \
      --output  output/hmamba \
      --epochs 30 --batch-size 32 --d-model 128 --num-layers 3 \
      --device cpu

  # GPU (mixed-precision)
  python3 train_hmamba.py \
      --frps    output/prepared_new/seeds_prepared.txt \
      --meta    output/prepared_new/meta.json \
      --vocabs  output/prepared_new/vocabs.json \
      --entropy output/prepared_new/bit_entropy_weights.json \
      --output  output/hmamba \
      --epochs 50 --batch-size 128 --d-model 256 --num-layers 6 \
      --device cuda --num-workers 4
"""
import argparse
import ipaddress
import json
import logging
import math
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler

# Allow running as both a script and a module.
try:
    from .config import Config
    from .generators.hmamba import HMambaFRPGenerator, load_entropy_weights
    from .utils.ipv6_utils import ipv6_prefix_to_bits
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import Config
    from generators.hmamba import HMambaFRPGenerator, load_entropy_weights
    from utils.ipv6_utils import ipv6_prefix_to_bits


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Data
# ============================================================================

def load_frp_data(frp_file: str,
                  meta_file: Optional[str] = None) -> List[Dict]:
    prefix_meta: Dict[str, Dict] = {}
    if meta_file and os.path.exists(meta_file):
        with open(meta_file) as f:
            prefix_meta = json.load(f)

    data = []
    with open(frp_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                net = ipaddress.IPv6Network(line, strict=False)
            except Exception:
                continue
            p = str(net)
            meta = prefix_meta.get(p, {})
            data.append({
                "prefix": p,
                "meta": {
                    "as_id":     int(meta.get("as_id", 0)),
                    "org_id":    int(meta.get("org_id", 0)),
                    "cat1_id":   int(meta.get("cat1_id", 0)),
                    "cat2_id":   int(meta.get("cat2_id", 0)),
                    "proto_id":  int(meta.get("proto_id", 0)),
                    "geo_id":    int(meta.get("geo_id", 0)),
                    "rtt_stats": meta.get("rtt_stats", [0.0, 0.0, 0.0]),
                },
            })
    n_with_meta = sum(1 for d in data if d['meta']['as_id'] != 0)
    logger.info(f"Loaded {len(data)} FRPs (with AS meta: {n_with_meta})")
    return data


class FRPDataset(Dataset):
    def __init__(self, data: List[Dict], seq_len: int = 128):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        bits, plen = ipv6_prefix_to_bits(d["prefix"])
        padded = bits + [0] * (self.seq_len - len(bits))
        m = d["meta"]
        return {
            "bit_target": torch.tensor(padded[:self.seq_len], dtype=torch.long),
            "prefix_len": torch.tensor(min(plen, self.seq_len), dtype=torch.long),
            "as_id":      torch.tensor(m["as_id"], dtype=torch.long),
            "org_id":     torch.tensor(m["org_id"], dtype=torch.long),
            "cat1_id":    torch.tensor(m["cat1_id"], dtype=torch.long),
            "cat2_id":    torch.tensor(m["cat2_id"], dtype=torch.long),
            "proto_id":   torch.tensor(m["proto_id"], dtype=torch.long),
            "geo_id":     torch.tensor(m["geo_id"], dtype=torch.long),
            "rtt_stats":  torch.tensor(m["rtt_stats"], dtype=torch.float32),
        }


class ASGroupedBatchSampler(Sampler):
    """
    Yield batches that contain a few AS groups (multiple samples per AS),
    so the inner Mamba has meaningful within-AS sequences (G > 1).

    Strategy:
      - For each batch:
          * Pick `groups_per_batch` AS keys at random.
          * Allocate batch_size slots roughly evenly among those AS keys.
          * Sample with-or-without replacement inside each group.
      - If a chosen AS group is too small, oversample that group
        (sampling with replacement is allowed).
    """
    def __init__(self, dataset: FRPDataset, batch_size: int,
                 groups_per_batch: int = 4, num_batches: Optional[int] = None,
                 shuffle: bool = True):
        self.batch_size = batch_size
        self.groups_per_batch = max(1, groups_per_batch)
        self.shuffle = shuffle
        self.as_to_indices: Dict[int, List[int]] = defaultdict(list)
        for i, d in enumerate(dataset.data):
            self.as_to_indices[d["meta"]["as_id"]].append(i)
        self.as_keys = list(self.as_to_indices.keys())
        self.num_batches = (num_batches if num_batches is not None
                            else len(dataset) // batch_size)

    def __iter__(self):
        for _ in range(self.num_batches):
            k = min(self.groups_per_batch, len(self.as_keys))
            if self.shuffle:
                groups = random.sample(self.as_keys, k=k)
            else:
                groups = self.as_keys[:k]
            per = max(2, self.batch_size // max(1, len(groups)))
            batch: List[int] = []
            for g in groups:
                pool = self.as_to_indices[g]
                if len(pool) >= per:
                    batch.extend(random.sample(pool, per))
                else:
                    batch.extend(pool)
                    while len(batch) < (groups.index(g) + 1) * per \
                            and len(batch) < self.batch_size:
                        batch.append(random.choice(pool))
            # Trim or pad to exactly batch_size
            while len(batch) < self.batch_size:
                a = random.choice(self.as_keys)
                batch.append(random.choice(self.as_to_indices[a]))
            batch = batch[:self.batch_size]
            yield batch

    def __len__(self):
        return self.num_batches


# ============================================================================
# Train / Eval
# ============================================================================

def _move_to(batch: Dict[str, torch.Tensor], device: str, non_blocking: bool):
    return {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}


def train_one_epoch(model, loader, optim, sched, device, *,
                    bit_weight=None, scaler=None, use_amp=False,
                    log_every: int = 50) -> Dict[str, float]:
    model.train()
    tot_loss = 0.0
    tot_acc = 0.0
    n = 0
    is_cuda = device.startswith("cuda")
    pin = is_cuda
    t_step = time.time()
    for step, batch in enumerate(loader):
        batch = _move_to(batch, device, non_blocking=pin)
        meta = {k: batch[k] for k in
                ["as_id", "org_id", "cat1_id", "cat2_id",
                 "proto_id", "geo_id", "rtt_stats"]}

        optim.zero_grad(set_to_none=True)

        if use_amp and is_cuda:
            with torch.cuda.amp.autocast():
                out = model.compute_mlm_loss(
                    bit_target=batch["bit_target"],
                    meta=meta,
                    prefix_lens=batch["prefix_len"],
                    bit_weight=bit_weight,
                    mask_stage=None)
                loss = out["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
        else:
            out = model.compute_mlm_loss(
                bit_target=batch["bit_target"],
                meta=meta,
                prefix_lens=batch["prefix_len"],
                bit_weight=bit_weight,
                mask_stage=None)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

        if sched is not None:
            sched.step()

        tot_loss += loss.item()
        tot_acc += out["bit_acc"]
        n += 1

        if log_every and (step + 1) % log_every == 0:
            dt = time.time() - t_step
            logger.info(f"  [step {step+1}/{len(loader)}] "
                        f"loss={tot_loss/n:.4f} acc={tot_acc/n:.4f} "
                        f"lr={sched.get_last_lr()[0]:.6f} "
                        f"({log_every/max(dt,1e-6):.1f} steps/s)")
            t_step = time.time()

    return {"loss": tot_loss / max(n, 1), "acc": tot_acc / max(n, 1)}


@torch.no_grad()
def eval_epoch(model, loader, device, *, bit_weight=None,
               use_amp=False) -> Dict[str, float]:
    model.eval()
    tot_loss = 0.0
    tot_acc = 0.0
    n = 0
    is_cuda = device.startswith("cuda")
    for batch in loader:
        batch = _move_to(batch, device, non_blocking=is_cuda)
        meta = {k: batch[k] for k in
                ["as_id", "org_id", "cat1_id", "cat2_id",
                 "proto_id", "geo_id", "rtt_stats"]}
        for stage in ("coarse", "medium", "fine"):
            if use_amp and is_cuda:
                with torch.cuda.amp.autocast():
                    out = model.compute_mlm_loss(
                        bit_target=batch["bit_target"],
                        meta=meta,
                        prefix_lens=batch["prefix_len"],
                        bit_weight=bit_weight,
                        mask_stage=stage)
            else:
                out = model.compute_mlm_loss(
                    bit_target=batch["bit_target"],
                    meta=meta,
                    prefix_lens=batch["prefix_len"],
                    bit_weight=bit_weight,
                    mask_stage=stage)
            tot_loss += out["loss"].item()
            tot_acc += out["bit_acc"]
            n += 1
    return {"loss": tot_loss / max(n, 1), "acc": tot_acc / max(n, 1)}


# ============================================================================
# Checkpoint helpers
# ============================================================================

def _build_ckpt_dict(model, optim, sched, scaler, epoch, cfg) -> Dict:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim.state_dict() if optim else None,
        "scheduler_state_dict": sched.state_dict() if sched else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "cfg": {
            "d_model":     cfg.model.d_model,
            "num_layers":  cfg.model.num_layers,
            "seq_len":     cfg.model.seq_len,
            "vocab_size":  cfg.model.vocab_size,
            "coarse_end":  cfg.model.coarse_end,
            "medium_end":  cfg.model.medium_end,
            "num_as":      cfg.model.num_as,
            "num_org":     cfg.model.num_org,
            "num_cat1":    cfg.model.num_cat1,
            "num_cat2":    cfg.model.num_cat2,
            "num_proto":   cfg.model.num_proto,
            "num_geo":     cfg.model.num_geo,
            "as_embed_dim":    cfg.model.as_embed_dim,
            "org_embed_dim":   cfg.model.org_embed_dim,
            "cat1_embed_dim":  cfg.model.cat1_embed_dim,
            "cat2_embed_dim":  cfg.model.cat2_embed_dim,
            "proto_embed_dim": cfg.model.proto_embed_dim,
            "geo_embed_dim":   cfg.model.geo_embed_dim,
            "rtt_feat_dim":    cfg.model.rtt_feat_dim,
        },
    }


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser("HMamba full trainer (CPU/GPU)")
    p.add_argument("--frps", required=True, help="seeds_prepared.txt")
    p.add_argument("--meta", default=None, help="meta.json")
    p.add_argument("--vocabs", default=None, help="vocabs.json")
    p.add_argument("--entropy", default=None, help="bit_entropy_weights.json")
    p.add_argument("--output", default="output/hmamba")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--ssm-d-state", type=int, default=16)
    p.add_argument("--ssm-d-conv", type=int, default=4)
    p.add_argument("--device", default="auto",
                   help="auto | cpu | cuda | cuda:0 ...")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--groups-per-batch", type=int, default=4,
                   help="how many AS groups to mix per batch (inner Mamba)")
    p.add_argument("--no-as-sampler", action="store_true",
                   help="disable AS-grouped sampler (random shuffle)")
    p.add_argument("--no-amp", action="store_true",
                   help="disable mixed precision (only relevant on cuda)")
    p.add_argument("--resume", action="store_true",
                   help="resume from output/last.pt")
    p.add_argument("--save-every", type=int, default=5,
                   help="save full ckpt every N epochs")
    p.add_argument("--log-every", type=int, default=50,
                   help="log every N steps within an epoch")
    p.add_argument("--val-ratio", type=float, default=0.1)
    return p.parse_args()


def resolve_device(arg: str) -> str:
    if arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if arg.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU.")
        return "cpu"
    return arg


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    is_cuda = device.startswith("cuda")
    use_amp = is_cuda and not args.no_amp

    if is_cuda:
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} "
                    f"({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB)")
    else:
        logger.info("Device: CPU")

    # ---- Config ----
    cfg = Config()
    cfg.device = device
    cfg.train.batch_size = args.batch_size
    cfg.train.learning_rate = args.lr
    cfg.train.weight_decay = args.weight_decay
    cfg.train.num_epochs = args.epochs
    cfg.train.num_workers = args.num_workers
    cfg.model.num_layers = args.num_layers
    cfg.model.d_model = args.d_model
    cfg.model.ssm_d_state = args.ssm_d_state
    cfg.model.ssm_d_conv = args.ssm_d_conv

    if args.vocabs and os.path.exists(args.vocabs):
        with open(args.vocabs) as f:
            v = json.load(f)
        def _pad(n): return max(16, int(n * 1.25) + 16)
        cfg.model.num_as   = _pad(len(v.get("as_vocab", {})) + 1)
        cfg.model.num_org  = _pad(len(v.get("org_vocab", {})) + 1)
        cfg.model.num_cat1 = _pad(len(v.get("cat1_vocab", {})) + 1)
        cfg.model.num_cat2 = _pad(len(v.get("cat2_vocab", {})) + 1)
        cfg.model.num_geo  = _pad(len(v.get("country_vocab", {})) + 1)
        logger.info(f"Vocab: as={cfg.model.num_as} org={cfg.model.num_org} "
                    f"cat1={cfg.model.num_cat1} cat2={cfg.model.num_cat2} "
                    f"geo={cfg.model.num_geo}")

    os.makedirs(args.output, exist_ok=True)

    # ---- Data ----
    data = load_frp_data(args.frps, args.meta)
    if not data:
        logger.error("No data loaded; aborting.")
        return
    random.shuffle(data)
    n_val = max(1, int(args.val_ratio * len(data)))
    val_data = data[:n_val]
    train_data = data[n_val:]
    train_ds = FRPDataset(train_data, cfg.model.seq_len)
    val_ds = FRPDataset(val_data, cfg.model.seq_len)

    if args.no_as_sampler:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, drop_last=True, pin_memory=is_cuda)
    else:
        sampler = ASGroupedBatchSampler(
            train_ds, args.batch_size,
            groups_per_batch=args.groups_per_batch)
        train_loader = DataLoader(
            train_ds, batch_sampler=sampler,
            num_workers=args.num_workers, pin_memory=is_cuda)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=is_cuda)

    # ---- Bit weight ----
    bit_weight = None
    if args.entropy and os.path.exists(args.entropy):
        bit_weight = load_entropy_weights(args.entropy, cfg.model.seq_len)
        if bit_weight is not None:
            bit_weight = bit_weight.to(device)
            logger.info(f"Loaded entropy weights from {args.entropy}")

    # ---- Model ----
    model = HMambaFRPGenerator(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"HMamba — params: {n_params/1e6:.2f} M, "
                f"d_model={cfg.model.d_model}, layers={cfg.model.num_layers}, "
                f"d_state={cfg.model.ssm_d_state}, AMP={use_amp}")

    optim = torch.optim.AdamW(model.parameters(),
                              lr=args.lr,
                              weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader)) * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if is_cuda else None

    # ---- Resume ----
    last_ckpt = os.path.join(args.output, "last.pt")
    best_ckpt = os.path.join(args.output, "best.pt")
    log_path = os.path.join(args.output, "train_log.json")
    train_log: List[Dict] = []
    start_epoch = 0
    best_val = float("inf")

    if args.resume and os.path.exists(last_ckpt):
        ck = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ck["model_state_dict"])
        if ck.get("optimizer_state_dict") is not None:
            optim.load_state_dict(ck["optimizer_state_dict"])
        if ck.get("scheduler_state_dict") is not None:
            sched.load_state_dict(ck["scheduler_state_dict"])
        if scaler is not None and ck.get("scaler_state_dict") is not None:
            try:
                scaler.load_state_dict(ck["scaler_state_dict"])
            except Exception:
                pass
        start_epoch = int(ck.get("epoch", 0))
        if os.path.exists(log_path):
            try:
                with open(log_path) as f:
                    train_log = json.load(f)
                vals = [r.get("val_loss", math.inf) for r in train_log]
                best_val = min(vals) if vals else math.inf
            except Exception:
                pass
        logger.info(f"Resumed from {last_ckpt} (epoch {start_epoch}, "
                    f"best_val={best_val:.4f})")

    # ---- Train loop ----
    logger.info("=" * 70)
    logger.info(f"Training on {len(train_data)} samples / {len(val_data)} val "
                f"| {len(train_loader)} steps/epoch | bs={args.batch_size}")
    logger.info("=" * 70)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optim, sched, device,
                             bit_weight=bit_weight, scaler=scaler,
                             use_amp=use_amp, log_every=args.log_every)
        va = eval_epoch(model, val_loader, device,
                        bit_weight=bit_weight, use_amp=use_amp)
        dt = time.time() - t0
        rec = {
            "epoch": epoch + 1,
            "train_loss": round(tr["loss"], 6),
            "train_acc":  round(tr["acc"], 6),
            "val_loss":   round(va["loss"], 6),
            "val_acc":    round(va["acc"], 6),
            "lr":         round(sched.get_last_lr()[0], 8),
            "time_s":     round(dt, 1),
        }
        train_log.append(rec)
        eta_min = (args.epochs - epoch - 1) * dt / 60
        logger.info(f"[E{epoch+1}/{args.epochs}] "
                    f"train_loss={tr['loss']:.4f} acc={tr['acc']:.4f} | "
                    f"val_loss={va['loss']:.4f} acc={va['acc']:.4f} | "
                    f"time={dt:.1f}s ETA={eta_min:.1f}min")

        # Save logs / last ckpt
        with open(log_path, "w") as f:
            json.dump(train_log, f, indent=2)
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            torch.save(_build_ckpt_dict(model, optim, sched, scaler,
                                        epoch + 1, cfg), last_ckpt)
            logger.info(f"  saved last ckpt -> {last_ckpt}")

        # Save best
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(_build_ckpt_dict(model, None, None, None,
                                        epoch + 1, cfg), best_ckpt)
            logger.info(f"  *new best* val_loss={best_val:.4f} -> {best_ckpt}")

        if is_cuda:
            torch.cuda.empty_cache()

    logger.info(f"Done. Best val_loss={best_val:.4f}")
    logger.info(f"  best ckpt: {best_ckpt}")
    logger.info(f"  last ckpt: {last_ckpt}")
    logger.info(f"  log:       {log_path}")


if __name__ == "__main__":
    main()
