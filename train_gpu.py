#!/usr/bin/env python3
"""
6FRP-AR v3 offline training — GPU-optimized.

Changes from CPU version (train.py):
  - Default device = cuda
  - Mixed precision training (torch.cuda.amp) for faster GPU throughput
  - Higher default num_workers for DataLoader
  - CUDA availability check at startup
  - GPU memory logging
"""
import argparse
import ipaddress
import json
import logging
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from .config import Config
    from .generators.unified import (
        UnifiedFRPGenerator, MASK_ID, load_entropy_weights,
    )
    from .verifier.verifier import EnsembleVerifier, prepare_verifier_input
    from .utils.ipv6_utils import (
        ipv6_prefix_to_bits, bits_to_ipv6_prefix, normalize_prefix,
    )
except ImportError:
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from my_design_final.config import Config
    from my_design_final.generators.unified import (
        UnifiedFRPGenerator, MASK_ID, load_entropy_weights,
    )
    from my_design_final.verifier.verifier import EnsembleVerifier, prepare_verifier_input
    from my_design_final.utils.ipv6_utils import (
        ipv6_prefix_to_bits, bits_to_ipv6_prefix, normalize_prefix,
    )

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Data
# ============================================================================

def load_frp_data(frp_file, metadata_file=None):
    # type: (str, Optional[str]) -> List[Dict]
    """Load FRP prefixes plus optional metadata."""
    prefix_meta = {}  # type: Dict[str, Dict]
    if metadata_file and os.path.exists(metadata_file):
        with open(metadata_file) as f:
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
                    "as_id": int(meta.get("as_id", 0)),
                    "org_id": int(meta.get("org_id", 0)),
                    "cat1_id": int(meta.get("cat1_id", 0)),
                    "cat2_id": int(meta.get("cat2_id", 0)),
                    "proto_id": int(meta.get("proto_id", 0)),
                    "geo_id": int(meta.get("geo_id", 0)),
                    "rtt_stats": meta.get("rtt_stats", [0.0, 0.0, 0.0]),
                },
            })

    logger.info(f"Loaded {len(data)} FRPs (metadata for "
                f"{sum(1 for d in data if any(v for v in d['meta'].values() if v))} of them)")
    return data


class FRPMLMDataset(Dataset):
    """Per-sample: full 128-bit target, prefix length, metadata."""
    def __init__(self, data, seq_len=128):
        # type: (List[Dict], int) -> None
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        bits, plen = ipv6_prefix_to_bits(d["prefix"])
        padded = bits + [0] * (self.seq_len - len(bits))
        meta = d["meta"]
        return {
            "bit_target": torch.tensor(padded[:self.seq_len], dtype=torch.long),
            "prefix_len": torch.tensor(min(plen, self.seq_len), dtype=torch.long),
            "as_id": torch.tensor(meta["as_id"], dtype=torch.long),
            "org_id": torch.tensor(meta["org_id"], dtype=torch.long),
            "cat1_id": torch.tensor(meta["cat1_id"], dtype=torch.long),
            "cat2_id": torch.tensor(meta["cat2_id"], dtype=torch.long),
            "proto_id": torch.tensor(meta["proto_id"], dtype=torch.long),
            "geo_id": torch.tensor(meta["geo_id"], dtype=torch.long),
            "rtt_stats": torch.tensor(meta["rtt_stats"], dtype=torch.float32),
        }


def make_hard_negatives(positives, ratio=1.0):
    # type: (List[Dict], float) -> List[Dict]
    """Random IPv6 prefixes + sibling flips + subnet swaps."""
    pos_set = {p["prefix"] for p in positives}
    negatives = []
    n = int(len(positives) * ratio)
    per_type = n // 3

    # 1) Random in 2000::/3
    for _ in range(per_type):
        high = 0x2000 << 112
        tail = random.getrandbits(125)
        addr = high | tail
        pos = random.choice(positives)
        _, plen = ipv6_prefix_to_bits(pos["prefix"])
        net = ipaddress.IPv6Network((addr, plen), strict=False)
        if str(net) in pos_set:
            continue
        negatives.append({"prefix": str(net), "meta": pos["meta"], "label": 0})

    # 2) Sibling flips
    for _ in range(per_type):
        pos = random.choice(positives)
        bits, plen = ipv6_prefix_to_bits(pos["prefix"])
        new_bits = list(bits)
        flips = random.sample(range(16, max(17, plen)), min(8, plen - 16))
        for p in flips:
            new_bits[p] = 1 - new_bits[p]
        try:
            neg_p = bits_to_ipv6_prefix(new_bits)
        except Exception:
            continue
        if neg_p in pos_set:
            continue
        negatives.append({"prefix": neg_p, "meta": pos["meta"], "label": 0})

    # 3) Parent-hole siblings
    for _ in range(per_type):
        pos = random.choice(positives)
        bits, plen = ipv6_prefix_to_bits(pos["prefix"])
        if plen < 2:
            continue
        new_bits = list(bits)
        sibling_bit = max(plen - 4, 16)
        if sibling_bit < 128:
            new_bits[sibling_bit] = 1 - new_bits[sibling_bit]
        try:
            neg_p = bits_to_ipv6_prefix(new_bits)
        except Exception:
            continue
        if neg_p in pos_set:
            continue
        negatives.append({"prefix": neg_p, "meta": pos["meta"], "label": 0})

    logger.info(f"Constructed {len(negatives)} hard negatives (target {n})")
    return negatives


# ============================================================================
# Generator training (MLM) — GPU with mixed precision
# ============================================================================

def _log_gpu_memory():
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"  GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def train_generator(generator, dataset, cfg, bit_weight=None,
                    val_dataset=None, output_dir="output", start_epoch=0,
                    use_amp=True):
    # type: (UnifiedFRPGenerator, FRPMLMDataset, Config, Optional[torch.Tensor], Optional[FRPMLMDataset], str, int, bool) -> List[Dict]
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size,
                        shuffle=True, num_workers=cfg.train.num_workers,
                        drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size,
                            num_workers=cfg.train.num_workers,
                            pin_memory=True) if val_dataset else None

    optim = torch.optim.AdamW(
        generator.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )
    total_steps = cfg.train.num_epochs * max(1, len(loader))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps)

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if bit_weight is not None:
        bit_weight = bit_weight.to(cfg.device)

    logger.info("=" * 60)
    logger.info(f"Training Unified Generator (MLM) on {cfg.device} [GPU mode]")
    logger.info(f"  Dataset: {len(dataset)} samples, Batch: {cfg.train.batch_size}")
    logger.info(f"  Epochs: {start_epoch+1} -> {cfg.train.num_epochs}")
    logger.info(f"  Workers: {cfg.train.num_workers}")
    logger.info(f"  Mixed Precision (AMP): {use_amp}")
    _log_gpu_memory()
    logger.info("=" * 60)

    train_log = []
    log_path = os.path.join(output_dir, "train_log.json")

    for epoch in range(start_epoch, cfg.train.num_epochs):
        epoch_start = time.time()
        generator.train()
        tot_loss = 0.0
        tot_acc = 0.0
        n = 0

        for batch in loader:
            batch = {k: v.to(cfg.device, non_blocking=True) for k, v in batch.items()}
            meta = {k: batch[k] for k in
                    ["as_id", "org_id", "cat1_id", "cat2_id",
                     "proto_id", "geo_id", "rtt_stats"]}

            optim.zero_grad()

            # Mixed precision forward pass
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = generator.compute_mlm_loss(
                    bit_target=batch["bit_target"],
                    meta=meta,
                    prefix_lens=batch["prefix_len"],
                    bit_weight=bit_weight,
                    mask_stage=None,
                )
                loss = out["loss"]

            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            sched.step()

            tot_loss += loss.item()
            tot_acc += out["bit_acc"]
            n += 1

        epoch_time = time.time() - epoch_start
        avg_loss = tot_loss / max(n, 1)
        avg_acc = tot_acc / max(n, 1)

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": round(avg_loss, 6),
            "train_bit_acc": round(avg_acc, 6),
            "lr": round(sched.get_last_lr()[0], 8),
            "epoch_time_s": round(epoch_time, 1),
        }

        log_line = (f"[Epoch {epoch+1}/{cfg.train.num_epochs}] "
                    f"loss={avg_loss:.4f} acc={avg_acc:.4f} "
                    f"lr={sched.get_last_lr()[0]:.6f} "
                    f"time={epoch_time:.1f}s")

        if val_loader:
            val_loss, val_acc = _eval_generator(generator, val_loader, cfg,
                                                bit_weight, use_amp)
            epoch_record["val_loss"] = round(val_loss, 6)
            epoch_record["val_bit_acc"] = round(val_acc, 6)
            log_line += f" | val_loss={val_loss:.4f} val_acc={val_acc:.4f}"

        # ETA estimation
        remaining = cfg.train.num_epochs - epoch - 1
        eta_s = remaining * epoch_time
        log_line += f" | ETA={eta_s/60:.1f}min"

        logger.info(log_line)
        train_log.append(epoch_record)

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 or (epoch + 1) == cfg.train.num_epochs:
            ckpt_path = os.path.join(output_dir, "generator.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": generator.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "scheduler_state_dict": sched.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            }, ckpt_path)
            # Save train log
            with open(log_path, "w") as f:
                json.dump(train_log, f, indent=2)
            _log_gpu_memory()

    return train_log


def _eval_generator(generator, loader, cfg, bit_weight, use_amp=True):
    generator.eval()
    tot_loss = 0.0
    tot_acc = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(cfg.device, non_blocking=True) for k, v in batch.items()}
            meta = {k: batch[k] for k in
                    ["as_id", "org_id", "cat1_id", "cat2_id",
                     "proto_id", "geo_id", "rtt_stats"]}
            for stage in ("coarse", "medium", "fine"):
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = generator.compute_mlm_loss(
                        batch["bit_target"], meta, batch["prefix_len"],
                        bit_weight=bit_weight, mask_stage=stage,
                    )
                tot_loss += out["loss"].item()
                tot_acc += out["bit_acc"]
                n += 1
    return tot_loss / max(n, 1), tot_acc / max(n, 1)


# ============================================================================
# Verifier training
# ============================================================================

class VerifierDataset(Dataset):
    def __init__(self, samples, seq_len=128):
        # type: (List[Dict], int) -> None
        self.samples = samples
        self.seq_len = seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        bits, plen = ipv6_prefix_to_bits(s["prefix"])
        padded = bits + [2] * (self.seq_len - len(bits))
        meta = s["meta"]
        return {
            "bits": torch.tensor(padded[:self.seq_len], dtype=torch.long),
            "length": torch.tensor(min(plen, self.seq_len), dtype=torch.long),
            "label": torch.tensor(s["label"], dtype=torch.float32),
            "as_id": torch.tensor(meta["as_id"], dtype=torch.long),
            "org_id": torch.tensor(meta["org_id"], dtype=torch.long),
            "cat1_id": torch.tensor(meta["cat1_id"], dtype=torch.long),
            "cat2_id": torch.tensor(meta["cat2_id"], dtype=torch.long),
            "proto_id": torch.tensor(meta["proto_id"], dtype=torch.long),
            "geo_id": torch.tensor(meta["geo_id"], dtype=torch.long),
            "rtt_stats": torch.tensor(meta["rtt_stats"], dtype=torch.float32),
        }


def train_verifier(verifier, generator, positives, negatives, cfg,
                   output_dir="output", use_amp=True):
    # type: (EnsembleVerifier, UnifiedFRPGenerator, List[Dict], List[Dict], Config, str, bool) -> None
    samples = [{**p, "label": 1} for p in positives] + negatives
    random.shuffle(samples)
    n_train = int(0.9 * len(samples))
    train_ds = VerifierDataset(samples[:n_train], cfg.model.verifier_max_seq_len)
    val_ds = VerifierDataset(samples[n_train:], cfg.model.verifier_max_seq_len)
    loader = DataLoader(train_ds, batch_size=cfg.train.batch_size,
                        shuffle=True, num_workers=cfg.train.num_workers,
                        pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size,
                            num_workers=cfg.train.num_workers,
                            pin_memory=True)

    optim = torch.optim.AdamW(
        verifier.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    logger.info("=" * 60)
    logger.info(f"Training Ensemble Verifier [GPU mode, AMP={use_amp}]")
    logger.info("=" * 60)

    for epoch in range(cfg.train.num_epochs):
        epoch_start = time.time()
        verifier.train()
        tot = 0.0
        n = 0
        for batch in loader:
            batch = {k: v.to(cfg.device, non_blocking=True) for k, v in batch.items()}
            meta = {k: batch[k] for k in
                    ["as_id", "org_id", "cat1_id", "cat2_id",
                     "proto_id", "geo_id", "rtt_stats"]}
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_amp):
                    condition = generator.condition_encoder(meta)

            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = verifier.compute_loss(
                    batch["bits"], batch["length"], condition, batch["label"])

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(verifier.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()

            tot += loss.item()
            n += 1

        verifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(cfg.device, non_blocking=True) for k, v in batch.items()}
                meta = {k: batch[k] for k in
                        ["as_id", "org_id", "cat1_id", "cat2_id",
                         "proto_id", "geo_id", "rtt_stats"]}
                with torch.cuda.amp.autocast(enabled=use_amp):
                    condition = generator.condition_encoder(meta)
                    mean_p, _ = verifier(batch["bits"], batch["length"], condition)
                pred = (mean_p > 0.5).float()
                correct += (pred == batch["label"]).sum().item()
                total += batch["label"].size(0)

        val_acc = correct / max(total, 1)
        epoch_time = time.time() - epoch_start
        logger.info(f"[Epoch {epoch+1}/{cfg.train.num_epochs}] "
                    f"loss={tot/max(n,1):.4f} val_acc={val_acc:.4f} "
                    f"time={epoch_time:.1f}s")

    # Save verifier
    ver_ckpt = os.path.join(output_dir, "verifier.pt")
    torch.save(verifier.state_dict(), ver_ckpt)
    logger.info(f"Saved verifier to {ver_ckpt}")
    _log_gpu_memory()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="6FRP-AR v3 offline training (GPU-optimized)")
    parser.add_argument('--frps', required=True,
                        help='Seed FRP file (one prefix per line)')
    parser.add_argument('--metadata', default=None,
                        help='JSON file mapping prefix -> meta dict')
    parser.add_argument('--output-dir', default='output',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--entropy-file', default=None,
                        help='exp1_result.json for entropy weighting')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size (default: 128 for GPU)')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers (default: 4 for GPU)')
    parser.add_argument('--device', default='cuda',
                        help='Device (default: cuda)')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable mixed precision training')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint in output-dir')
    parser.add_argument('--skip-generator', action='store_true')
    parser.add_argument('--skip-verifier', action='store_true')
    parser.add_argument('--vocabs', default=None,
                        help='vocabs.json from prepare.py for embedding size auto-fit')
    args = parser.parse_args()

    # CUDA availability check
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        logger.error("CUDA is not available! Please check your GPU driver and PyTorch installation.")
        logger.error("Falling back to CPU. Use train.py for CPU-optimized training.")
        args.device = 'cpu'

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

    use_amp = not args.no_amp and args.device.startswith('cuda')

    cfg = Config()
    cfg.train.num_epochs = args.epochs
    cfg.train.batch_size = args.batch_size
    cfg.train.learning_rate = args.lr
    cfg.train.num_workers = args.num_workers
    cfg.device = args.device

    # Tighten embedding sizes from vocabs
    if args.vocabs and os.path.exists(args.vocabs):
        with open(args.vocabs) as f:
            vocabs = json.load(f)

        def _pad(n):
            return max(16, int(n * 1.25) + 16)

        cfg.model.num_as = _pad(len(vocabs.get("as_vocab", {})) + 1)
        cfg.model.num_org = _pad(len(vocabs.get("org_vocab", {})) + 1)
        cfg.model.num_cat1 = _pad(len(vocabs.get("cat1_vocab", {})) + 1)
        cfg.model.num_cat2 = _pad(len(vocabs.get("cat2_vocab", {})) + 1)
        cfg.model.num_geo = _pad(len(vocabs.get("country_vocab", {})) + 1)
        logger.info(f"Vocab sizes: as={cfg.model.num_as} org={cfg.model.num_org} "
                    f"cat1={cfg.model.num_cat1} cat2={cfg.model.num_cat2} "
                    f"geo={cfg.model.num_geo}")

    if args.entropy_file:
        cfg.train.entropy_file = args.entropy_file

    os.makedirs(args.output_dir, exist_ok=True)

    # Data
    data = load_frp_data(args.frps, args.metadata)
    n_train = int(0.9 * len(data))
    random.shuffle(data)
    train_data = data[:n_train]
    val_data = data[n_train:]

    train_ds = FRPMLMDataset(train_data, cfg.model.seq_len)
    val_ds = FRPMLMDataset(val_data, cfg.model.seq_len)

    # Bit weight
    bit_weight = None
    if cfg.train.use_entropy_weighting and cfg.train.entropy_file:
        bit_weight = load_entropy_weights(cfg.train.entropy_file, cfg.model.seq_len)
        if bit_weight is not None:
            logger.info(f"Loaded entropy weights from {cfg.train.entropy_file}")

    # Models
    generator = UnifiedFRPGenerator(cfg.model).to(cfg.device)
    verifier = EnsembleVerifier(
        ensemble_size=cfg.model.verifier_ensemble_size,
        max_seq_len=cfg.model.verifier_max_seq_len,
        bit_embed_dim=32,
        hidden_dim=cfg.model.verifier_d_model,
        num_layers=cfg.model.verifier_num_layers,
        num_heads=cfg.model.verifier_num_heads,
        condition_dim=cfg.model.d_model,
        dropout=0.1,
    ).to(cfg.device)

    logger.info(f"Generator params: {sum(p.numel() for p in generator.parameters()):,}")
    logger.info(f"Verifier params: {sum(p.numel() for p in verifier.parameters()):,}")
    _log_gpu_memory()

    # Resume support
    start_epoch = 0
    gen_ckpt_path = os.path.join(args.output_dir, "generator.pt")
    if args.resume and os.path.exists(gen_ckpt_path):
        ckpt = torch.load(gen_ckpt_path, map_location=cfg.device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            generator.load_state_dict(ckpt["model_state_dict"])
            start_epoch = ckpt.get("epoch", 0)
            logger.info(f"Resumed generator from epoch {start_epoch}")
        else:
            generator.load_state_dict(ckpt)
            logger.info("Loaded generator (no epoch info, starting from 0)")

    # Generator training
    if not args.skip_generator:
        train_generator(generator, train_ds, cfg, bit_weight, val_ds,
                        output_dir=args.output_dir, start_epoch=start_epoch,
                        use_amp=use_amp)
    elif os.path.exists(gen_ckpt_path):
        ckpt = torch.load(gen_ckpt_path, map_location=cfg.device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            generator.load_state_dict(ckpt["model_state_dict"])
        else:
            generator.load_state_dict(ckpt)
        logger.info(f"Loaded generator from {gen_ckpt_path}")

    # Verifier training
    if not args.skip_verifier:
        negatives = make_hard_negatives(train_data, ratio=1.0)
        train_verifier(verifier, generator, train_data, negatives, cfg,
                       output_dir=args.output_dir, use_amp=use_amp)

    logger.info("Training complete!")
    logger.info(f"  Output dir: {args.output_dir}")
    logger.info(f"  Generator: {gen_ckpt_path}")
    logger.info(f"  Verifier: {os.path.join(args.output_dir, 'verifier.pt')}")
    _log_gpu_memory()


if __name__ == "__main__":
    main()
