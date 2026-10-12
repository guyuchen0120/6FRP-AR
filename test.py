#!/usr/bin/env python3
"""
Training script for DualScanHMamba FRP Generator.

Loads preprocessed seed data from data_fixed/ and trains the model
using Masked Language Modeling (MLM) with layer-wise masking.

Usage:
    python3 train_dual_scan.py --data-dir data_fixed --output-dir checkpoints/dual_scan
    python3 train_dual_scan.py --data-dir data_fixed --output-dir checkpoints/dual_scan --epochs 50 --batch-size 128
"""
import argparse
import ipaddress
import json
import logging
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add project root to path
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here) if os.path.basename(_here) != "6FRP" else _here
if _root not in sys.path:
    sys.path.insert(0, _root)

from generators.dual_scan_hmamba import (
    DualScanHMamba, DualScanConfig, load_entropy_weights, MASK_ID
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Dataset
# ============================================================================

def prefix_to_bits(prefix_str: str, seq_len: int = 128) -> List[int]:
    """Convert an IPv6 prefix string to a list of bits (length=seq_len)."""
    try:
        net = ipaddress.IPv6Network(prefix_str, strict=False)
    except Exception:
        return [0] * seq_len
    addr_int = int(net.network_address)
    bits = []
    for i in range(seq_len):
        bits.append((addr_int >> (127 - i)) & 1)
    return bits


class FRPDataset(Dataset):
    """
    Dataset for FRP training.

    Loads seeds from seeds_prepared.txt and metadata from meta.json.
    Each sample is a (bit_sequence, metadata, prefix_length) tuple.
    """
    def __init__(self, seeds_path: str, meta_path: str, seq_len: int = 128):
        super().__init__()
        self.seq_len = seq_len

        # Load seeds
        self.prefixes = []
        with open(seeds_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    self.prefixes.append(line)

        # Load metadata
        with open(meta_path) as f:
            self.meta_db = json.load(f)

        # Build samples
        self.samples = []
        for prefix in self.prefixes:
            if prefix in self.meta_db:
                meta = self.meta_db[prefix]
                try:
                    net = ipaddress.IPv6Network(prefix, strict=False)
                    prefix_len = net.prefixlen
                except Exception:
                    prefix_len = 48
                self.samples.append({
                    "prefix": prefix,
                    "prefix_len": prefix_len,
                    "meta": meta,
                })

        logger.info(f"FRPDataset: {len(self.samples)} samples loaded "
                    f"(from {len(self.prefixes)} prefixes)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        bits = prefix_to_bits(sample["prefix"], self.seq_len)
        meta = sample["meta"]

        return {
            "bits": torch.tensor(bits, dtype=torch.long),
            "prefix_len": torch.tensor(sample["prefix_len"], dtype=torch.long),
            "as_id": torch.tensor(meta["as_id"], dtype=torch.long),
            "org_id": torch.tensor(meta["org_id"], dtype=torch.long),
            "cat1_id": torch.tensor(meta["cat1_id"], dtype=torch.long),
            "cat2_id": torch.tensor(meta["cat2_id"], dtype=torch.long),
            "proto_id": torch.tensor(meta["proto_id"], dtype=torch.long),
            "geo_id": torch.tensor(meta["geo_id"], dtype=torch.long),
            "rtt_stats": torch.tensor(meta["rtt_stats"], dtype=torch.float32),
        }


def collate_fn(batch):
    """Collate batch samples into tensors."""
    return {
        "bits": torch.stack([b["bits"] for b in batch]),
        "prefix_len": torch.stack([b["prefix_len"] for b in batch]),
        "as_id": torch.stack([b["as_id"] for b in batch]),
        "org_id": torch.stack([b["org_id"] for b in batch]),
        "cat1_id": torch.stack([b["cat1_id"] for b in batch]),
        "cat2_id": torch.stack([b["cat2_id"] for b in batch]),
        "proto_id": torch.stack([b["proto_id"] for b in batch]),
        "geo_id": torch.stack([b["geo_id"] for b in batch]),
        "rtt_stats": torch.stack([b["rtt_stats"] for b in batch]),
    }


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(model: DualScanHMamba, dataloader: DataLoader,
                optimizer: optim.Optimizer, scheduler,
                bit_weight: Optional[torch.Tensor],
                device: torch.device, epoch: int) -> Dict:
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_bits = 0
    n_batches = 0
    stage_counts = {"coarse": 0, "medium": 0, "fine": 0}

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        bits = batch["bits"].to(device)
        prefix_lens = batch["prefix_len"].to(device)
        meta = {
            "as_id": batch["as_id"].to(device),
            "org_id": batch["org_id"].to(device),
            "cat1_id": batch["cat1_id"].to(device),
            "cat2_id": batch["cat2_id"].to(device),
            "proto_id": batch["proto_id"].to(device),
            "geo_id": batch["geo_id"].to(device),
            "rtt_stats": batch["rtt_stats"].to(device),
        }

        bw = bit_weight.to(device) if bit_weight is not None else None

        # Forward + loss
        result = model.compute_mlm_loss(
            bit_target=bits,
            meta=meta,
            prefix_lens=prefix_lens,
            bit_weight=bw,
        )

        loss = result["loss"]

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_acc += result["bit_acc"]
        total_bits += result["num_masked_bits"]
        stage_counts[result["mask_stage"]] += 1
        n_batches += 1

        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / n_batches
            avg_acc = total_acc / n_batches
            logger.info(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(dataloader)} | "
                        f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")

    return {
        "loss": total_loss / max(n_batches, 1),
        "bit_acc": total_acc / max(n_batches, 1),
        "total_masked_bits": total_bits,
        "stage_counts": stage_counts,
        "n_batches": n_batches,
    }


def validate(model: DualScanHMamba, dataloader: DataLoader,
             bit_weight: Optional[torch.Tensor],
             device: torch.device) -> Dict:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            bits = batch["bits"].to(device)
            prefix_lens = batch["prefix_len"].to(device)
            meta = {
                "as_id": batch["as_id"].to(device),
                "org_id": batch["org_id"].to(device),
                "cat1_id": batch["cat1_id"].to(device),
                "cat2_id": batch["cat2_id"].to(device),
                "proto_id": batch["proto_id"].to(device),
                "geo_id": batch["geo_id"].to(device),
                "rtt_stats": batch["rtt_stats"].to(device),
            }

            bw = bit_weight.to(device) if bit_weight is not None else None

            # Evaluate all 3 stages
            for stage in ["coarse", "medium", "fine"]:
                result = model.compute_mlm_loss(
                    bit_target=bits,
                    meta=meta,
                    prefix_lens=prefix_lens,
                    bit_weight=bw,
                    mask_stage=stage,
                )
                total_loss += result["loss"].item()
                total_acc += result["bit_acc"]
                n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "bit_acc": total_acc / max(n_batches, 1),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train DualScanHMamba FRP Generator")
    parser.add_argument('--data-dir', default='data_fixed',
                        help='Directory with preprocessed data')
    parser.add_argument('--output-dir', default='checkpoints/dual_scan',
                        help='Output directory for checkpoints')
    parser.add_argument('--entropy-file', default=None,
                        help='Path to bit_entropy_weights.json (optional)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--val-split', type=float, default=0.05,
                        help='Validation split ratio')
    parser.add_argument('--d-model', type=int, default=128,
                        help='Model dimension')
    parser.add_argument('--num-layers', type=int, default=4,
                        help='Number of DualScan layers')
    parser.add_argument('--ssm-d-state', type=int, default=16,
                        help='SSM state dimension')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save-every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--resume', default=None,
                        help='Resume from checkpoint path')
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Paths
    data_dir = args.data_dir
    seeds_path = os.path.join(data_dir, "seeds_prepared.txt")
    meta_path = os.path.join(data_dir, "meta.json")
    vocabs_path = os.path.join(data_dir, "vocabs.json")
    stats_path = os.path.join(data_dir, "stats.json")

    # Load vocabs for config
    with open(vocabs_path) as f:
        vocabs = json.load(f)
    with open(stats_path) as f:
        stats = json.load(f)

    # Build config
    cfg = DualScanConfig(
        d_model=args.d_model,
        num_layers=args.num_layers,
        ssm_d_state=args.ssm_d_state,
        seq_len=128,  # IPv6 prefix length is 128 bits
        num_as=stats["vocab_sizes"]["as"],
        num_org=stats["vocab_sizes"]["org"],
        num_cat1=stats["vocab_sizes"]["cat1"],
        num_cat2=stats["vocab_sizes"]["cat2"],
        num_geo=stats["vocab_sizes"]["country (geo)"],
    )

    logger.info(f"Model config: d_model={cfg.d_model}, layers={cfg.num_layers}, "
                f"ssm_d_state={cfg.ssm_d_state}")
    logger.info(f"Vocab sizes: AS={cfg.num_as}, org={cfg.num_org}, "
                f"cat1={cfg.num_cat1}, cat2={cfg.num_cat2}, geo={cfg.num_geo}")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = FRPDataset(seeds_path, meta_path, seq_len=cfg.seq_len)

    # # Train/val split
    # n_val = int(len(dataset) * args.val_split)
    # n_train = len(dataset) - n_val
    # train_dataset, val_dataset = torch.utils.data.random_split(
    #     dataset, [n_train, n_val],
    #     generator=torch.Generator().manual_seed(args.seed))

    # logger.info(f"Train: {n_train}, Val: {n_val}")

    # train_loader = DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=True,
    #     collate_fn=collate_fn, num_workers=args.num_workers,
    #     pin_memory=(device.type == 'cuda'),  # Only pin memory for GPU
    #     drop_last=True)
    # val_loader = DataLoader(
    #     val_dataset, batch_size=args.batch_size, shuffle=False,
    #     collate_fn=collate_fn, num_workers=args.num_workers,
    #     pin_memory=(device.type == 'cuda'),  # Only pin memory for GPU
    #     drop_last=False)

    # # Load entropy weights
    # bit_weight = None
    # if args.entropy_file and os.path.exists(args.entropy_file):
    #     bit_weight = load_entropy_weights(args.entropy_file, cfg.seq_len)
    #     if bit_weight is not None:
    #         logger.info(f"Loaded entropy weights from {args.entropy_file}")

    # # Build model
    # model = DualScanHMamba(cfg).to(device)
    # n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # logger.info(f"Model parameters: {n_params:,}")

    # # Add model debugging
    # logger.info(f"Model device: {next(model.parameters()).device}")
    # logger.info(f"Input shape test: batch_size={args.batch_size}, seq_len={cfg.seq_len}")

    # # Optimizer + scheduler
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr,
    #                         weight_decay=args.weight_decay)
    # total_steps = len(train_loader) * args.epochs
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=total_steps, eta_min=args.lr * 0.01)

    # # Resume
    # start_epoch = 0
    # if args.resume and os.path.exists(args.resume):
    #     ckpt = torch.load(args.resume, map_location=device)
    #     model.load_state_dict(ckpt["model_state_dict"])
    #     optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    #     start_epoch = ckpt.get("epoch", 0) + 1
    #     logger.info(f"Resumed from {args.resume} (epoch {start_epoch})")

    # # Output directory
    # os.makedirs(args.output_dir, exist_ok=True)

    # # Save config
    # cfg_path = os.path.join(args.output_dir, "config.json")
    # with open(cfg_path, 'w') as f:
    #     json.dump(cfg.to_dict(), f, indent=2)

    # # Test a single batch before starting training
    # logger.info("Testing single batch...")
    # try:
    #     test_batch = next(iter(train_loader))
    #     logger.info(f"Batch keys: {list(test_batch.keys())}")
    #     logger.info(f"Batch shapes: {{k: v.shape for k, v in test_batch.items()}}")
        
    #     # Test model forward pass
    #     test_bits = test_batch["bits"][:2].to(device)  # Use smaller batch for test
    #     test_meta = {k: v[:2].to(device) for k, v in test_batch.items() 
    #                  if k not in ["bits", "prefix_len"]}
    #     test_prefix_lens = test_batch["prefix_len"][:2].to(device)
        
    #     with torch.no_grad():
    #         test_result = model.compute_mlm_loss(
    #             bit_target=test_bits,
    #             meta=test_meta,
    #             prefix_lens=test_prefix_lens,
    #         )
    #     logger.info(f"Test forward pass successful: loss={test_result['loss'].item():.4f}")
    # except Exception as e:
    #     logger.error(f"Test failed: {e}")
    #     logger.error("Stopping training due to test failure")
    #     return

    # # Training loop
    # logger.info("=" * 60)
    # logger.info("Starting training...")
    # logger.info("=" * 60)

    # best_val_loss = float('inf')
    # train_history = []

    # for epoch in range(start_epoch, args.epochs):
    #     t0 = time.time()

    #     # Train
    #     train_metrics = train_epoch(
    #         model, train_loader, optimizer, scheduler,
    #         bit_weight, device, epoch)

    #     # Validate
    #     val_metrics = validate(model, val_loader, bit_weight, device)

    #     elapsed = time.time() - t0

    #     logger.info(
    #         f"Epoch {epoch}/{args.epochs-1} ({elapsed:.1f}s) | "
    #         f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['bit_acc']:.4f} | "
    #         f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['bit_acc']:.4f} | "
    #         f"Stages: C={train_metrics['stage_counts']['coarse']} "
    #         f"M={train_metrics['stage_counts']['medium']} "
    #         f"F={train_metrics['stage_counts']['fine']}")

    #     train_history.append({
    #         "epoch": epoch,
    #         "train_loss": train_metrics["loss"],
    #         "train_acc": train_metrics["bit_acc"],
    #         "val_loss": val_metrics["loss"],
    #         "val_acc": val_metrics["bit_acc"],
    #         "elapsed": elapsed,
    #     })

    #     # Save best model
    #     if val_metrics["loss"] < best_val_loss:
    #         best_val_loss = val_metrics["loss"]
    #         best_path = os.path.join(args.output_dir, "best_model.pt")
    #         torch.save({
    #             "epoch": epoch,
    #             "model_state_dict": model.state_dict(),
    #             "optimizer_state_dict": optimizer.state_dict(),
    #             "config": cfg.to_dict(),
    #             "val_loss": val_metrics["loss"],
    #             "val_acc": val_metrics["bit_acc"],
    #         }, best_path)
    #         logger.info(f"  -> New best model saved (val_loss={best_val_loss:.4f})")

    #     # Periodic checkpoint
    #     if (epoch + 1) % args.save_every == 0:
    #         ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pt")
    #         torch.save({
    #             "epoch": epoch,
    #             "model_state_dict": model.state_dict(),
    #             "optimizer_state_dict": optimizer.state_dict(),
    #             "config": cfg.to_dict(),
    #         }, ckpt_path)

    # # Save final model
    # final_path = os.path.join(args.output_dir, "final_model.pt")
    # torch.save({
    #     "epoch": args.epochs - 1,
    #     "model_state_dict": model.state_dict(),
    #     "optimizer_state_dict": optimizer.state_dict(),
    #     "config": cfg.to_dict(),
    # }, final_path)

    # # Save training history
    # history_path = os.path.join(args.output_dir, "train_history.json")
    # with open(history_path, 'w') as f:
    #     json.dump(train_history, f, indent=2)

    # logger.info("=" * 60)
    # logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    # logger.info(f"Checkpoints saved to: {args.output_dir}")
    # logger.info("=" * 60)


if __name__ == "__main__":
    main()
