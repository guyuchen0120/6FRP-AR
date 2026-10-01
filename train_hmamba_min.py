#!/usr/bin/env python3
"""
Minimal runnable training script for Hierarchical Mamba (HMamba) FRP generator.

Usage:
    python train_hmamba_min.py \
        --frps   output/prepared_new/seeds_prepared.txt \
        --meta   output/prepared_new/meta.json \
        --vocabs output/prepared_new/vocabs.json \
        --entropy output/prepared_new/bit_entropy_weights.json \
        --output output/hmamba_min \
        --epochs 5 --batch-size 32 --lr 2e-4 --device cpu

bash```
python3 train_hmamba_min.py --frps output/prepared_new/seeds_prepared.txt --meta output/prepared_new/meta.json --vocabs output/prepared_new/vocabs.json --entropy output/prepared_new/bit_entropy_weights.json --output output/hmamba_min --epochs 5 --batch-size 32 --lr 2e-4 --device cpu
```


Notes:
    - Pure PyTorch (no mamba-ssm CUDA dep). Works on CPU.
    - Trains only the generator (no verifier) — minimal recipe.
    - Reuses data formats produced by data_prep/prepare.py.
"""
import argparse
import ipaddress
import json
import logging
import os
import random
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Run as a module:  python -m my_design_final.train_hmamba_min ...
# Or as a script with relative imports gracefully handled.
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


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def load_frp_data(frp_file: str, meta_file: Optional[str] = None) -> List[Dict]:
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
    logger.info(f"Loaded {len(data)} FRPs (with meta: "
                f"{sum(1 for d in data if d['meta']['as_id'] != 0)})")
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


class ASGroupedSampler(torch.utils.data.Sampler):
    """
    Sampler that prefers placing samples of the same AS in the same batch,
    so the inner Mamba has meaningful within-AS sequences.

    Strategy: for each batch, pick a few AS groups uniformly at random and
    fill the batch from those groups (with replacement if needed).
    """
    def __init__(self, dataset: FRPDataset, batch_size: int,
                 groups_per_batch: int = 4, shuffle: bool = True):
        self.batch_size = batch_size
        self.groups_per_batch = groups_per_batch
        self.shuffle = shuffle

        self.as_to_indices: Dict[int, List[int]] = {}
        for i, d in enumerate(dataset.data):
            a = d["meta"]["as_id"]
            self.as_to_indices.setdefault(a, []).append(i)
        self.as_keys = list(self.as_to_indices.keys())
        self.num_samples = len(dataset)

    def __iter__(self):
        n_batches = self.num_samples // self.batch_size
        for _ in range(n_batches):
            if self.shuffle:
                groups = random.sample(
                    self.as_keys,
                    k=min(self.groups_per_batch, len(self.as_keys)))
            else:
                groups = self.as_keys[:self.groups_per_batch]
            batch = []
            per = max(1, self.batch_size // max(1, len(groups)))
            for g in groups:
                pool = self.as_to_indices[g]
                k = min(per, len(pool))
                batch.extend(random.sample(pool, k))
            # Fill up to batch_size
            while len(batch) < self.batch_size:
                a = random.choice(self.as_keys)
                batch.append(random.choice(self.as_to_indices[a]))
            for idx in batch[:self.batch_size]:
                yield idx

    def __len__(self):
        return (self.num_samples // self.batch_size) * self.batch_size


# ----------------------------------------------------------------------------
# Train loop
# ----------------------------------------------------------------------------

def train_one_epoch(model, loader, optim, sched, device,
                    bit_weight=None) -> Dict[str, float]:
    model.train()
    tot_loss = 0.0
    tot_acc = 0.0
    n = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        meta = {k: batch[k] for k in
                ["as_id", "org_id", "cat1_id", "cat2_id",
                 "proto_id", "geo_id", "rtt_stats"]}
        out = model.compute_mlm_loss(
            bit_target=batch["bit_target"],
            meta=meta,
            prefix_lens=batch["prefix_len"],
            bit_weight=bit_weight,
            mask_stage=None)
        loss = out["loss"]
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        if sched is not None:
            sched.step()
        tot_loss += loss.item()
        tot_acc += out["bit_acc"]
        n += 1
    return {"loss": tot_loss / max(n, 1), "acc": tot_acc / max(n, 1)}


@torch.no_grad()
def eval_epoch(model, loader, device, bit_weight=None) -> Dict[str, float]:
    model.eval()
    tot_loss = 0.0
    tot_acc = 0.0
    n = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        meta = {k: batch[k] for k in
                ["as_id", "org_id", "cat1_id", "cat2_id",
                 "proto_id", "geo_id", "rtt_stats"]}
        for stage in ("coarse", "medium", "fine"):
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


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser("HMamba minimal trainer")
    p.add_argument("--frps", required=True, help="seeds_prepared.txt")
    p.add_argument("--meta", default=None, help="meta.json (prefix -> meta dict)")
    p.add_argument("--vocabs", default=None, help="vocabs.json from prepare.py")
    p.add_argument("--entropy", default=None, help="bit_entropy_weights.json")
    p.add_argument("--output", default="output/hmamba_min")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num-layers", type=int, default=3,
                   help="HMamba stack depth (default 3, smaller for CPU)")
    p.add_argument("--d-model", type=int, default=128,
                   help="model dim (default 128 for CPU)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers (default 0; AS sampler is "
                        "single-process-friendly)")
    p.add_argument("--groups-per-batch", type=int, default=4,
                   help="how many AS groups to mix per batch")
    p.add_argument("--no-as-sampler", action="store_true",
                   help="disable AS-grouped sampler (use random shuffle)")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = Config()
    cfg.device = args.device
    cfg.train.batch_size = args.batch_size
    cfg.train.learning_rate = args.lr
    cfg.train.num_epochs = args.epochs
    cfg.model.num_layers = args.num_layers
    cfg.model.d_model = args.d_model

    # Tighten vocab sizes from vocabs.json (compact ids)
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

    # Data
    data = load_frp_data(args.frps, args.meta)
    random.shuffle(data)
    n_train = int(0.9 * len(data))
    train_data, val_data = data[:n_train], data[n_train:]
    train_ds = FRPDataset(train_data, cfg.model.seq_len)
    val_ds = FRPDataset(val_data, cfg.model.seq_len)

    if args.no_as_sampler:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  drop_last=True)
    else:
        sampler = ASGroupedSampler(train_ds, args.batch_size,
                                   groups_per_batch=args.groups_per_batch)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=sampler, num_workers=args.num_workers,
                                  drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    # Bit entropy weights
    bit_weight = None
    if args.entropy and os.path.exists(args.entropy):
        bit_weight = load_entropy_weights(args.entropy, cfg.model.seq_len)
        if bit_weight is not None:
            bit_weight = bit_weight.to(args.device)
            logger.info(f"Loaded bit-entropy weights from {args.entropy}")

    # Model
    model = HMambaFRPGenerator(cfg.model).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"HMamba model — params: {n_params/1e6:.2f} M, "
                f"d_model={cfg.model.d_model}, layers={cfg.model.num_layers}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    total_steps = max(1, len(train_loader)) * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps)

    log = []
    best_val = float("inf")
    ckpt_path = os.path.join(args.output, "hmamba_generator.pt")
    log_path = os.path.join(args.output, "train_log.json")

    for epoch in range(args.epochs):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optim, sched,
                             args.device, bit_weight)
        va = eval_epoch(model, val_loader, args.device, bit_weight)
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
        log.append(rec)
        logger.info(f"[E{epoch+1}/{args.epochs}] "
                    f"train_loss={tr['loss']:.4f} train_acc={tr['acc']:.4f} "
                    f"| val_loss={va['loss']:.4f} val_acc={va['acc']:.4f} "
                    f"| time={dt:.1f}s")

        # Save best
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "cfg": {
                    "d_model": cfg.model.d_model,
                    "num_layers": cfg.model.num_layers,
                    "num_as": cfg.model.num_as,
                    "num_org": cfg.model.num_org,
                    "num_cat1": cfg.model.num_cat1,
                    "num_cat2": cfg.model.num_cat2,
                    "num_geo": cfg.model.num_geo,
                },
            }, ckpt_path)
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

    logger.info(f"Done. Best val_loss={best_val:.4f}; ckpt={ckpt_path}")


if __name__ == "__main__":
    main()
