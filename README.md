# 6FRP-AR v3 — Single-Server IPv6 FRP Detection

## Overview

Single-server architecture: all components (training, inference, probing) run on
one Linux server with IPv6 global address. No GPU required.

## Architecture

```
my_design_final/
├── config.py                      # Config (device=cpu, concurrency=16)
├── data_prep/
│   └── prepare.py                 # Seed ↔ AS/BGP binding
├── generators/
│   └── unified.py                 # Unified Transformer (MLM + staged gen)
├── verifier/
│   └── verifier.py                # Ensemble Verifier (5-member)
├── alignment/
│   └── tree_align.py              # TreeAligner (upward + lateral)
├── active_learning/
│   └── sampler.py                 # Uncertainty + Diversity + Representativeness
├── online/
│   └── feedback.py                # Three-tier buffers + Reptile fast-adapt
├── modes/
│   ├── seed_expand.py             # Mode A: Seed Expansion
│   └── target.py                  # Mode B: Target Discovery
├── utils/
│   ├── ipv6_utils.py              # IPv6 bit/prefix helpers
│   ├── apd_core.py                # ICMPv6 probe kernel (concurrent)
│   └── apd.py                     # LocalAPDProber + SimulatedAPDProber
├── train.py                       # Offline training (CPU, resume support)
├── infer.py                       # Inference entry (local probing)
└── requirements.txt               # torch>=2.0.0
```

## Key Differences from v2

| v2 (my_design_new) | v3 (my_design_final) |
|---------------------|---------------------|
| Dual-server: GPU host + Prober host | Single server (no GPU) |
| HTTP communication (aiohttp/requests) | Direct local ICMPv6 calls |
| `RemoteAPDProber` + `apd_server.py` | `LocalAPDProber` (no HTTP) |
| device=cuda | device=cpu |
| No checkpoint resume | `--resume` support |
| Single result.json | Structured output (candidates/probed/frps/summary) |

## Quick Start

### 1. Server Prerequisites

```bash
# Verify IPv6 connectivity
ping6 -c 3 2001:4860:4860::8888

# Verify root access (needed for raw ICMPv6 sockets)
whoami  # should be root

# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Data Preparation

```bash
cd /path/to/project

# Prepare seeds with BGP/AS metadata binding
sudo python -m my_design_final.data_prep.prepare \
    --bgp data/20250123.1600.dat \
    --as-csv data/as_org_category.csv \
    --seeds data/YOUR_SEEDS.txt \
    --out-dir output/prepared
```

Output:
- `output/prepared/meta.json` — per-seed metadata
- `output/prepared/as_seed_db.json` — AS → seeds mapping
- `output/prepared/vocabs.json` — vocabulary tables
- `output/prepared/seeds_prepared.txt` — normalized seed list
- `output/prepared/stats.json` — preparation statistics

### 3. Training (CPU)

```bash
# Full training (~2-4 hours on 4-core CPU with ~1000 seeds)
sudo python -m my_design_final train \
    --frps output/prepared/seeds_prepared.txt \
    --metadata output/prepared/meta.json \
    --vocabs output/prepared/vocabs.json \
    --output-dir output \
    --epochs 50 --batch-size 64 --lr 2e-4 \
    --num-workers 2

# Resume interrupted training
sudo python -m my_design_final train \
    --frps output/prepared/seeds_prepared.txt \
    --metadata output/prepared/meta.json \
    --vocabs output/prepared/vocabs.json \
    --output-dir output \
    --epochs 50 --resume

# Background training with nohup
nohup sudo python -m my_design_final train \
    --frps output/prepared/seeds_prepared.txt \
    --metadata output/prepared/meta.json \
    --vocabs output/prepared/vocabs.json \
    --output-dir output \
    --epochs 50 --batch-size 64 \
    > output/train.log 2>&1 &
```

Output:
- `output/generator.pt` — generator checkpoint (with optimizer state for resume)
- `output/verifier.pt` — verifier checkpoint
- `output/train_log.json` — per-epoch metrics (loss, accuracy, time, ETA)

### 4. Inference — Mode A (Seed Expand)

```bash
sudo python -m my_design_final infer \
    --mode seed_expand \
    --seeds output/prepared/seeds_prepared.txt \
    --meta 'as_id=1234' \
    --checkpoint-dir output \
    --output-dir output/results_expand \
    --candidates 1000 --budget 200
```

### 5. Inference — Mode B (Target Discovery)

```bash
sudo python -m my_design_final infer \
    --mode target \
    --targets data/bgp_prefixes.txt \
    --per-bgp-meta output/prepared/meta.json \
    --as-seed-db output/prepared/as_seed_db.json \
    --checkpoint-dir output \
    --output-dir output/results_target \
    --target-plen 64
```
当BGP前缀没有直接对应的AS信息时，可以通过其子前缀的AS和meta信息来进行模仿生成。
优先使用直接的meta信息：如果BGP前缀本身在meta.json中有信息，则直接使用。
从子前缀获取AS信息：如果没有直接的meta信息，则查找所有属于该BGP前缀的子前缀，并使用它们的AS信息。
从父前缀获取AS信息：如果也没有子前缀信息，则查找该BGP前缀所属的父前缀。
统计最常见的AS ID：如果有多个子/父前缀，会选择最常见的AS ID作为代表。
保留原有逻辑：同时保留了原有的父子关系检测逻辑，将重叠的种子添加到结果中。
这样，即使某个BGP前缀没有直接的AS信息，系统也能通过其子前缀的AS信息进行模仿生成，从而提高发现类似FRP的可能性。

### 6. Offline Testing (No Root/IPv6 Needed)

```bash
# Use simulated prober for development
python -m my_design_final infer \
    --mode seed_expand \
    --simulate data/known_frps.txt \
    --seeds data/test_seeds.txt \
    --checkpoint-dir output \
    --output-dir output/results_sim
```

## Output Structure

After inference, `output-dir/` contains:

```
output/results_expand/
├── candidates.json      # Generated candidates with verifier scores
├── probed.json          # Probe results with ICMPv6 classification
├── final_frps.txt       # Confirmed FRPs (one per line)
└── summary.json         # Run statistics and per-BGP breakdown
```

## Performance Notes

- **Training**: ~50 epochs on 1000 seeds takes 2-4 hours on a 4-core CPU
- **Probing**: With `concurrency=16`, single prefix takes ~3.8s (vs 48s serial)
- **Memory**: Peak ~2GB RAM during training (batch_size=64)

## Troubleshooting

### All probes return TIMEOUT
```bash
# Check IPv6 connectivity
ping6 -c 3 2001:4860:4860::8888

# Check ICMPv6 rate limiting
sysctl net.ipv6.icmp.ratelimit
sudo sysctl -w net.ipv6.icmp.ratelimit=0  # disable for testing
```

### Permission denied
```bash
# Must run as root for raw sockets
sudo python -m my_design_final infer ...

# Or set capability (alternative to root)
sudo setcap cap_net_raw+ep $(which python3)
```

### Training too slow
```bash
# Reduce model size for faster iteration
# Edit config.py: num_layers=4, d_model=128

# Or reduce epochs
sudo python -m my_design_final train --epochs 20 ...
```
