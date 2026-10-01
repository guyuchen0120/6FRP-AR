"""
6FRP-AR v3 configuration — single-server, CPU-only.

Key changes from v2:
  - device defaults to "cpu" (no GPU on the target server)
  - APDConfig gains `concurrency` for parallel in-prefix probing
  - New OutputConfig for structured per-stage output
"""


class ModelConfig(object):
    """Unified Transformer generator + verifier config."""
    def __init__(self):
        # Metadata vocab
        self.num_as = 100000
        self.num_org = 50000
        self.num_cat1 = 64
        self.num_cat2 = 256
        self.num_proto = 4
        self.num_geo = 10000

        # Embedding dims
        self.as_embed_dim = 64
        self.org_embed_dim = 32
        self.cat1_embed_dim = 16
        self.cat2_embed_dim = 16
        self.proto_embed_dim = 16
        self.geo_embed_dim = 16
        self.rtt_feat_dim = 3

        # Unified Transformer
        self.d_model = 256
        self.num_heads = 8
        self.num_layers = 6
        self.ff_dim = 1024
        self.dropout = 0.1

        # Three-layer structure (bit ranges)
        self.coarse_end = 48   # bits [0, 48)
        self.medium_end = 64   # bits [48, 64)
        # fine = bits [64, 128)

        self.seq_len = 128
        self.vocab_size = 3    # 0, 1, MASK (=2)

        # Few-shot seed encoder (for Mode A)
        self.max_seeds_per_batch = 8
        self.seed_encoder_layers = 2

        # Verifier
        self.verifier_d_model = 128
        self.verifier_num_layers = 4
        self.verifier_num_heads = 8
        self.verifier_ensemble_size = 5
        self.verifier_max_seq_len = 128


class TrainConfig(object):
    """Training and online finetune."""
    def __init__(self):
        self.batch_size = 64
        self.learning_rate = 2e-4
        self.weight_decay = 1e-5
        self.num_epochs = 50
        self.warmup_steps = 1000
        self.num_workers = 2  # DataLoader workers for CPU parallelism

        # Stage mask probability
        self.mask_coarse_prob = 0.33
        self.mask_medium_prob = 0.33
        self.mask_fine_prob = 0.34

        # Entropy-weighted loss
        self.use_entropy_weighting = True
        self.entropy_file = ""

        # Online feedback (three-tier buffers)
        self.buffer_a_size = 32
        self.buffer_b_size = 200
        self.buffer_c_size = 5000

        self.verifier_update_interval = 5
        self.reptile_update_interval = 10
        self.standard_finetune_interval = 1000

        # Reptile hyperparameters
        self.reptile_inner_steps = 3
        self.reptile_inner_lr = 1e-4
        self.reptile_epsilon = 0.1

        # Verifier online lr
        self.verifier_online_lr = 5e-5

        # Active learning weights
        self.al_uncertainty_weight = 0.5
        self.al_diversity_weight = 0.3
        self.al_representativeness_weight = 0.2
        self.al_budget_per_round = 200
        self.al_candidates_per_round = 1000


class APDConfig(object):
    """APD probing parameters."""
    def __init__(self):
        self.probe_count = 16
        self.timeout = 3.0
        self.inter_probe_delay = 0.05
        self.extra_rounds = 2
        self.success_rate_threshold = 0.875
        self.concurrency = 16  # parallel probes within a single prefix


class AlignConfig(object):
    """Tree alignment parameters."""
    def __init__(self):
        self.min_prefix_len = 16
        self.max_prefix_len = 128
        self.lateral_budget = 16
        self.max_tree_depth = 10
        self.dedupe_siblings = True


class OutputConfig(object):
    """Structured output configuration."""
    def __init__(self):
        self.output_dir = "output"
        # Per-stage output files (relative to output_dir)
        self.candidates_file = "candidates.json"
        self.probed_file = "probed.json"
        self.final_frps_file = "final_frps.txt"
        self.summary_file = "summary.json"
        self.train_log_file = "train_log.json"


class Config(object):
    def __init__(self):
        self.model = ModelConfig()
        self.train = TrainConfig()
        self.apd = APDConfig()
        self.align = AlignConfig()
        self.output = OutputConfig()
        self.device = "cpu"  # No GPU on target server
        self.seed = 42
