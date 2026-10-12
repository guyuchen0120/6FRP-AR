#!/usr/bin/env python3
"""
Debug script for DualScanHMamba training issues.
"""
import os
import sys
import torch
import json

# Add project root to path
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from generators.dual_scan_hmamba import DualScanHMamba, DualScanConfig

def test_model_creation():
    """Test basic model creation and forward pass."""
    print("=== Test 1: Model Creation ===")
    
    # Create minimal config
    cfg = DualScanConfig(
        d_model=128,
        num_layers=2,  # Use fewer layers for debugging
        num_as=300,
        num_org=273,
        num_cat1=15,
        num_cat2=24,
        num_geo=51,
    )
    
    model = DualScanHMamba(cfg)
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model, cfg

def test_forward_pass(model, cfg):
    """Test model forward pass with dummy data."""
    print("\n=== Test 2: Forward Pass ===")
    
    # Create dummy batch
    batch_size = 4
    bits = torch.randint(0, 3, (batch_size, cfg.seq_len))  # 0,1,2 (MASK)
    
    meta = {
        "as_id": torch.randint(0, cfg.num_as, (batch_size,)),
        "org_id": torch.randint(0, cfg.num_org, (batch_size,)),
        "cat1_id": torch.randint(0, cfg.num_cat1, (batch_size,)),
        "cat2_id": torch.randint(0, cfg.num_cat2, (batch_size,)),
        "proto_id": torch.randint(0, 4, (batch_size,)),
        "geo_id": torch.randint(0, cfg.num_geo, (batch_size,)),
        "rtt_stats": torch.randn(batch_size, 3),
    }
    
    prefix_lens = torch.randint(32, 96, (batch_size,))
    
    print(f"✓ Created dummy batch: bits={bits.shape}, meta keys={list(meta.keys())}")
    
    # Test forward pass
    try:
        with torch.no_grad():
            logits = model(bits, meta)
            print(f"✓ Forward pass successful: logits shape={logits.shape}")
            
            # Test MLM loss
            result = model.compute_mlm_loss(bits, meta, prefix_lens)
            print(f"✓ MLM loss successful: loss={result['loss'].item():.4f}")
            
            return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """Test data loading from the actual dataset."""
    print("\n=== Test 3: Data Loading ===")
    
    try:
        from train_dual_scan import FRPDataset, collate_fn
        
        # Test with small subset
        dataset = FRPDataset(
            "data_fixed/seeds_prepared.txt",
            "data_fixed/meta.json",
            seq_len=128
        )
        
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        
        # Test single sample
        sample = dataset[0]
        print(f"✓ Sample loaded: keys={list(sample.keys())}")
        
        # Test collate function
        mini_batch = [dataset[i] for i in range(2)]
        batch = collate_fn(mini_batch)
        print(f"✓ Collate function successful: batch keys={list(batch.keys())}")
        
        return True
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_step(model, cfg):
    """Test a single training step."""
    print("\n=== Test 4: Training Step ===")
    
    try:
        import torch.optim as optim
        
        # Create optimizer
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        
        # Create dummy data
        batch_size = 2
        bits = torch.randint(0, 3, (batch_size, cfg.seq_len))
        meta = {
            "as_id": torch.randint(0, cfg.num_as, (batch_size,)),
            "org_id": torch.randint(0, cfg.num_org, (batch_size,)),
            "cat1_id": torch.randint(0, cfg.num_cat1, (batch_size,)),
            "cat2_id": torch.randint(0, cfg.num_cat2, (batch_size,)),
            "proto_id": torch.randint(0, 4, (batch_size,)),
            "geo_id": torch.randint(0, cfg.num_geo, (batch_size,)),
            "rtt_stats": torch.randn(batch_size, 3),
        }
        prefix_lens = torch.randint(32, 96, (batch_size,))
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        result = model.compute_mlm_loss(bits, meta, prefix_lens)
        loss = result["loss"]
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        print(f"✓ Training step successful: loss={loss.item():.4f}")
        return True
        
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("DualScanHMamba Debug Script")
    print("=" * 50)
    
    # Test 1: Model creation
    model, cfg = test_model_creation()
    
    # Test 2: Forward pass
    forward_ok = test_forward_pass(model, cfg)
    
    # Test 3: Data loading
    data_ok = test_data_loading()
    
    # Test 4: Training step
    training_ok = test_training_step(model, cfg)
    
    print("\n" + "=" * 50)
    print("DEBUG SUMMARY:")
    print(f"  Model Creation: {'✓' if model else '✗'}")
    print(f"  Forward Pass: {'✓' if forward_ok else '✗'}")
    print(f"  Data Loading: {'✓' if data_ok else '✗'}")
    print(f"  Training Step: {'✓' if training_ok else '✗'}")
    
    if all([forward_ok, data_ok, training_ok]):
        print("\n🎉 All tests passed! The issue might be in the full training loop.")
        print("Try running the training script again with the fixes applied.")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()