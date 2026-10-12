# compare_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
from hmamba import HMambaFRPGenerator
from outer_only_mamba import OuterOnlyMambaFRPGenerator
from inner_only_mamba import InnerOnlyMambaFRPGenerator

def create_sample_data(batch_size=8, seq_len=128):
    """创建示例数据"""
    # 创建随机比特序列 (0, 1, MASK_ID=2)
    bit_seq = torch.randint(0, 2, (batch_size, seq_len))
    
    # 随机应用一些MASK
    mask_ratio = 0.3
    mask_positions = torch.rand(batch_size, seq_len) < mask_ratio
    bit_seq[mask_positions] = 2  # MASK_ID
    
    # 创建元数据
    meta = {
        'as_id': torch.randint(0, 10, (batch_size,)),  # 10个不同的AS
        'org_id': torch.randint(0, 20, (batch_size,)),
        'cat1_id': torch.randint(0, 5, (batch_size,)),
        'cat2_id': torch.randint(0, 8, (batch_size,)),
        'proto_id': torch.randint(0, 3, (batch_size,)),
        'geo_id': torch.randint(0, 15, (batch_size,)),
        'rtt_stats': torch.randn(batch_size, 5)  # 5维RTT统计特征
    }
    
    # 目标序列（未被mask的部分作为监督信号）
    target_seq = torch.randint(0, 2, (batch_size, seq_len)).long()
    prefix_lens = torch.randint(seq_len//2, seq_len, (batch_size,))
    
    return bit_seq, meta, target_seq, prefix_lens

def train_single_epoch(model, data, optimizer, criterion):
    """训练单个epoch"""
    model.train()
    bit_seq, meta, target_seq, prefix_lens = data
    
    optimizer.zero_grad()
    
    result = model.compute_mlm_loss(
        bit_target=target_seq,
        meta=meta,
        prefix_lens=prefix_lens
    )
    
    loss = result['loss']
    accuracy = result['bit_acc']
    
    loss.backward()
    optimizer.step()
    
    return loss.item(), accuracy

def evaluate_model(model, test_data):
    """评估模型"""
    model.eval()
    bit_seq, meta, target_seq, prefix_lens = test_data
    
    with torch.no_grad():
        result = model.compute_mlm_loss(
            bit_target=target_seq,
            meta=meta,
            prefix_lens=prefix_lens
        )
    
    return result['loss'].item(), result['bit_acc']

def main():
    # 创建配置
    class Config:
        def __init__(self):
            self.d_model = 32  # 减小尺寸以便快速测试
            self.seq_len = 128
            self.vocab_size = 3  # {0, 1, MASK}
            self.num_layers = 2  # 减少层数以便快速测试
            self.num_as = 50
            self.num_org = 100
            self.num_cat1 = 10
            self.num_cat2 = 20
            self.num_proto = 5
            self.num_geo = 50
            self.as_embed_dim = 8
            self.org_embed_dim = 8
            self.cat1_embed_dim = 4
            self.cat2_embed_dim = 4
            self.proto_embed_dim = 4
            self.geo_embed_dim = 8
            self.rtt_feat_dim = 5
            self.coarse_end = 32
            self.medium_end = 64
            self.ssm_d_state = 4
            self.ssm_d_conv = 3

    cfg = Config()
    
    # 创建三个模型
    print("创建模型...")
    full_model = HMambaFRPGenerator(cfg)
    outer_only_model = OuterOnlyMambaFRPGenerator(cfg)
    inner_only_model = InnerOnlyMambaFRPGenerator(cfg)
    
    models = {
        'Full HMamba': full_model,
        'Outer Only': outer_only_model,
        'Inner Only': inner_only_model
    }
    
    # 创建优化器
    optimizers = {
        name: torch.optim.Adam(model.parameters(), lr=1e-3)
        for name, model in models.items()
    }
    
    # 创建数据
    print("创建数据...")
    train_data = create_sample_data(batch_size=8, seq_len=128)
    test_data = create_sample_data(batch_size=4, seq_len=128)
    
    # 训练和记录结果
    print("开始对比实验...")
    
    epochs = 10
    results = {name: {'train_losses': [], 'train_accs': [], 'val_losses': [], 'val_accs': []} 
               for name in models.keys()}
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        for name, model in models.items():
            # 训练
            train_loss, train_acc = train_single_epoch(
                model, train_data, optimizers[name], nn.CrossEntropyLoss()
            )
            results[name]['train_losses'].append(train_loss)
            results[name]['train_accs'].append(train_acc)
            
            # 验证
            val_loss, val_acc = evaluate_model(model, test_data)
            results[name]['val_losses'].append(val_loss)
            results[name]['val_accs'].append(val_acc)
            
            print(f"  {name}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
    
    # 可视化结果
    print("\n绘制对比结果...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 训练损失
    for name in models.keys():
        axes[0, 0].plot(results[name]['train_losses'], label=name)
    axes[0, 0].set_title('Training Loss Comparison')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 训练准确率
    for name in models.keys():
        axes[0, 1].plot(results[name]['train_accs'], label=name)
    axes[0, 1].set_title('Training Accuracy Comparison')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 验证损失
    for name in models.keys():
        axes[1, 0].plot(results[name]['val_losses'], label=name)
    axes[1, 0].set_title('Validation Loss Comparison')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 验证准确率
    for name in models.keys():
        axes[1, 1].plot(results[name]['val_accs'], label=name)
    axes[1, 1].set_title('Validation Accuracy Comparison')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 最终性能总结
    print("\n最终性能总结:")
    print("-" * 60)
    for name in models.keys():
        final_train_loss = results[name]['train_losses'][-1]
        final_train_acc = results[name]['train_accs'][-1]
        final_val_loss = results[name]['val_losses'][-1]
        final_val_acc = results[name]['val_accs'][-1]
        
        print(f"{name:12s} | Train Loss: {final_train_loss:.4f} | Train Acc: {final_train_acc:.4f} | "
              f"Val Loss: {final_val_loss:.4f} | Val Acc: {final_val_acc:.4f}")
    
    # 分析各模型的特点
    print("\n模型分析:")
    print("-" * 60)
    print("Full HMamba: 结合了全局(外层)和局部(内层)模式捕捉，理论上能最好地建模FRP结构")
    print("Outer Only: 仅捕捉全局序列依赖关系，适合建模整体FRP结构")
    print("Inner Only: 仅捕捉同一AS内部的模式，适合建模AS特有的FRP特征")
    
    # 找出最佳模型
    best_model = max(models.keys(), key=lambda name: results[name]['val_accs'][-1])
    print(f"\n验证集最佳模型: {best_model} (准确率: {results[best_model]['val_accs'][-1]:.4f})")

if __name__ == "__main__":
    main()