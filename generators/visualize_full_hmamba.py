# visualize_full_hmamba_with_save.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
from hmamba import HMambaFRPGenerator, HierarchicalMambaLayer, MambaBlock, selective_scan
import seaborn as sns
import os
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class HMambaVisualizer:
    def __init__(self, model: HMambaFRPGenerator, save_dir: str = "./chart"):
        self.model = model
        self.activations = {}
        self.hooks = []
        self.save_dir = save_dir
        
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成时间戳标识符
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def register_hooks(self):
        """注册钩子函数以捕获中间激活值"""
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = {
                    'input': input[0].detach().cpu() if isinstance(input, tuple) else input.detach().cpu(),
                    'output': output.detach().cpu()
                }
            return hook
        
        # 注册各个层的钩子
        for i, layer in enumerate(self.model.layers):
            self.hooks.append(layer.register_forward_hook(get_activation(f'layer_{i}')))
            self.hooks.append(layer.outer.register_forward_hook(get_activation(f'outer_block_{i}')))
            self.hooks.append(layer.inner.register_forward_hook(get_activation(f'inner_block_{i}')))

    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def save_figure(self, fig, filename):
        """保存图片到指定目录"""
        filepath = os.path.join(self.save_dir, f"{filename}_{self.timestamp}.png")
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {filepath}")

    def visualize_forward_pass(self, bit_seq: torch.Tensor, meta: Dict[str, torch.Tensor]):
        """可视化前向传播过程"""
        print("="*60)
        print("Start HMamba Forward Pass Visualization")
        print("="*60)
        
        # 清空之前的激活值
        self.activations.clear()
        
        # 执行前向传播
        with torch.no_grad():
            logits = self.model.forward(bit_seq, meta)
        
        print(f"Input shape: {bit_seq.shape}")
        print(f"Meta shape: {[v.shape for v in meta.values()]}")
        print(f"Output logits shape: {logits.shape}")
        
        # 可视化嵌入层
        self.visualize_embeddings(bit_seq, meta)
        
        # 可视化每一层的激活
        self.visualize_layer_activations()
        
        # 可视化门控融合机制
        self.visualize_gating_mechanism(meta)
        
        return logits

    def visualize_embeddings(self, bit_seq: torch.Tensor, meta: Dict[str, torch.Tensor]):
        """可视化嵌入层"""
        print("\n--- Embedding Visualization ---")
        
        # 显示输入比特序列
        print(f"Input bit sequence sample (first 10 bits): {bit_seq[0, :10].tolist()}")
        
        # 计算并可视化嵌入
        B = bit_seq.size(0)
        device = bit_seq.device
        pos = torch.arange(self.model.cfg.seq_len, device=device)
        
        bit_embedded = self.model.bit_embed(bit_seq)  # (B, L, d_model)
        pos_embedded = self.model.pos_embed(pos).unsqueeze(0)  # (1, L, d_model)
        layer_embedded = self.model.layer_embed(self.model.layer_of_bit).unsqueeze(0)  # (1, L, d_model)
        
        print(f"Bit embedding shape: {bit_embedded.shape}")
        print(f"Position embedding shape: {pos_embedded.shape}")
        print(f"Layer embedding shape: {layer_embedded.shape}")
        
        # 条件嵌入
        cond = self.model.condition_encoder(meta)  # (B, d_model)
        print(f"Condition embedding shape: {cond.shape}")
        
        # 可视化嵌入组合
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 比特嵌入热力图 - 修复：添加 .detach()
        im1 = axes[0,0].imshow(bit_embedded[0, :32, :16].detach().cpu().numpy(), aspect='auto', cmap='viridis')
        axes[0,0].set_title('Bit Embedding (First 32 positions, First 16 dimensions)')
        axes[0,0].set_xlabel('Dimension')
        axes[0,0].set_ylabel('Position')
        plt.colorbar(im1, ax=axes[0,0])
        
        # 位置嵌入热力图 - 修复：添加 .detach()
        im2 = axes[0,1].imshow(pos_embedded[0, :32, :16].detach().cpu().numpy(), aspect='auto', cmap='plasma')
        axes[0,1].set_title('Position Embedding (First 32 positions, First 16 dimensions)')
        axes[0,1].set_xlabel('Dimension')
        axes[0,1].set_ylabel('Position')
        plt.colorbar(im2, ax=axes[0,1])
        
        # 层级嵌入热力图 - 修复：添加 .detach()
        im3 = axes[1,0].imshow(layer_embedded[0, :32, :16].detach().cpu().numpy(), aspect='auto', cmap='coolwarm')
        axes[1,0].set_title('Layer Embedding (First 32 positions, First 16 dimensions)')
        axes[1,0].set_xlabel('Dimension')
        axes[1,0].set_ylabel('Position')
        plt.colorbar(im3, ax=axes[1,0])
        
        # 条件嵌入热力图 - 修复：添加 .detach()
        im4 = axes[1,1].imshow(cond[:4, :16].detach().cpu().numpy(), aspect='auto', cmap='RdYlBu')
        axes[1,1].set_title('Condition Embedding (First 4 samples, First 16 dimensions)')
        axes[1,1].set_xlabel('Dimension')
        axes[1,1].set_ylabel('Sample')
        plt.colorbar(im4, ax=axes[1,1])
        
        plt.tight_layout()
        self.save_figure(fig, "embeddings")
        plt.show()

    def visualize_layer_activations(self):
        """可视化每一层的激活"""
        print("\n--- Layer Activation Visualization ---")
        
        n_layers = len(self.model.layers)
        fig, axes = plt.subplots(n_layers, 2, figsize=(15, 5*n_layers))
        
        if n_layers == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_layers):
            layer_key = f'layer_{i}'
            if layer_key in self.activations:
                layer_out = self.activations[layer_key]['output']
                
                # 显示该层的输出 - 修复：添加 .detach()
                im1 = axes[i, 0].imshow(layer_out[0, :32, :16].detach().cpu().numpy(), 
                                        aspect='auto', cmap='viridis')
                axes[i, 0].set_title(f'Layer {i} Output Activation (First 32 positions, First 16 dimensions)')
                axes[i, 0].set_xlabel('Dimension')
                axes[i, 0].set_ylabel('Position')
                plt.colorbar(im1, ax=axes[i, 0])
                
                # 显示该层的梯度变化（近似）
                if i > 0:
                    prev_key = f'layer_{i-1}'
                    if prev_key in self.activations:
                        prev_out = self.activations[prev_key]['output']
                        # 计算差异 - 修复：添加 .detach()
                        diff = torch.abs(layer_out.detach() - prev_out.detach()).mean(dim=-1)  # 平均每维的变化
                        im2 = axes[i, 1].imshow(diff[0, :32].unsqueeze(1).cpu().numpy(), 
                                                aspect='auto', cmap='hot', vmin=0)
                        axes[i, 1].set_title(f'Layer {i} Activation Change Intensity')
                        axes[i, 1].set_xlabel('Position')
                        axes[i, 1].set_ylabel('Change Intensity')
                        plt.colorbar(im2, ax=axes[i, 1])
        
        plt.tight_layout()
        self.save_figure(fig, "layer_activations")
        plt.show()

    def visualize_gating_mechanism(self, meta: Dict[str, torch.Tensor]):
        """可视化门控融合机制"""
        print("\n--- Gating Fusion Mechanism Visualization ---")
        
        # 重新运行一次获取门控权重
        with torch.no_grad():
            # 获取条件编码
            cond = self.model.condition_encoder(meta)  # (B, d_model)
            
            # 计算门控权重
            gate_weights = torch.softmax(self.model.layers[0].gate(cond), dim=-1)  # (B, 2)
            
            outer_weights = gate_weights[:, 0]  # 外层权重
            inner_weights = gate_weights[:, 1]  # 内层权重
            
            print(f"Gating weights shape: {gate_weights.shape}")
            print(f"First 5 samples outer weights: {outer_weights[:5].tolist()}")
            print(f"First 5 samples inner weights: {inner_weights[:5].tolist()}")
        
        # 可视化门控权重分布
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 外层权重分布 - 修复：添加 .detach()
        axes[0].hist(outer_weights.detach().cpu().numpy(), bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0].set_title('Outer Scan Weight Distribution')
        axes[0].set_xlabel('Weight Value')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        # 内层权重分布 - 修复：添加 .detach()
        axes[1].hist(inner_weights.detach().cpu().numpy(), bins=20, alpha=0.7, color='red', edgecolor='black')
        axes[1].set_title('Inner Scan Weight Distribution')
        axes[1].set_xlabel('Weight Value')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, "gating_weights")
        plt.show()
        
        # 按AS ID分组显示权重
        as_ids = meta['as_id'].cpu().numpy()
        unique_as = np.unique(as_ids)
        
        print(f"\nAverage gating weights by AS:")
        for as_id in unique_as[:5]:  # 只显示前5个AS
            mask = as_ids == as_id
            # 修复：添加 .detach()
            avg_outer = outer_weights.detach()[mask].mean().item()
            avg_inner = inner_weights.detach()[mask].mean().item()
            count = mask.sum()
            print(f"AS {as_id}: Outer={avg_outer:.3f}, Inner={avg_inner:.3f}, Count={count}")

    def visualize_training_process(self, dataloader, criterion, optimizer, epochs=50):  # 修改：从3改为50
        """可视化训练过程"""
        print("\n" + "="*60)
        print("Start Training Process Visualization")
        print("="*60)
        
        self.model.train()
        
        # 记录训练指标
        losses = []
        accuracies = []
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        for epoch in range(epochs):
            epoch_losses = []
            epoch_accuracies = []
            
            for batch_idx, (bit_seqs, meta, targets, prefix_lens) in enumerate(dataloader):
                if batch_idx >= 10:  # 限制每个epoch的批次数以加快可视化
                    break
                
                optimizer.zero_grad()
                
                # 计算MLM损失
                result = self.model.compute_mlm_loss(
                    bit_target=targets,
                    meta=meta,
                    prefix_lens=prefix_lens
                )
                
                loss = result['loss']
                accuracy = result['bit_acc']
                
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
                epoch_accuracies.append(accuracy)
                
                # 每5个批次打印一次进度
                if batch_idx % 5 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {accuracy:.4f}")
            
            losses.extend(epoch_losses)
            accuracies.extend(epoch_accuracies)
            
            # 实时更新图表
            ax1.clear()
            ax1.plot(losses, label='Training Loss', color='blue')
            ax1.set_title('Training Loss Over Time')
            ax1.set_xlabel('Batch')
            ax1.set_ylabel('Loss')
            ax1.grid(True, alpha=0.3)
            
            ax2.clear()
            ax2.plot(accuracies, label='Accuracy', color='green')
            ax2.set_title('Bit Accuracy Over Time')
            ax2.set_xlabel('Batch')
            ax2.set_ylabel('Accuracy')
            ax2.grid(True, alpha=0.3)
            
            ax3.clear()
            ax3.hist(epoch_losses, bins=15, alpha=0.7, color='blue', edgecolor='black')
            ax3.set_title(f'Loss Distribution - Epoch {epoch+1}')
            ax3.set_xlabel('Loss Value')
            ax3.set_ylabel('Frequency')
            
            ax4.clear()
            ax4.hist(epoch_accuracies, bins=15, alpha=0.7, color='green', edgecolor='black')
            ax4.set_title(f'Accuracy Distribution - Epoch {epoch+1}')
            ax4.set_xlabel('Accuracy Value')
            ax4.set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.pause(0.1)  # 暂停以更新图表
        
        # 保存最终的训练过程图
        self.save_figure(fig, "training_process")
        plt.show()
        
        print(f"\nFinal Statistics - Total batches: {len(losses)}")
        print(f"Average loss: {np.mean(losses):.4f}")
        print(f"Average accuracy: {np.mean(accuracies):.4f}")
        print(f"Min loss: {min(losses):.4f}")
        print(f"Max accuracy: {max(accuracies):.4f}")


def create_sample_data(batch_size=8, seq_len=128):
    """创建示例数据用于可视化"""
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
    
    return [(bit_seq, meta, target_seq, prefix_lens)]  # 返回单个批次的数据模拟dataloader


def main():
    # 创建配置
    class Config:
        def __init__(self):
            self.d_model = 64
            self.seq_len = 128
            self.vocab_size = 3  # {0, 1, MASK}
            self.num_layers = 3
            self.num_as = 50
            self.num_org = 100
            self.num_cat1 = 10
            self.num_cat2 = 20
            self.num_proto = 5
            self.num_geo = 50
            self.as_embed_dim = 16
            self.org_embed_dim = 16
            self.cat1_embed_dim = 8
            self.cat2_embed_dim = 8
            self.proto_embed_dim = 4
            self.geo_embed_dim = 16
            self.rtt_feat_dim = 5
            self.coarse_end = 32
            self.medium_end = 64
            self.ssm_d_state = 8
            self.ssm_d_conv = 4
    
    cfg = Config()
    
    # 创建模型
    model = HMambaFRPGenerator(cfg)
    
    # 创建可视化器，指定保存目录
    visualizer = HMambaVisualizer(model, save_dir="./chart")
    visualizer.register_hooks()
    
    # 创建示例数据
    sample_data = create_sample_data(batch_size=4, seq_len=128)
    bit_seq, meta, target_seq, prefix_lens = sample_data[0]
    
    print("Sample data shapes:")
    print(f"  Bit sequence: {bit_seq.shape}")
    print(f"  Metadata: {list(meta.keys())}")
    print(f"  Target sequence: {target_seq.shape}")
    print(f"  Prefix lengths: {prefix_lens.shape}")
    
    # 可视化前向传播
    visualizer.visualize_forward_pass(bit_seq, meta)
    
    # 模拟训练过程可视化
    print("\nPreparing training process visualization...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    # 模拟dataloader
    class SimpleDataLoader:
        def __init__(self, data, batch_size=2):
            self.data = data
            self.batch_size = batch_size
        
        def __iter__(self):
            for i in range(0, len(self.data), self.batch_size):
                batch = self.data[i:i+self.batch_size]
                yield batch[0]  # 这里简化，实际应该是多个样本的批次
    
    dataloader = SimpleDataLoader(sample_data * 5, batch_size=2)  # 重复数据以模拟多个批次
    
    visualizer.visualize_training_process(dataloader, criterion, optimizer, epochs=50)  # 修改：从2改为50
    
    # 移除钩子
    visualizer.remove_hooks()


if __name__ == "__main__":
    main()