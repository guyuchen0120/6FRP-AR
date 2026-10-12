# demonstrate_hmamba_flow.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 导入HMamba相关组件
from hmamba import HMambaFRPGenerator, HierarchicalMambaLayer, MambaBlock, selective_scan

def demonstrate_data_flow():
    print("=" * 60)
    print("HMamba 数据流演示")
    print("=" * 60)
    
    # 创建一个简单的配置
    class Config:
        def __init__(self):
            self.d_model = 8    # 为了演示，使用较小的维度
            self.seq_len = 16   # 为了演示，使用较短的序列
            self.vocab_size = 3 # {0, 1, MASK}
            self.num_layers = 1
            self.num_as = 10
            self.num_org = 20
            self.num_cat1 = 5
            self.num_cat2 = 8
            self.num_proto = 3
            self.num_geo = 15
            self.as_embed_dim = 4
            self.org_embed_dim = 4
            self.cat1_embed_dim = 2
            self.cat2_embed_dim = 2
            self.proto_embed_dim = 2
            self.geo_embed_dim = 4
            self.rtt_feat_dim = 3
            self.coarse_end = 8
            self.medium_end = 12
            self.ssm_d_state = 4
            self.ssm_d_conv = 3

    cfg = Config()
    
    print("1. 创建模型...")
    model = HMambaFRPGenerator(cfg)
    
    print("\n2. 创建示例数据...")
    # 批次大小为4，序列长度为16，模型维度为8
    B, L, D = 4, 16, 8
    batch_size = B
    seq_len = L
    
    # 创建输入数据
    bit_seq = torch.randint(0, 2, (batch_size, seq_len))  # 0或1的比特序列
    print(f"   输入比特序列: {bit_seq.shape} -> {bit_seq[0, :8].tolist()}")
    
    # 创建元数据
    meta = {
        'as_id': torch.tensor([0, 0, 1, 1]),  # 前两个样本属于AS 0，后两个属于AS 1
        'org_id': torch.randint(0, 5, (batch_size,)),
        'cat1_id': torch.randint(0, 3, (batch_size,)),
        'cat2_id': torch.randint(0, 4, (batch_size,)),
        'proto_id': torch.randint(0, 2, (batch_size,)),
        'geo_id': torch.randint(0, 5, (batch_size,)),
        'rtt_stats': torch.randn(batch_size, 3)
    }
    print(f"   AS ID: {meta['as_id'].tolist()} (前两个同AS，后两个同AS)")
    
    print("\n3. 前向传播过程演示...")
    
    # 3.1 嵌入步骤
    print("   3.1 嵌入层处理:")
    pos = torch.arange(cfg.seq_len)  # [0, 1, 2, ..., 15]
    print(f"   位置索引: {pos[:8].tolist()}")
    
    # 比特嵌入
    bit_embedded = model.bit_embed(bit_seq)  # (B, L, D)
    print(f"   比特嵌入: {bit_seq.shape} -> {bit_embedded.shape}")
    
    # 位置嵌入
    pos_embedded = model.pos_embed(pos).unsqueeze(0)  # (1, L, D) -> 广播到(B, L, D)
    print(f"   位置嵌入: {pos.shape} -> {pos_embedded.shape}")
    
    # 层级嵌入（前8位粗粒度，8-12中粒度，12-16细粒度）
    layer_embedded = model.layer_embed(model.layer_of_bit).unsqueeze(0)  # (1, L, D)
    print(f"   层级嵌入: {model.layer_of_bit.shape} -> {layer_embedded.shape}")
    
    # 组合嵌入
    x = bit_embedded + pos_embedded + layer_embedded
    print(f"   组合嵌入: 每个位置{x.shape}，前几个值: {x[0, 0, :4].tolist()}")
    
    # 3.2 条件编码
    print("   3.2 条件编码:")
    cond = model.condition_encoder(meta)  # (B, D)
    print(f"   条件编码: 各元数据 -> {cond.shape}")
    print(f"   样本0条件: {cond[0, :4].tolist()}")
    print(f"   样本1条件: {cond[1, :4].tolist()}")  # 应该与样本0相似（同AS）
    print(f"   样本2条件: {cond[2, :4].tolist()}")
    print(f"   样本3条件: {cond[3, :4].tolist()}")  # 应该与样本2相似（同AS）
    
    # 添加条件信息到主特征
    x = x + cond.unsqueeze(1)  # (B, L, D) + (B, 1, D) -> (B, L, D)
    print(f"   添加条件后: {x.shape}")
    
    # 3.3 层次化Mamba层处理
    print("   3.3 层次化Mamba层处理:")
    layer = model.layers[0]  # 取第一个层
    
    print(f"   输入到层: {x.shape}")
    print(f"   AS IDs: {meta['as_id'].tolist()}")
    
    # 外层扫描演示
    print("   3.3.1 外层扫描 (每个样本独立处理其16个位置):")
    print(f"     输入: {x.shape} (4个样本，每个16个位置，8维特征)")
    out_outer = layer.outer(x)  # (B, L, D)
    print(f"     输出: {out_outer.shape}")
    print(f"     样本0位置0输出: {out_outer[0, 0, :4].tolist()}")
    print(f"     样本1位置0输出: {out_outer[1, 0, :4].tolist()}")
    print(f"     样本2位置0输出: {out_outer[2, 0, :4].tolist()}")
    print(f"     样本3位置0输出: {out_outer[3, 0, :4].tolist()}")
    print("     注意: 每个样本的位置0输出可能不同，因为各自独立处理")
    
    # 内层扫描演示
    print("   3.3.2 内层扫描 (相同AS的样本间交互):")
    as_ids = meta['as_id']
    print(f"     AS分组: AS 0 -> 样本 [0, 1], AS 1 -> 样本 [2, 3]")
    
    # 手动演示AS 0的内层扫描过程
    print("     演示AS 0的内层扫描:")
    idx_as0 = (as_ids == 0).nonzero(as_tuple=True)[0]  # [0, 1]
    print(f"     AS 0的样本索引: {idx_as0.tolist()}")
    
    grp_as0 = x[idx_as0]  # (2, 16, 8) - 2个样本，16个位置，8维特征
    print(f"     AS 0样本组: {grp_as0.shape}")
    
    # 调换维度以便内层Mamba处理 (G, L, D) -> (L, G, D)
    grp_as0_t = grp_as0.transpose(0, 1).contiguous()  # (16, 2, 8)
    print(f"     调换维度后: {grp_as0_t.shape}")
    print(f"     位置0的2个样本特征: {grp_as0_t[0, :, :4].tolist()}")
    print(f"     位置1的2个样本特征: {grp_as0_t[1, :, :4].tolist()}")
    
    # 内层Mamba处理 - 沿着样本维度扫描
    scanned_as0 = layer.inner(grp_as0_t)  # (16, 2, 8)
    print(f"     内层扫描输出: {scanned_as0.shape}")
    print(f"     位置0扫描后: {scanned_as0[0, :, :4].tolist()}")
    print(f"     注意: 两个样本在这个位置的输出可能相互影响!")
    
    # 转换回原始形状
    scanned_as0 = scanned_as0.transpose(0, 1).contiguous()  # (2, 16, 8)
    print(f"     转换回原形状: {scanned_as0.shape}")
    
    # 完整的内层扫描
    out_inner = layer._scan_within_as(x, as_ids)
    print(f"     完整内层扫描输出: {out_inner.shape}")
    
    # 3.4 门控融合
    print("   3.4 门控融合:")
    g = F.softmax(layer.gate(cond), dim=-1)  # (B, 2)
    print(f"     门控权重: {g.shape}")
    print(f"     样本0权重: 外层={g[0, 0]:.3f}, 内层={g[0, 1]:.3f}")
    print(f"     样本1权重: 外层={g[1, 0]:.3f}, 内层={g[1, 1]:.3f}")
    print(f"     样本2权重: 外层={g[2, 0]:.3f}, 内层={g[2, 1]:.3f}")
    print(f"     样本3权重: 外层={g[3, 0]:.3f}, 内层={g[3, 1]:.3f}")
    
    # 加权融合
    g_outer = g[:, 0:1].unsqueeze(-1)  # (B, 1, 1)
    g_inner = g[:, 1:2].unsqueeze(-1)  # (B, 1, 1)
    fused = g_outer * out_outer + g_inner * out_inner
    print(f"     融合后: {fused.shape}")
    
    # 最终输出
    final_output = layer.fuse_norm(fused + x)  # 残差连接
    print(f"     层输出: {final_output.shape}")
    
    print("\n4. 完整前向传播:")
    with torch.no_grad():
        logits = model.forward(bit_seq, meta)
    print(f"   最终logits: {logits.shape}")
    print(f"   样本0的预测: 前5个位置 -> {F.softmax(logits[0, :5], dim=-1).tolist()}")
    
    print("\n关键洞察:")
    print("• 外层扫描: 每个样本独立处理其序列，捕捉序列内部模式")
    print("• 内层扫描: 相同AS的样本相互影响，捕捉同类IP的共同特征") 
    print("• 门控融合: 动态决定相信全局模式还是局部模式")
    print("• 虽然outer和inner使用相同架构，但处理的数据方向完全不同")

def demonstrate_selective_scan():
    print("\n" + "=" * 60)
    print("Selective Scan 机制演示")
    print("=" * 60)
    
    # 创建一个小的示例 - 使用明确的Python整数
    B = 2  # 批次大小
    L = 4  # 序列长度
    D = 3  # 特征维度
    N = 2  # 状态维度
    
    # 确保所有变量都是Python整数
    B, L, D, N = int(B), int(L), int(D), int(N)
    
    # 输入数据 - 使用Python整数
    u = torch.randn((B, L, D))  # 输入
    delta = torch.rand((B, L, D)) * 0.1 + 0.01  # 时间步长
    A = torch.randn((D, N)) * 0.1  # 状态矩阵
    B_tensor = torch.randn((B, L, N))  # 输入投影
    C = torch.randn((B, L, N))  # 输出投影
    
    print(f"输入 u: {u.shape}")
    print(f"delta: {delta.shape}")
    print(f"A: {A.shape}")
    print(f"B: {B_tensor.shape}")
    print(f"C: {C.shape}")
    
    # 手动执行selective_scan
    B_size, L_size, D_size = u.shape  # 从张量中获取维度
    N_size = A.shape[-1]
    
    print(f"\n手动执行selective_scan过程:")
    print(f"序列长度 L = {L_size}")
    
    # 离散化
    dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, D, N)
    dB = delta.unsqueeze(-1) * B_tensor.unsqueeze(2)  # (B, L, D, N)
    
    print(f"dA: {dA.shape}, dB: {dB.shape}")
    
    h = torch.zeros((B_size, D_size, N_size), device=u.device, dtype=u.dtype)
    print(f"初始状态 h: {h.shape}")
    
    ys = []
    for t in range(L_size):
        print(f"\n时间步 {t}:")
        print(f"  输入 u[{t}]: {u[:, t, :].tolist()}")
        print(f"  状态 h 之前: {h[0, 0, :].tolist()}")
        
        # 更新状态
        h = dA[:, t] * h + dB[:, t] * u[:, t].unsqueeze(-1)
        print(f"  状态 h 之后: {h[0, 0, :].tolist()}")
        
        # 计算输出
        y = (h * C[:, t].unsqueeze(1)).sum(-1)
        print(f"  输出 y[{t}]: {y[0, :].tolist()}")
        
        ys.append(y)
    
    manual_result = torch.stack(ys, dim=1)
    
    # 与函数结果对比
    function_result = selective_scan(u, delta, A, B_tensor, C)
    
    print(f"\n手动计算结果: {manual_result.shape}")
    print(f"函数计算结果: {function_result.shape}")
    print(f"结果是否相等: {torch.allclose(manual_result, function_result, atol=1e-6)}")
    
    if not torch.allclose(manual_result, function_result, atol=1e-6):
        print(f"最大差异: {torch.max(torch.abs(manual_result - function_result)).item()}")


if __name__ == "__main__":
    demonstrate_data_flow()
    demonstrate_selective_scan()