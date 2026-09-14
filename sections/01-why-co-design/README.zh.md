<!--
LOCALIZED-SOURCE
id: 01-why-co-design
language: zh-CN
-->

# 为什么 FP8 与稀疏注意力需要协同设计

世界模型通过更高的计算成本换取输出质量。在 MiniMax-H3 中，大规模投影层与 MLP 让 DiT 具有很高的稠密计算量；视觉、音频与条件 token 又带来了昂贵的长序列注意力。单一优化无法同时解决两类开销。

| 技术 | 主要收益 | 尚未解决的开销 |
|---|---|---|
| FP8 Linear | 降低投影层与 MLP 开销 | 长序列注意力 |
| FP8 attention | 降低 QK/PV 精度与带宽 | 稠密 token pair |
| Sol-Attn | 减少参与计算的 attention block | 稠密 Transformer 层 |
| Ulysses SP | 降低单卡序列状态与计算 | GPU 间通信 |

直接把这些功能全部打开并不能得到最优方案。通用量化库通常能够加速 Linear，却不一定包含目标 GPU 所需的稀疏 attention kernel；一些稀疏实现仍要求 BF16 QKV，格式转换会抵消部分收益。序列并行改变 attention 数据在不同 GPU 上的分布方式，而 Adapter 又改变了低精度推理所表示的实际模型权重。

硬件差异进一步放大了这个问题。低精度格式与 kernel 并不能统一覆盖所有 GPU 架构；面向更新硬件优化的 MXFP8 或 NVFP4 方案也不会自动提供等价的 SM90 实现。因此，框架层面标记“已开启 FP8”，并不代表稠密 DiT 计算和稀疏 attention 都能保持低精度执行。

TeleFuser 将这种组合视为一个完整的系统能力。对数值敏感的归一化和位置变换保持较高精度，主要 Linear 计算与 Sol attention 则使用与硬件匹配的 FP8 路径。稀疏路由、QKV 精度、质量修正、Adapter 加载和分布式执行遵循同一份模型契约，使不同优化可以叠加，而不是相互抵消。
