<!--
LOCALIZED-SOURCE
id: 01-why-co-design
language: zh-CN
-->

# 量化与稀疏 attention 的布局冲突

世界模型的主要计算开销来自两部分：Linear/MLP 的稠密计算，以及视觉、音频和条件 token 带来的长序列 attention。FP8 可以降低前者的计算与带宽开销，Sol-Attn 则通过跳过 attention block 减少后者的计算量。进一步加速需要同时利用这两类优化。

| 方法 | 主要作用 | 没有解决的问题 |
|---|---|---|
| FP8 Linear | 降低投影层和 MLP 开销 | 长序列 attention |
| FP8 attention | 降低 QK/PV 精度和带宽 | 仍会计算全部 token pair |
| Sol-Attn | 跳过不重要的 attention block | 其余 Transformer 计算 |
| Ulysses SP | 将序列拆到多张 GPU | 跨卡通信 |

两类优化的核心冲突在于数据布局。量化按照 tensor、row、group 或 block 划分数据，并为每个量化单元保存对应的 scale；该映射通常依赖固定的 tensor 布局。Sol-Attn、Top-K、Top-P 等稀疏方法会在运行时选择并淘汰 attention block，保留的 K/V 随后需要 gather、压紧或重新编号。此时，物理 tile 与原量化分组不再对齐，dense FP8 kernel 无法将重排后的数据直接关联到原有 scale。

一种兼容方案是将选中的 block 反量化为 BF16，再交由稀疏 kernel 处理，但额外的数据转换和高精度计算会抵消部分量化收益。要保留端到端低精度执行，稀疏索引、量化 scale 和 kernel tile 必须采用兼容的布局。

序列并行进一步引入跨 rank 的布局约束。token 分片后，每个 rank 仅持有局部 tensor，而稀疏选择仍需保持一致的全局语义。量化元数据、稀疏 block 编号和 rank 内布局如果不能协同映射，单卡 kernel 将无法直接扩展到多卡。

硬件差异是另一项约束。低精度格式和 kernel 并不覆盖所有 GPU 架构；面向更新硬件的 MXFP8、NVFP4 实现也不会自动支持 SM90。因此，框架层面的“FP8 enabled”标记并不能说明 Linear、稀疏 attention 和多卡分片已经构成端到端低精度执行链。

Q-SPA 通过统一量化、稀疏与分布式 attention 的布局约定解决上述问题。归一化、位置编码等敏感计算保留高精度，主要 Linear 和被选中的 attention block 则继续使用与硬件匹配的 FP8 kernel。
