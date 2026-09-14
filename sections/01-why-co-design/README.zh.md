<!--
LOCALIZED-SOURCE
id: 01-why-co-design
language: zh-CN
-->

# 为什么量化和稀疏 attention 不能直接拼在一起

世界模型对算力的需求主要来自两处：Linear/MLP 的稠密计算，以及视觉、音频和条件 token 带来的长序列 attention。FP8 擅长降低前者的计算量和带宽，Sol-Attn 则通过跳过 attention block 来减少后者的工作量。要进一步提速，自然会想到把两者同时打开。

| 方法 | 主要作用 | 没有解决的问题 |
|---|---|---|
| FP8 Linear | 降低投影层和 MLP 开销 | 长序列 attention |
| FP8 attention | 降低 QK/PV 精度和带宽 | 仍会计算全部 token pair |
| Sol-Attn | 跳过不重要的 attention block | 其余 Transformer 计算 |
| Ulysses SP | 将序列拆到多张 GPU | 跨卡通信 |

真正的冲突在数据布局上。量化会按照 tensor、row、group 或 block 划分数据，并为每个量化单元保存对应的 scale。这个划分通常假设 tensor 的布局是固定的。Sol-Attn、Top-K、Top-P 一类方法却会在运行时选择并淘汰 attention block；保留下来的 K/V 需要 gather、压紧或重新编号。经过这一步，物理 tile 与原来的量化分组不再天然对齐，dense FP8 kernel 也就不能直接拿原有 scale 去计算。

一种简单做法是把选中的 block 反量化回 BF16，再交给稀疏 kernel。但这样会重新引入数据搬运和高精度计算，量化本来想节省的开销又回来了。因此，问题不是“缺少一个开关”，而是稀疏索引、量化 scale 和 kernel tile 必须采用兼容的布局。

序列并行又增加了一层约束。token 被切到不同 rank 后，每张卡看到的是局部 tensor，而稀疏选择仍要保持全局一致的含义。量化元数据、稀疏 block 编号和 rank 内布局必须一起设计，否则单卡能跑通的 kernel 到多卡就会失效。

硬件差异是另一个问题。低精度格式和 kernel 并不覆盖所有 GPU 架构；为更新硬件准备的 MXFP8、NVFP4 实现也不会自动在 SM90 上工作。因此，框架显示“FP8 已开启”，不等于 Linear、稀疏 attention 和多卡分片已经连成一条低精度执行链。

Q-SPA 的出发点就是统一这几份布局约定。归一化、位置编码等敏感计算保留高精度，主要 Linear 和被选中的 attention block 则继续使用与硬件匹配的 FP8 kernel。
