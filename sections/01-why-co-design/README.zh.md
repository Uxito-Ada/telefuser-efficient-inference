<!--
LOCALIZED-SOURCE
id: 01-why-co-design
language: zh-CN
-->

# 量化与稀疏 attention 的布局冲突

世界模型的主要计算开销来自两部分：Linear/MLP 的稠密计算，以及视觉、音频和条件 token 带来的长序列 attention。FP8 降低投影层和 MLP 的计算与带宽开销，Sol-Attn 则跳过不重要的 attention block。两者分别覆盖 DiT 的主要稠密计算和长序列 attention，进一步加速需要让它们在同一执行路径中工作。

两类优化的核心冲突在于数据布局。量化按照 tensor、row、group 或 block 划分数据，并为每个量化单元保存对应的 scale；该映射通常依赖固定的 tensor 布局。Sol-Attn、Top-K、Top-P 等稀疏方法会在运行时选择并淘汰 attention block，保留的 K/V 随后需要 gather、压紧或重新编号。此时，物理 tile 与原量化分组不再对齐，dense FP8 kernel 无法将重排后的数据直接关联到原有 scale。

现有量化路径通常把量化视为一次独立的数据转换：offline 量化在模型加载前完成，online 或 lazy 量化则在 tensor 首次使用或进入算子时生成低精度数据和 scale；转换完成后，kernel 默认数据布局保持不变。稀疏 attention 不同，它会在每次 forward 中根据当前 Q/K/V 重新选择、淘汰和压紧 block。持续发生的布局重排会改变 row、channel 和 group 的边界，使预先组织的量化数据与 scale 失去对应关系。即使 per-tensor 只有一个全局 scale，数值上仍然有效，既有 kernel 的打包顺序和寻址约定也已被打破。因此，量化与动态稀疏不能作为两个独立开关直接叠加；它们需要共同维护重排后的数据、scale、索引和 tile 映射。

一种兼容方案是将选中的 block 反量化为 BF16，再交由稀疏 kernel 处理，但额外的数据转换和高精度计算会抵消部分量化收益。要保留端到端低精度执行，稀疏索引、量化 scale 和 kernel tile 必须采用兼容的布局。

硬件差异是另一项约束。低精度格式和 kernel 并不覆盖所有 GPU 架构。例如，面向更新硬件的 MXFP8、NVFP4 实现不会自动支持 SM90 等仍被大量使用的机器平台。新设备的发布不会使既有算力随之消失；TeleFuser 针对这些平台补齐低精度执行路径，让它们也能高效运行最新的 world model。

Q-SPA 让稀疏索引、量化 scale 与 kernel tile 采用一致的布局约定。归一化、位置编码等敏感计算保留高精度，主要 Linear 和被选中的 attention block 则继续使用与硬件匹配的 FP8 kernel。
