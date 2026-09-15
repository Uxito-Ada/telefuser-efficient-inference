<!--
LOCALIZED-SOURCE
id: 01-why-co-design
language: zh-CN
-->

# 量化与稀疏 attention 的布局冲突

**FP8 与稀疏 attention 解决不同的计算瓶颈。** FP8 降低 Linear/MLP 的计算与带宽开销，Sol-Attn 跳过长序列中不重要的 attention block；要进一步加速 DiT，需要让两者在同一执行路径中工作。

**核心冲突是固定的量化布局遇上每次 forward 的动态重排。** Offline、online 或 lazy 量化按 tensor、row、channel、group 或 block 生成低精度数据和 scale，kernel 随后假设布局不变。Sol-Attn、Top-K 和 Top-P 则会根据当前 Q/K/V 选择、淘汰并压紧 block，改变分组边界和 tile 编号，使量化数据、scale 与稀疏索引失去原有对应关系。

**现有量化与稀疏实现无法作为两个独立开关直接叠加。** Per-row、per-channel 和 per-group scale 会直接失配；per-tensor scale 虽然数值仍有效，既有 kernel 的打包和寻址约定也已被破坏。将选中的 block 反量化为 BF16 可以恢复兼容，但会抵消低精度计算的收益。

**低精度 kernel 还必须适配实际部署的 GPU。** 面向新硬件的 MXFP8、NVFP4 实现不会自动覆盖 SM90 等现有平台，TeleFuser 因此为这些 GPU 补齐硬件匹配的低精度执行路径。

**Q-SPA 统一重排后的数据、scale、稀疏索引和 kernel tile。** 主要 Linear 与选中的 attention block 继续使用硬件匹配的 FP8 kernel，归一化和位置编码等敏感计算保留高精度。
