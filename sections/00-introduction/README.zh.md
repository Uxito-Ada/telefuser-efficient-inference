<!--
LOCALIZED-SOURCE
id: 00-introduction
language: zh-CN
-->

# Q-SPA：用 TeleFuser 实现量化、稀疏与并行协同的世界模型推理

[TeleFuser](https://github.com/Tele-AI/TeleFuser) 是一个面向实时世界模型和多模态生成的开源推理与服务框架，支持分布式推理、有状态服务和流式传输。本文介绍 Q-SPA（Quantized Sparse-Parallel Attention）：一套协同设计 FP8 量化、动态稀疏 attention 与序列并行的执行方案。

本文选择 MiniMax-H3 进行验证。H3 的 DiT 同时生成高分辨率视频和音频，既包含大规模 Linear/MLP 计算，也包含长序列 attention。因此，评测不仅关注性能，还覆盖运动稳定性、画面细节和音频完整性。

Q-SPA 包含三个相互关联的执行维度：

- 面向 Linear 与 attention 的 FP8 低精度计算；
- 基于 Sol-Attn 的动态 block 稀疏，以及质量敏感区域的稠密计算；
- Ulysses SP、Tensor Parallel 与通信计算重叠。

## Insights

- 量化与稀疏可以同时使用。稀疏 attention kernel 直接消费 FP8 表示时，两项
  优化能够共同贡献加速收益。
- 通用量化方案在 world model 的 DiT 上容易出现质量波动。attention 布局、
  激活范围和硬件相关 kernel 需要有针对性的重构，而不是套用通用量化封装。
- 模型能够放入单卡，不代表分布式执行没有意义。DiT 去噪受计算吞吐限制，
  多卡可以分摊计算，并通过通信计算重叠进一步降低延迟。

Turbo LoRA 和 FastH3 Adapter 作为模型变体，用于验证量化、稀疏和并行实现对不同 H3 推理配置的兼容性。

在四卡 Base H3 对比中，LightX2V 和 TeleFuser 均使用 `TP2 x Ulysses SP2`。TeleFuser 的生成速度达到 LightX2V 的 **2.64 倍**，代表性单卡峰值显存降低 **40.3%**。Turbo LoRA 和 FastH3 Adapter 的性能结果与完整生成视频将在评测章节中分别展示。

在 TeleFuser 内部，相同的 Base H3 FP8+Sol 请求从单卡扩展到四卡后，去噪吞吐
提升 **3.40 倍**，去噪时间降低 **70.6%**。

后续章节依次讨论四个问题：

1. 为什么现有量化 kernel 很难直接接上动态稀疏 attention？
2. Q-SPA 如何统一量化 scale、稀疏 block 和跨卡分片的数据布局？
3. FP8 attention 如何控制生成质量损失？
4. Q-SPA 如何扩展到多卡？
