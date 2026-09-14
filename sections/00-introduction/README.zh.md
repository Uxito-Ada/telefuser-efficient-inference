<!--
LOCALIZED-SOURCE
id: 00-introduction
language: zh-CN
-->

# Q-SPA：用 TeleFuser 实现量化、稀疏与并行协同的世界模型推理

[TeleFuser](https://github.com/Tele-AI/TeleFuser) 是一个面向实时世界模型和多模态生成的开源推理、服务框架。它已经具备分布式推理、有状态服务和流式传输能力。这篇文章介绍我们最近补上的一块：Q-SPA（Quantized Sparse-Parallel Attention），即让 FP8 量化、动态稀疏 attention 和序列并行在同一套执行逻辑下工作。

我们用 MiniMax-H3 来检验这套方案。H3 的 DiT 同时生成高分辨率视频和音频，既有大规模 Linear/MLP 计算，也有长序列 attention。速度不是唯一指标：运动是否稳定、画面细节是否保留、声音是否正常，都必须一起检查。

Q-SPA 主要包含三部分：

- 用 FP8 处理 Linear 和 attention 中适合低精度执行的计算；
- 用 Sol-Attn 跳过不重要的 attention block，同时保留必要的稠密区域；
- 用 Ulysses SP、Tensor Parallel 和通信计算重叠扩展到多卡。

我们还用 Turbo LoRA 和 FastH3 Adapter 两种模型变体，检查这套量化、稀疏和并行实现能否覆盖实际使用场景。

在四卡 Base H3 对比中，LightX2V 和 TeleFuser 都使用 `TP2 x Ulysses SP2`。TeleFuser 的生成速度是 LightX2V 的 **2.64 倍**，代表性单卡峰值显存低 **40.3%**。后文还会给出 Turbo LoRA、FastH3 Adapter 的性能结果和完整生成视频。

下面依次讨论四个问题：

1. 为什么现有量化 kernel 很难直接接上动态稀疏 attention？
2. Q-SPA 如何统一量化 scale、稀疏 block 和跨卡分片的数据布局？
3. FP8 attention 如何控制生成质量损失？
4. 这套 attention 实现如何扩展到多卡？
