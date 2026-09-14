<!--
LOCALIZED-SOURCE
id: 00-introduction
language: zh-CN
-->

# TeleFuser：高性能、高质量的世界模型推理

*在 H100 上协同优化 FP8、稀疏注意力、Adapter 与多 GPU 执行*

[TeleFuser](https://github.com/Tele-AI/TeleFuser) 是一个面向实时世界模型与多模态生成的开源流式推理和服务框架。它在同一套运行时中提供模型执行、分布式 GPU 推理、有状态服务与流式传输。本文介绍 TeleFuser 推理优化栈的一项重要进展：面向计算密集型 Diffusion Transformer 的高质量 FP8 稀疏推理方案。

我们选择 MiniMax-H3 作为验证模型。它通过一个 DiT 联合生成高分辨率视频和同步音频，对运动稳定性、画面细节与音画事件一致性都有较高要求。同时，它包含大规模稠密计算、长序列注意力、Adapter 和多 GPU 执行，能够充分检验世界模型推理框架中不同优化能否真正协同工作。

TeleFuser 在一条推理路径中集成了：

- FP8 Linear 与面向 H100 的 FP8 Sol-Attn；
- 质量感知的 FP8 attention 与选择性稠密计算；
- Base、Turbo LoRA 与 FastH3 风格 Adapter；
- Ulysses 序列并行、张量并行与通信计算重叠。

在对齐的四卡 Base H3 测试中，两套框架均运行 `TP2 x Ulysses SP2`，TeleFuser 的视频生成速度达到 LightX2V 的 **2.64 倍**，代表性单卡峰值显存降低 **40.3%**。随后，我们分别测试两种 Adapter，并与能够正确运行对应模型的外部框架比较，同时直接展示生成的视频与音频。

全文围绕四个问题展开：

1. 为什么 FP8 与稀疏注意力必须协同设计？
2. TeleFuser 为 H100 增加了哪些关键能力？
3. TeleFuser FP8 如何在保持速度的同时维持生成质量？
4. Adapter 与多 GPU 执行如何进入同一条优化路径？

最后，我们给出对齐的性能、显存、tensor 误差与生成效果测试。
