<!--
LOCALIZED-SOURCE
id: 02-system-overview
language: zh-CN
-->

# TeleFuser Optimization Stack：三层优化

这次 MiniMax-H3 工作从三个层面扩展了 TeleFuser：DiT 稠密计算、长序列 attention 和模型级分布式执行。用户只需选择受支持的推理配置，即可组合使用这些能力。

## 面向 H100 的 FP8 稀疏 attention

TeleFuser 使用 FP8 加速 DiT 中的投影层与 MLP，并通过 SM90 版本的 Sol-Attn 将低精度扩展到 attention。Sol-Attn 在运行时选择重要的 attention 区域，TeleFuser 则让稀疏 attention 继续使用 FP8 QKV 计算，而不是重新回到 BF16 backend。

精度和稀疏性由同一个 attention 实现负责，减少了量化 Transformer 与稀疏 attention 之间的格式转换。对于尚未覆盖的情况，TeleFuser 会使用经过验证的稠密 fallback。

## 支持 Adapter 的低精度推理

MiniMax-H3 既可以运行 Base 模型，也可以叠加加速或风格 Adapter。TeleFuser 支持官方 Turbo LoRA，以及 FastH3 风格的 LoRA 和 Dense Adapter。Adapter 的变化会进入模型最终使用的 FP8 表示，从而保证低精度执行对应用户实际选择的模型。

这项支持带来的不只是兼容性。蒸馏 Adapter 减少 DiT 的执行次数，FP8 与稀疏 attention 则降低每次执行的成本，TeleFuser 可以在同一个请求中叠加两类加速。

## 面向长序列的分布式执行

在更大规模的部署中，Ulysses 序列并行负责拆分长序列 attention，张量并行负责拆分较宽的 Transformer 层。TeleFuser 支持 MiniMax-H3 的双卡和四卡配置，包括 `TP2 x Ulysses SP2`，并支持将 Ulysses 通信与 attention 计算重叠。

由此，减少 DiT 执行次数、FP8 稠密计算、FP8 稀疏 attention 和多 GPU 执行被组合为一套优化方案。由于 FP8 舍入与稀疏计算会共同影响去噪轨迹，生成质量的保持也成为这条路径的一部分。
