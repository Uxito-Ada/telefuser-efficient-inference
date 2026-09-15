<!--
LOCALIZED-SOURCE
id: 02-system-overview
language: zh-CN
-->

# Q-SPA 在 TeleFuser 中的实现 {#q-spa-system}

在 MiniMax-H3 上，Q-SPA 覆盖三层执行逻辑：DiT 稠密计算、长序列 attention 和多卡 tensor 布局。

## 让 FP8 和稀疏 block 使用同一份布局

TeleFuser 将 FP8 应用于 DiT 的投影层和 MLP，并将低精度执行延伸到 Sol-Attn。稀疏 block 索引、量化 scale 和 attention kernel 使用的 tile 遵循统一的布局约定，使选中的 QKV block 能够直接参与 FP8 计算，无需预先转换为 BF16。

## 分布式 attention

TeleFuser 不只是把 Ulysses SP 接入 MiniMax-H3，还针对 FP8、Sol-Attn 和视频 token 布局重构了多卡执行路径。Ulysses All-to-All 建立各 rank 的局部 attention 视图后，再完成 FP8 quantization 和 Sol-Attn routing，使量化 scale、稀疏 block 与实际计算布局保持一致。3D 视频 token 的 reorder 被移到序列切分之前；全局共享的 scalar timestep 在各 rank 复制，per-token timestep 则随 token 一起切分，从而保持模型语义和单卡结果一致。

在此基础上，TeleFuser 为 sequence parallel 补充了专用 kernel，并将 Ulysses 通信与 attention 计算重叠。该路径支持双卡和四卡 MiniMax-H3，包括 `TP2 × Ulysses SP2`，使 FP8 与稀疏带来的算术加速能够继续转化为多卡端到端收益。

## 覆盖不同 H3 模型变体

MiniMax-H3 除了 Base 模型，还有 Turbo LoRA 和 FastH3 一类加速 Adapter。TeleFuser 会先把 Adapter 合并到有效权重，再建立可复用的 FP8 表示，避免量化的仍是原始 Base 权重。

Turbo LoRA 等蒸馏 Adapter 可以减少去噪步数，其他 Adapter 则用于提高特定任务的输出质量。Q-SPA 会把 Adapter 合并后的有效权重纳入 FP8、稀疏和并行执行路径，从而降低这些模型的端到端推理成本，而不只优化 Base checkpoint。
