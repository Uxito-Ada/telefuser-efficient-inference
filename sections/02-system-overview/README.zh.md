<!--
LOCALIZED-SOURCE
id: 02-system-overview
language: zh-CN
-->

# Q-SPA 在 TeleFuser 中的实现 {#q-spa-system}

在 MiniMax-H3 上，Q-SPA 覆盖三层执行逻辑：DiT 稠密计算、长序列 attention 和多卡 tensor 布局。

## 让 FP8 和稀疏 block 使用同一份布局

TeleFuser 将 FP8 应用于 DiT 的投影层和 MLP，并将低精度执行延伸到 Sol-Attn。稀疏 block 索引、量化 scale 和 attention kernel 使用的 tile 遵循统一的布局约定，使选中的 QKV block 能够直接参与 FP8 计算，无需预先转换为 BF16。

当前 kernel 尚未覆盖的 shape 将使用经过验证的 dense fallback。

## 分布式 attention

Ulysses 序列并行负责拆分长序列，Tensor Parallel 负责拆分较宽的 Transformer 层。TeleFuser 支持 MiniMax-H3 双卡和四卡配置，包括 `TP2 x Ulysses SP2`；Ulysses 的通信还能与 attention 计算重叠。

稀疏选择所需的统计量基于 Ulysses 建立的 attention 视图计算，从而使单卡与多卡保持一致的 block 语义。FP8 降低算术计算时间后，通信占比相应提高，通信计算重叠的作用也更为显著。

## 覆盖不同 H3 模型变体

MiniMax-H3 除了 Base 模型，还有 Turbo LoRA 和 FastH3 一类加速 Adapter。TeleFuser 会先把 Adapter 合并到有效权重，再建立可复用的 FP8 表示，避免量化的仍是原始 Base 权重。

蒸馏 Adapter 可以减少 DiT 调用次数，Q-SPA 则降低每次调用的成本。TeleFuser 可以在一次请求中同时使用两者。
