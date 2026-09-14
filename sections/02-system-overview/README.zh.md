<!--
LOCALIZED-SOURCE
id: 02-system-overview
language: zh-CN
-->

# Q-SPA 在 TeleFuser 中的实现 {#q-spa-system}

在 MiniMax-H3 上，Q-SPA 改动了三层执行逻辑：DiT 的稠密计算、长序列 attention，以及多卡上的 tensor 布局。

## 让 FP8 和稀疏 block 使用同一份布局

TeleFuser 先把 FP8 用到 DiT 的投影层和 MLP，再把低精度执行延伸到 Sol-Attn。这里的关键不是同时打开两个配置项，而是让稀疏 block 索引、量化 scale 和 attention kernel 消费的 tile 遵守同一份布局约定。这样，选中的 QKV block 可以继续以 FP8 参与计算，不需要先还原成 BF16。

没有被当前 kernel 覆盖的 shape 会走经过验证的 dense fallback，不会静默进入错误路径。

## 把 attention 拆到多张 GPU

Ulysses 序列并行负责拆分长序列，Tensor Parallel 负责拆分较宽的 Transformer 层。TeleFuser 支持 MiniMax-H3 双卡和四卡配置，包括 `TP2 x Ulysses SP2`；Ulysses 的通信还能与 attention 计算重叠。

稀疏选择所需的统计量按照 Ulysses 建立的 attention 视图计算，因此单卡和多卡使用相同的 block 语义。FP8 减少算术时间后，通信占比会变高，通信计算重叠也就更重要。

## 覆盖不同 H3 模型变体

MiniMax-H3 除了 Base 模型，还有 Turbo LoRA 和 FastH3 一类加速 Adapter。TeleFuser 会先把 Adapter 合并到有效权重，再建立可复用的 FP8 表示，避免量化的仍是原始 Base 权重。

蒸馏 Adapter 可以减少 DiT 调用次数，Q-SPA 则降低每次调用的成本。TeleFuser 可以在一次请求中同时使用两者。
