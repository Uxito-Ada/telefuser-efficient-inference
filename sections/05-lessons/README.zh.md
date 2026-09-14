<!--
LOCALIZED-SOURCE
id: 05-lessons
language: zh-CN
-->

# TeleFuser 的统一高效推理路径

Q-SPA 让 TeleFuser 从支持 MiniMax-H3 推理，进一步扩展到优化完整 DiT 路径。当前框架可以组合使用：

- FP8 Linear 与 FP8 Sol 稀疏 attention；
- 质量感知的 FP8 attention；
- 可配置的质量敏感稠密区域；
- Base、Turbo LoRA 与 FastH3 风格 Adapter；
- Ulysses 序列并行、张量并行与通信计算重叠。

这些能力的价值在于可以协同工作：蒸馏 Adapter 减少 DiT 执行次数，FP8 降低稠密 Transformer 成本，Sol-Attn 减少 attention 计算，质量修正保持生成轨迹，Ulysses 再将同一条路径扩展到多张 GPU。

在对齐的四卡 Base H3 工作负载中，两套框架均运行 `TP2 x Ulysses SP2`，TeleFuser 的生成速度达到 LightX2V 的 2.64 倍，代表性峰值显存降低 40.3%。Adapter 评测进一步表明，同一套运行时也优于可正确运行的 LightX2V Turbo 与 FastVideo FastH3 baseline。配套的 tensor profile、视频与同步音频同时覆盖数值误差和最终效果，而不只展示性能。

## 延伸阅读

- [TeleFuser](https://github.com/Tele-AI/TeleFuser)
- [MiniMax-H3 模型与官方 Pipeline](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [Sol-Attn：在线注意力稀疏化](https://nvlabs.github.io/Sana/Sol-Attn/)
- [FastVideo MiniMax-H3 Cookbook](https://haoailab.com/FastVideo/cookbook/minimax-h3/)
- [LightX2V MiniMax-H3 示例](https://github.com/ModelTC/LightX2V/tree/main/scripts/minimax_h3)
- [TorchAO 量化推理工作流](https://docs.pytorch.org/ao/stable/workflows/inference.html)
