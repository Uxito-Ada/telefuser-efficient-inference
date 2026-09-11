<!--
LOCALIZED-SOURCE
id: 05-lessons
language: zh-CN
-->

# TeleFuser 的统一高效推理路径

这项工作让 TeleFuser 从支持 MiniMax-H3 推理，进一步扩展到在 H100 上优化完整 DiT 路径。当前框架可以组合使用：

- FP8 Linear 与 FP8 Sol 稀疏 attention；
- 质量感知的 FP8 attention；
- 可配置的质量敏感稠密区域；
- Base、Turbo LoRA 与 FastH3 风格 Adapter；
- Ulysses 序列并行、张量并行与通信计算重叠。

这些能力的价值在于可以协同工作：蒸馏 Adapter 减少 DiT 执行次数，FP8 降低稠密 Transformer 成本，Sol-Attn 减少 attention 计算，质量修正保持生成轨迹，Ulysses 再将同一条路径扩展到多张 GPU。

在对齐的 FastH3 工作负载中，相比 FastVideo BF16 + FA4 baseline，TeleFuser 的去噪时间降低 25.7%，DiT 吞吐提升 34.6%，峰值显存降低 14.3%。Tensor profile 和生成视频进一步展示了质量保持策略对数值误差与最终画面的影响。

## 延伸阅读

- [TeleFuser](https://github.com/Tele-AI/TeleFuser)
- [MiniMax-H3 模型与官方 Pipeline](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [Sol-Attn：在线注意力稀疏化](https://nvlabs.github.io/Sana/Sol-Attn/)
- [FastVideo MiniMax-H3 Cookbook](https://haoailab.com/FastVideo/cookbook/minimax-h3/)
- [TorchAO 量化推理工作流](https://docs.pytorch.org/ao/stable/workflows/inference.html)
