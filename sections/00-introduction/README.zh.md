<!--
LOCALIZED-SOURCE
id: 00-introduction
language: zh-CN
-->

# Q-SPA：用 TeleFuser 实现量化、稀疏与并行协同的世界模型推理

[TeleFuser](https://github.com/Tele-AI/TeleFuser) 是一个面向实时世界模型和多模态生成的开源推理与服务框架，支持分布式推理、有状态服务和流式传输。本文介绍 Q-SPA（Quantized Sparse-Parallel Attention）：一套协同设计 FP8 量化、动态稀疏 attention 与序列并行的执行方案。

<div class="hero-result">
  <strong>4 × H100：52.27 秒生成 5 秒、1344 × 768、124 帧视频和同步立体声音频</strong>
  <span>相同四卡拓扑下，完整请求比 LightX2V 快 2.64 倍，比 SGLang 快 1.52 倍。</span>
</div>

<img class="results-montage" src="assets/results-montage.webp" width="960" height="549" alt="Base、Turbo、FastH3 与 FP8 质量实验输出总览">

本文以 MiniMax-H3 为主要测试模型。它的 DiT 联合生成高分辨率视频和音频，计算同时集中在大规模 Linear/MLP 和长序列 attention。性能优化必须和运动稳定性、画面细节及音频完整性一起验证。

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

TeleFuser 的 Base H3 请求从 1 卡扩展到 2 卡和 4 卡，四卡去噪吞吐达到单卡的 **3.40 倍**。评测还包括 SGLang、LightX2V、FastVideo 以及 Turbo LoRA、FastH3 Adapter，所有视频均可在文中直接播放。

后续章节依次讨论四个问题：

1. 为什么现有量化 kernel 很难直接接上动态稀疏 attention？
2. Q-SPA 如何统一量化 scale、稀疏 block 和跨卡分片的数据布局？
3. FP8 attention 如何控制生成质量损失？
4. Q-SPA 如何扩展到多卡？
