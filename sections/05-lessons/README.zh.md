<!--
LOCALIZED-SOURCE
id: 05-lessons
language: zh-CN
-->

# 结语

Q-SPA 在 TeleFuser 中解决了三个直接相关的问题：

- FP8 scale 与动态稀疏 block 的布局对齐；
- FP8 和稀疏共同作用时的生成误差控制；
- 稀疏 FP8 attention 在 Ulysses SP 与 Tensor Parallel 下的多卡执行。

TeleFuser 支持直接运行 Base H3，也支持在合并 Turbo LoRA 或 FastH3 Adapter 后生成相应的 FP8 权重。Adapter 可以减少去噪步数或提高特定任务的输出质量；Q-SPA 将合并后的有效权重纳入低精度、稀疏和并行优化路径，降低实际 Adapter 模型的端到端推理成本。

四卡 Base H3 测试中，TeleFuser 完整生成耗时 52.27 秒，相比 LightX2V、FastVideo 和 SGLang 分别快 2.64 倍、2.19 倍和 1.52 倍。Turbo LoRA 相比 LightX2V 快 1.59 倍，与 SGLang 的差异为 0.3%；FastH3 相比 FastVideo 快 3.09 倍。

这些实现与实验带来了三点面向 world-model 推理实践的 Insights：

- FP8 与稀疏 attention 应当作为一条执行路径共同设计。单独使用任一优化
  仍会留下大量 DiT 计算；让稀疏 kernel 直接消费量化表示，才能同时获得两者的收益。
- 通用量化封装不能保证 world model 的生成质量。长序列 attention 的布局和
  激活统计具有模型与硬件相关性，需要重新设计面向质量的量化与 kernel 实现。
- World model 的单次请求同时承担条件理解与推理、长序列视频/音频联合去噪和解码，算力压力不能用权重能否装入单卡来衡量。Ulysses SP、Tensor Parallel 与通信计算重叠共同缩短完整生成链路；MiniMax-H3 的去噪时间从单卡的 167.41 秒降至两卡的 90.90 秒和四卡的 49.28 秒，四卡吞吐达到单卡的 3.40 倍。

## 延伸阅读

- [TeleFuser](https://github.com/Tele-AI/TeleFuser)
- [MiniMax-H3 模型与官方 Pipeline](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [Sol-Attn：在线注意力稀疏化](https://nvlabs.github.io/Sana/Sol-Attn/)
- [FastVideo MiniMax-H3 Cookbook](https://haoailab.com/FastVideo/cookbook/minimax-h3/)
- [LightX2V MiniMax-H3 示例](https://github.com/ModelTC/LightX2V/tree/main/scripts/minimax_h3)
- [SGLang MiniMax-H3 Cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx)
- [TorchAO 量化推理工作流](https://docs.pytorch.org/ao/stable/workflows/inference.html)
