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

TeleFuser 支持直接运行 Base H3，也支持在合并 Turbo LoRA 或 FastH3 Adapter 后生成相应的 FP8 权重。Adapter 决定模型与采样方式，Q-SPA 降低单次 DiT 执行成本。

四卡 Base H3 测试中，LightX2V 和 TeleFuser 均使用 `TP2 x Ulysses SP2`。TeleFuser 的生成速度达到 LightX2V 的 2.64 倍，代表性峰值显存降低 40.3%。Turbo LoRA 与 FastH3 测试也分别优于对应的 LightX2V 和 FastVideo 对照。评测同时提供性能图、tensor 误差、完整视频和同步音频，用于综合检查性能与输出质量。

## 延伸阅读

- [TeleFuser](https://github.com/Tele-AI/TeleFuser)
- [MiniMax-H3 模型与官方 Pipeline](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [Sol-Attn：在线注意力稀疏化](https://nvlabs.github.io/Sana/Sol-Attn/)
- [FastVideo MiniMax-H3 Cookbook](https://haoailab.com/FastVideo/cookbook/minimax-h3/)
- [LightX2V MiniMax-H3 示例](https://github.com/ModelTC/LightX2V/tree/main/scripts/minimax_h3)
- [TorchAO 量化推理工作流](https://docs.pytorch.org/ao/stable/workflows/inference.html)
