<!--
LOCALIZED-SOURCE
id: 03-quality-and-scale
language: zh-CN
-->

# 质量保持与多 GPU 扩展

Diffusion 模型会把当前预测继续用于下一次去噪更新，因此 FP8 和稀疏 attention 不仅要减少计算，还需要保持稳定的生成轨迹。TeleFuser 将质量保持直接集成到 FP8 路径与稀疏策略中。

## 质量感知的 FP8 attention

对 MiniMax-H3 真实层的 profiling 显示，K/V 存在明显的非零均值分布，这会占用对称 FP8 的有效动态范围。TeleFuser 将它作为 attention-specific asymmetric quantization 问题处理：在 FP8 计算前对 K/V 做中心化，并在 attention 输出中补偿等价的偏移。

中心化和修正操作被融合到已有 FP8 Sol-Attn 路径中，不需要单独的质量模式。最初未融合的实现带来 11.7% 去噪开销，融合后降至 2.2%。在捕获的 MiniMax-H3 真实层上，K 量化 MSE 降低 21.65%，attention 输出 MSE 降低 8.18%。优化配置默认启用 KV smoothing 和 V correction。

在 50-step 质量实验中，这条 FP8 路径的去噪吞吐仍比 BF16 Linear + FlashAttention 4 高 39.4%，peak allocated memory 降低 42.6%。相比未使用 smoothing 的 FP8，frame cosine、PSNR 和 mean SSIM 均有提升；音频距离指标有升有降。下面可以同步播放三段完整视频与音频进行比较。

<div class="video-grid video-grid-three" data-sync-group="smoothing">
  <figure>
    <figcaption>BF16 Linear + FlashAttention 4</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="bf16-quality"></video>
  </figure>
  <figure>
    <figcaption>FP8 Linear + FP8 Sol，未使用 smoothing</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="fp8-unsmoothed"></video>
  </figure>
  <figure>
    <figcaption>FP8 Linear + FP8 Sol，质量感知 FP8</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="fp8-smoothed"></video>
  </figure>
</div>

## 质量感知的稀疏策略

不同去噪阶段和 Transformer 层对长程依赖的敏感程度不同。TeleFuser 支持在开头若干次更新和指定层中保留稠密 attention，再由 Sol-Attn 完成其余计算。本文测试的 FastH3 配置使用两个 dense opening update 和两个 dense layer，之后采用 `tau=1.0` 的 exact sparse routing。Dense step、dense layer、threshold mode 和 `tau` 均可针对其他 MiniMax-H3 schedule 调整。

## 同一条路径扩展到多 GPU

TeleFuser 将 Ulysses 序列并行与张量并行组合，用于长序列和宽 Transformer 的分布式执行。质量统计基于 Ulysses 建立的 attention 视图计算，从而保持单卡与多卡语义一致。运行时还支持重叠 Ulysses 通信与 attention 计算；当 FP8 和稀疏计算减少算术时间后，这项优化会更加重要。
