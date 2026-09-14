<!--
LOCALIZED-SOURCE
id: 03-quality-and-scale
language: zh-CN
-->

# 控制误差，并扩展到多卡

Diffusion 的每一步输出都会成为下一步输入。FP8 或稀疏 attention 引入的局部误差可能沿去噪过程累积，因此评测同时覆盖 kernel 吞吐、tensor 误差、最终视频和音频。

## FP8 attention 的误差控制

MiniMax-H3 实际层的 profile 显示，部分 K/V 的均值明显偏离零点，直接使用对称 FP8 量化会损失有效动态范围。TeleFuser 在计算前对 K/V 进行中心化，并在 attention 输出中恢复等价偏移。这是一种针对 attention 数据分布设计的非对称量化处理。

中心化和输出修正已经融合进 FP8 Sol-Attn。最早的非融合版本让去噪时间增加 11.7%，融合后开销降到 2.2%。在捕获的真实 H3 层上，K 的量化 MSE 降低 21.65%，attention 输出 MSE 降低 8.18%；KV smoothing 和 V correction 在优化配置中默认开启。

在 50-step 测试中，该版本的去噪吞吐比 BF16 Linear + FlashAttention 4 提高 39.4%，peak allocated memory 降低 42.6%。相较未使用 smoothing 的 FP8，frame cosine、PSNR 和 mean SSIM 均有所改善，音频距离指标则有升有降。以下播放器展示三组完整输出，并支持同步播放。

<div class="video-grid video-grid-three" data-sync-group="smoothing">
  <figure>
    <figcaption>BF16 Linear + FlashAttention 4</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="bf16-quality"></video>
  </figure>
  <figure>
    <figcaption>FP8 Linear + FP8 Sol，不使用 smoothing</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="fp8-unsmoothed"></video>
  </figure>
  <figure>
    <figcaption>FP8 Linear + FP8 Sol，开启误差修正</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="fp8-smoothed"></video>
  </figure>
</div>

## 哪些位置仍然保留稠密 attention

去噪早期和部分 Transformer 层对长程依赖更敏感。TeleFuser 可以保留开头若干次完整 attention，也可以指定部分层始终使用稠密计算，其余位置使用 Sol-Attn。本文的 FastH3 配置保留两个开头 update 和两个 dense layer，之后使用 `tau=1.0` 的 exact routing。Dense step、dense layer、threshold mode 和 `tau` 均可按 schedule 调整。

## 多卡执行

Q-SPA 使用 Ulysses SP 拆分长序列，并与 Tensor Parallel 组合。稀疏统计基于 Ulysses 的 attention 视图计算，单卡和多卡因此保持相同的 block 语义。运行时还会重叠 Ulysses 通信和 attention 计算；FP8 与稀疏计算越快，这部分通信隐藏带来的收益越明显。
