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

质量实验使用固定机位拍摄雪地电车，三组运行采用相同 prompt、seed 和输出规格。车体、车窗、轨道和受电弓的刚性结构可直接反映时序几何是否稳定。

| 模型 | 分辨率与帧数 | 采样 | GPU | Prompt / seed |
|---|---|---|---:|---|
| MiniMax-H3 Base，T2VA | 1344 × 768，107 帧，4 秒，24 fps | 50 points / 49 DiT updates | 1 × H100 | 雪地电车固定机位 / 17 |

![MiniMax-H3 FP8 smoothing 单卡性能](assets/smoothing-performance.svg)

平滑 FP8 的去噪吞吐比 BF16 Linear + FlashAttention 4 提高 37.2%，峰值分配显存降低 42.6%；相对未平滑 FP8，融合修正增加 2.1% 去噪时间。该 seed 的音频 cosine 从 0.850 提升到 0.889，频谱收敛误差从 0.506 降到 0.454；视频 PSNR 和 SSIM 分别变化 -0.36 dB 和 -0.0013。局部 tensor 误差和最终媒体指标并不等价，因此两类结果分别报告。以下播放器可同步检查三段完整输出。

<div class="video-grid video-grid-three" data-sync-group="smoothing">
  <figure>
    <figcaption>BF16 参考</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="bf16-quality"></video>
  </figure>
  <figure>
    <figcaption>FP8，不使用 smoothing</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="fp8-unsmoothed"></video>
  </figure>
  <figure>
    <figcaption>FP8，开启 smoothing</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="fp8-smoothed"></video>
  </figure>
</div>

三段输出的差异在视频后半段最明显：未平滑 FP8 的车顶标识更杂乱，受电弓连线和窗框也不够规整。下面同步放大三段原始视频的同一区域。

<div class="video-grid video-grid-three" data-sync-group="smoothing-detail">
  <figure>
    <figcaption>BF16 参考局部</figcaption>
    <video controls muted playsinline preload="metadata" data-result-slot="bf16-quality-detail"></video>
  </figure>
  <figure>
    <figcaption>FP8，未平滑局部</figcaption>
    <video controls muted playsinline preload="metadata" data-result-slot="fp8-unsmoothed-detail"></video>
  </figure>
  <figure>
    <figcaption>FP8，平滑后局部</figcaption>
    <video controls muted playsinline preload="metadata" data-result-slot="fp8-smoothed-detail"></video>
  </figure>
</div>

## 哪些位置仍然保留稠密 attention

去噪早期和部分 Transformer 层对长程依赖更敏感。TeleFuser 可以保留开头若干次完整 attention，也可以指定部分层始终使用稠密计算，其余位置使用 Sol-Attn。本文的 FastH3 配置保留两个开头 update 和两个 dense layer，之后使用 `tau=1.0` 的 exact routing。Dense step、dense layer、threshold mode 和 `tau` 均可按 schedule 调整。

## 多卡执行

Q-SPA 使用 Ulysses SP 拆分长序列，并与 Tensor Parallel 组合。稀疏统计基于 Ulysses 的 attention 视图计算，单卡和多卡因此保持相同的 block 语义。运行时还会重叠 Ulysses 通信和 attention 计算；FP8 与稀疏计算越快，这部分通信隐藏带来的收益越明显。
