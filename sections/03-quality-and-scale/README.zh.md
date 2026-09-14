<!--
LOCALIZED-SOURCE
id: 03-quality-and-scale
language: zh-CN
-->

# 控制误差，并扩展到多卡

Diffusion 的每一步输出都会成为下一步输入。FP8 或稀疏 attention 在某一步引入的误差，可能沿着去噪过程继续累积。因此，我们没有只看 kernel 吞吐，还检查了 tensor 误差、最终视频和音频。

## FP8 attention 的误差控制

对 MiniMax-H3 实际层做 profile 后，我们发现部分 K/V 的均值明显偏离零点。直接做对称 FP8 量化会浪费一部分动态范围。TeleFuser 在计算前先对 K/V 做中心化，再在 attention 输出中补回等价偏移。这可以看作专门针对 attention 的非对称量化处理。

中心化和输出修正已经融合进 FP8 Sol-Attn。最早的非融合版本让去噪时间增加 11.7%，融合后开销降到 2.2%。在捕获的真实 H3 层上，K 的量化 MSE 降低 21.65%，attention 输出 MSE 降低 8.18%；KV smoothing 和 V correction 在优化配置中默认开启。

50-step 测试中，这个版本的去噪吞吐仍比 BF16 Linear + FlashAttention 4 高 39.4%，peak allocated memory 低 42.6%。相较没有 smoothing 的 FP8，frame cosine、PSNR 和 mean SSIM 都有改善，音频距离指标则有升有降。下面三段是完整输出，可以同步播放。

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

去噪早期和部分 Transformer 层对长程依赖更敏感。TeleFuser 可以保留开头若干次完整 attention，也可以指定部分层始终使用稠密计算，其余位置再交给 Sol-Attn。本文的 FastH3 配置保留两个开头 update 和两个 dense layer，之后使用 `tau=1.0` 的 exact routing。Dense step、dense layer、threshold mode 和 `tau` 都可以按 schedule 调整。

## 多卡执行

Q-SPA 使用 Ulysses SP 拆分长序列，并与 Tensor Parallel 组合。稀疏统计基于 Ulysses 的 attention 视图计算，单卡和多卡因此保持相同的 block 语义。运行时还会重叠 Ulysses 通信和 attention 计算；FP8 与稀疏计算越快，这部分通信隐藏带来的收益越明显。
