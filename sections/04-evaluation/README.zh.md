<!--
LOCALIZED-SOURCE
id: 04-evaluation
language: zh-CN
-->

# H100 Results：性能与效果

主要对比选择 FastVideo 的 MiniMax-H3 FastH3 Dense/Data-Free Adapter 作为外部 baseline。两种方案均使用单张 H100 80GB，以及相同的 Base checkpoint、Adapter strength、prompt、seed、1344 x 768 分辨率、124 帧、24 FPS 和包含四次 DiT 执行的五点 schedule。每组结果均经过一次完整 warm-up，并取三次正式生成的中位数。

FastVideo 使用 BF16 Linear + FlashAttention 4；TeleFuser 使用 FP8 Linear + FP8 Sol-Attn，并配置 `tau=1.0`、两个 dense opening update、两个 dense layer、KV smoothing 和 V correction。两种方案在去噪期间都没有将 DiT offload 到 CPU。

![MiniMax-H3 Adapter 对齐性能测试](assets/end-to-end.svg)

<div class="result-summary">
  <div><strong>降低 25.7%</strong><span>相比 FastVideo BF16 + FA4 的去噪时间</span></div>
  <div><strong>提升 34.6%</strong><span>实际 DiT forward 吞吐</span></div>
  <div><strong>降低 14.3%</strong><span>进程峰值 GPU 显存</span></div>
</div>

去噪时间中位数从 21.685 秒降至 16.116 秒。每段视频实际执行四次 DiT，因此吞吐从每秒 0.184 次 forward 提升至 0.248 次。整个进程的峰值显存从 79.03 GiB 降至 67.70 GiB。显存并没有按 FP8 权重比例减半，是因为文本与媒体编码器、非 Linear 权重、activation 和 workspace 仍包含在进程峰值中。

图中只比较口径一致的去噪阶段；两次记录使用了不同的 prompt-conditioning cache 策略，因此没有将 whole-request latency 合并为加速比。

## 生成效果

两段输出均包含 124 帧 1344 x 768 H.264 视频和 32kHz 双声道 AAC 音频。HTML 播放器会同步两段视频，便于直接比较。

<div class="video-pair" data-sync-group="primary">
  <figure>
    <figcaption>FastVideo：BF16 Linear + FlashAttention 4</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="fastvideo-primary"></video>
  </figure>
  <figure>
    <figcaption>TeleFuser：FP8 Linear + 高质量 FP8 Sol-Attn</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="telefuser-primary"></video>
  </figure>
</div>

前面的 50-step ablation 在同一运行时中对比了 BF16、普通 FP8 和质量感知 FP8。这里的 FastH3 视频则展示了最终 Adapter 配置在性能测试条件下生成的完整视频与音频。

## 多 GPU 执行

另一组 Base H3 测试在四张 H100 上运行 TeleFuser 的常驻 `TP2 x Ulysses SP2` 配置。在五组 prompt/seed 中，通信计算重叠将平均请求时间从 78.546 秒降低至 75.766 秒，提升 **3.539%**，五段 MP4 输出均保持 byte-identical。

这组结果使用 50-point Base H3 schedule，因此与五点 FastH3 Adapter 外部对比分开呈现。两组实验分别覆盖 TeleFuser 的单卡低精度稀疏性能和多卡分布式能力。
