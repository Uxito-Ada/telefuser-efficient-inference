<!--
LOCALIZED-SOURCE
id: 04-evaluation
language: zh-CN
-->

# H100 Results：性能与效果

## 四卡 Base H3

主要框架对比在四张 H100 80GB 上运行 MiniMax-H3 Base。LightX2V 与 TeleFuser 均使用 `TP2 x Ulysses SP2`，并对齐 prompt、seed、1344 x 768 输出、124 帧、24 FPS 和 50-point schedule，同时关闭 feature cache。LightX2V 使用输出正确的 BF16 + SageAttention2 路径；TeleFuser 使用 FP8 Linear + 质量感知 FP8 Sol-Attn。

![四卡 MiniMax-H3 Base 性能](assets/lightx2v-base-h3.svg)

<div class="result-summary">
  <div><strong>快 2.64 倍</strong><span>相比 LightX2V 的生成时间</span></div>
  <div><strong>快 2.62 倍</strong><span>去噪速度，step 吞吐提升 162.2%</span></div>
  <div><strong>降低 40.3%</strong><span>代表性单卡峰值显存</span></div>
</div>

生成时间从 137.76 秒降至 52.27 秒，去噪时间从 129.22 秒降至 49.28 秒。这是本文主要的端到端结果：外部 baseline 与 TeleFuser 使用相同的四卡并行拓扑，而不是用分布式结果与单卡结果比较。

<div class="video-pair" data-sync-group="base-h3">
  <figure>
    <figcaption>LightX2V：BF16 + SageAttention2，TP2 x Ulysses SP2</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="lightx2v-base-h3"></video>
  </figure>
  <figure>
    <figcaption>TeleFuser：FP8 Linear + 质量感知 FP8 Sol，TP2 x Ulysses SP2</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="telefuser-base-h3"></video>
  </figure>
</div>

两段视频均生成了连贯的 124 帧拉面场景和同步双声道音频。性能记录包含一次 warm-up 和一次正式测量；由于 GPU 0 在两次测试期间都存在固定的无关占用，显存采用其余空闲 GPU 的代表性峰值。

## Adapter 工作负载

TeleFuser 同时支持 MiniMax-H3 Turbo LoRA 与 FastH3 dense hybrid adapter。两组测试分别选择能够正确运行对应 Adapter 的外部框架，并对齐模型输入、输出规格与 DiT 计算量。

### MiniMax-H3 Turbo LoRA

Turbo 测试使用单张 H100、8-step v1.0 768p Adapter 和八次 DiT update，两种框架均将 DiT 常驻 GPU。LightX2V 使用 BF16 + Sol；TeleFuser 在 FP8 转换前合并 LoRA，再运行 FP8 Linear + FP8 Sol。图中不包含 CPU block offload 数据。

![MiniMax-H3 Turbo Adapter 性能](assets/turbo-performance.svg)

<div class="result-summary">
  <div><strong>降低 26.7%</strong><span>相比 LightX2V 的去噪时间</span></div>
  <div><strong>提升 36.5%</strong><span>8-step 去噪吞吐</span></div>
  <div><strong>降低 12.3%</strong><span>进程峰值 GPU 显存</span></div>
</div>

<div class="video-pair" data-sync-group="turbo">
  <figure>
    <figcaption>LightX2V：MiniMax-H3 Turbo，常驻 BF16 DiT + Sol</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="turbo-lightx2v"></video>
  </figure>
  <figure>
    <figcaption>TeleFuser：MiniMax-H3 Turbo，FP8 Linear + FP8 Sol</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="turbo-telefuser"></video>
  </figure>
</div>

### FastH3 dense adapter

FastH3 测试在单张 H100 上对齐 dense adapter、prompt、seed、1344 x 768 输出、124 帧和四次实际 DiT 执行。FastVideo 使用 BF16 Linear + FlashAttention 4；TeleFuser 使用 FP8 Linear + 质量感知 FP8 Sol-Attn。两者在去噪阶段都没有 offload DiT。结果经过一次 warm-up，并取三次正式生成的中位数。

![FastH3 Adapter 对齐性能](assets/end-to-end.svg)

<div class="result-summary">
  <div><strong>降低 25.7%</strong><span>相比 FastVideo 的去噪时间</span></div>
  <div><strong>提升 34.6%</strong><span>实际 DiT forward 吞吐</span></div>
  <div><strong>降低 14.3%</strong><span>进程峰值 GPU 显存</span></div>
</div>

<div class="video-pair" data-sync-group="fasth3">
  <figure>
    <figcaption>FastVideo：FastH3，BF16 Linear + FlashAttention 4</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="fastvideo-primary"></video>
  </figure>
  <figure>
    <figcaption>TeleFuser：FastH3，FP8 Linear + 质量感知 FP8 Sol</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="telefuser-primary"></video>
  </figure>
</div>

FastH3 图只比较口径一致的去噪阶段，因为两套记录使用了不同的 prompt-conditioning cache 策略。以上四段 Adapter 输出都包含 124 帧 1344 x 768 H.264 视频和 32kHz 双声道 AAC 音频。

## 通信计算重叠

TeleFuser 的四卡 Base 路径还包含通信计算重叠。在另一组五 case 回归测试中，该优化将平均请求时间从 78.546 秒降低到 75.766 秒，提升 **3.539%**，全部 MP4 均保持 byte-identical。它与正文四卡 Base H3 主结果使用同一条分布式执行路径。
