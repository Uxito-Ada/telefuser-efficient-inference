<!--
LOCALIZED-SOURCE
id: 04-evaluation
language: zh-CN
-->

# 性能与生成效果 {#results}

本文数据来自当时可用的 H100 80GB，图中保留硬件型号只是为了方便复现。

## 四卡 Base H3

主测试使用四张 GPU 跑 MiniMax-H3 Base。LightX2V 和 TeleFuser 都开 `TP2 x Ulysses SP2`，prompt、seed、分辨率、帧数、帧率和 50-point schedule 全部一致，并关闭 feature cache。LightX2V 采用能够正确生成视频的 BF16 + SageAttention2；TeleFuser 采用 FP8 Linear + Q-SPA。

![四 GPU MiniMax-H3 Base 性能](assets/lightx2v-base-h3.svg)

<div class="result-summary">
  <div><strong>快 2.64 倍</strong><span>完整生成</span></div>
  <div><strong>快 2.62 倍</strong><span>去噪，step 吞吐提升 162.2%</span></div>
  <div><strong>降低 40.3%</strong><span>代表性单卡峰值显存</span></div>
</div>

完整生成时间从 137.76 秒降到 52.27 秒，去噪时间从 129.22 秒降到 49.28 秒。两边使用相同的四卡并行配置，因此可以直接比较加速比。

<div class="video-pair" data-sync-group="base-h3">
  <figure>
    <figcaption>LightX2V：BF16 + SageAttention2，TP2 x Ulysses SP2</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="lightx2v-base-h3"></video>
  </figure>
  <figure>
    <figcaption>TeleFuser：FP8 Linear + Q-SPA，TP2 x Ulysses SP2</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="telefuser-base-h3"></video>
  </figure>
</div>

两段输出都是 124 帧拉面场景，并带有同步双声道音频。每个框架先 warm-up 一次，再记录一次正式请求。GPU 0 当时有一份固定的无关显存占用，因此图中的显存取其余三张空闲卡的峰值中位数。

## Adapter 工作负载

下面再看两个常用模型变体：MiniMax-H3 Turbo LoRA 和 FastH3 dense hybrid adapter。每组都选择能够正确运行该模型的外部框架作为对照，并对齐输入、输出规格和 DiT 调用次数。

### MiniMax-H3 Turbo LoRA

Turbo 测试使用单张 GPU 和 8-step v1.0 768p Adapter，总共执行八次 DiT。两边的 DiT 都常驻 GPU：LightX2V 使用 BF16 + Sol，TeleFuser 先合并 LoRA，再生成 FP8 权重并运行 Q-SPA。CPU block offload 的数据没有画进来。

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
    <figcaption>TeleFuser：MiniMax-H3 Turbo，FP8 Linear + Q-SPA</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="turbo-telefuser"></video>
  </figure>
</div>

### FastH3 dense adapter

FastH3 同样使用单张 GPU。两边采用相同的 dense adapter、prompt、seed、1344 x 768 输出和 124 帧配置，实际执行四次 DiT。FastVideo 使用 BF16 Linear + FlashAttention 4，TeleFuser 使用 FP8 Linear + Q-SPA；去噪时都不 offload DiT。每边先 warm-up 一次，正式结果取三次生成的中位数。

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
    <figcaption>TeleFuser：FastH3，FP8 Linear + Q-SPA</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="telefuser-primary"></video>
  </figure>
</div>

FastH3 只比较去噪阶段，因为两套框架对 prompt-conditioning cache 的处理不同，完整请求时间不能直接相除。上面的四段 Adapter 输出均为 1344 x 768、124 帧的 H.264 视频，并带有 32kHz 双声道 AAC 音频。

## 通信计算重叠

四卡执行还可以把 Ulysses 通信藏到 attention 计算后面。另一组五个 case 的回归测试中，平均请求时间从 78.546 秒降到 75.766 秒，改善 **3.539%**；五个 MP4 都保持 byte-identical。这项优化已经用在四卡 Base H3 的分布式执行中。
