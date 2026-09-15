<!--
LOCALIZED-SOURCE
id: 04-evaluation
language: zh-CN
-->

# 性能与生成效果 {#results}

## 测试配置

除特别说明外，测试均使用 MiniMax-H3 的 768p 配置，输出 24 fps H.264 视频和 32 kHz 双声道 AAC 音频。性能结果来自 warm-up 后的正式请求。

| 实验 | 模型与任务 | 输出规格 | 采样 | GPU | 并行拓扑 |
|---|---|---|---|---:|---|
| TeleFuser 扩展性 | Base H3，T2VA | 1344 × 768，124 帧，5 秒 | 50 points / 49 DiT updates | 1 / 2 / 4 | 单卡 / TP2 / TP2 × Ulysses SP2 |
| 四卡框架对比 | Base H3，T2VA | 1344 × 768，124 帧，5 秒 | 50 points / 49 DiT updates | 4 | TP2 × Ulysses SP2 |
| FP8 smoothing | Base H3，T2VA | 1344 × 768，107 帧，4 秒 | 50 points / 49 DiT updates | 1 | 单卡 |
| Turbo LoRA | MiniMax-H3 Turbo，I2AV | 1344 × 768，124 帧，5 秒 | 9 points / 8 DiT updates | 1 | 单卡 |
| FastH3 Adapter | FastH3 dense，T2VA | 1344 × 768，124 帧，5 秒 | 5 points / 4 DiT updates | 1 | 单卡 |

扩展性实验关闭 feature cache 和 KV smoothing，用于单独测量并行收益。Smoothing、Turbo 和 FastH3 的具体低精度配置在对应小节列出。

## TeleFuser 1、2、4 卡扩展性

三组运行使用相同 prompt、seed、FP8 Linear 和 FP8 Sol-Attn。单卡直接执行；两卡使用 TP2 切分模型权重和计算；四卡在 TP2 之上增加 Ulysses SP2 切分长序列。

![TeleFuser MiniMax-H3 Base 1、2、4 卡扩展性能](assets/base-scaling.svg)

<div class="result-summary">
  <div><strong>提升 3.40 倍</strong><span>四卡相对单卡的去噪吞吐</span></div>
  <div><strong>降低 70.6%</strong><span>去噪时间：167.41 → 49.28 秒</span></div>
  <div><strong>降低 35.5%</strong><span>单卡最大采样峰值显存</span></div>
</div>

去噪时间从单卡的 167.41 秒降至两卡的 90.90 秒，再降至四卡的 49.28 秒。相邻两次扩展分别获得 1.84 倍和 1.84 倍加速；50-step 吞吐为 0.299、0.550 和 1.015 step/s。

## 四卡 Base H3 框架对比

SGLang、LightX2V 与 TeleFuser 均使用四张 H100 和 `TP2 × Ulysses SP2`。请求采用同一 Base H3 输出规格，不启用 feature cache。[SGLang 官方 MiniMax-H3 cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx)列出了该 H100 拓扑；本图采用 TeleFuser 文档中记录的同规格本机测试结果。

![四 GPU MiniMax-H3 Base 性能](assets/lightx2v-base-h3.svg)

<div class="result-summary">
  <div><strong>快 2.64 倍</strong><span>相比 LightX2V 的完整生成</span></div>
  <div><strong>快 1.52 倍</strong><span>相比 SGLang 的完整请求</span></div>
  <div><strong>降低 40.3% / 37.3%</strong><span>相比 LightX2V / SGLang 的峰值显存</span></div>
</div>

TeleFuser 完整生成耗时 52.27 秒；LightX2V 为 137.76 秒，SGLang 为 79.37 秒。对应地，TeleFuser 比 LightX2V 降低 62.1% 请求时延和 40.3% 峰值显存，比 SGLang 降低 34.1% 请求时延和 37.3% 峰值显存。LightX2V 与 TeleFuser 的去噪时间分别为 129.22 秒和 49.28 秒，TeleFuser 的 50-step 吞吐提高 162.2%。

<div class="video-pair" data-sync-group="base-h3">
  <figure>
    <figcaption>LightX2V</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="lightx2v-base-h3"></video>
  </figure>
  <figure>
    <figcaption>TeleFuser</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="telefuser-base-h3"></video>
  </figure>
</div>

两段视频采用相同 prompt 和 seed，均包含 124 帧画面与同步立体声音频。每个框架先 warm-up 一次，再记录一次正式请求。显存指标采用 `nvidia-smi` 以 100 ms 间隔采样。

## Adapter 工作负载

### MiniMax-H3 Turbo LoRA

两套框架均使用 8-step v1.0 768p Adapter，DiT 常驻 GPU。TeleFuser 在构建 FP8 权重前合并 LoRA；对比不包含 CPU block offload。

![MiniMax-H3 Turbo Adapter 性能](assets/turbo-performance.svg)

<div class="result-summary">
  <div><strong>降低 26.7%</strong><span>相比 LightX2V 的去噪时间</span></div>
  <div><strong>提升 36.5%</strong><span>8-step 去噪吞吐</span></div>
  <div><strong>降低 12.3%</strong><span>进程峰值 GPU 显存</span></div>
</div>

<div class="video-pair" data-sync-group="turbo">
  <figure>
    <figcaption>LightX2V：MiniMax-H3 Turbo</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="turbo-lightx2v"></video>
  </figure>
  <figure>
    <figcaption>TeleFuser：MiniMax-H3 Turbo</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="turbo-telefuser"></video>
  </figure>
</div>

性能测试使用一致的输入。展示视频使用固定水平机位的 prompt，避免旋转镜头干扰 Adapter 输出的观察。

### FastH3 dense adapter

FastVideo 和 TeleFuser 使用相同 dense adapter、prompt 与 seed，实际执行四次 DiT，去噪期间均不使用 CPU offload。结果为一次 warm-up 后三次正式生成的中位数。

![FastH3 Adapter 对齐性能](assets/end-to-end.svg)

<div class="result-summary">
  <div><strong>降低 25.7%</strong><span>相比 FastVideo 的去噪时间</span></div>
  <div><strong>提升 34.6%</strong><span>实际 DiT forward 吞吐</span></div>
  <div><strong>降低 14.3%</strong><span>进程峰值 GPU 显存</span></div>
</div>

<div class="video-pair" data-sync-group="fasth3">
  <figure>
    <figcaption>FastVideo：FastH3</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="fastvideo-primary"></video>
  </figure>
  <figure>
    <figcaption>TeleFuser：FastH3</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="telefuser-primary"></video>
  </figure>
</div>

FastH3 只比较去噪阶段，因为两套记录采用了不同的 prompt-conditioning cache 策略，完整请求时间不具备相同口径。
