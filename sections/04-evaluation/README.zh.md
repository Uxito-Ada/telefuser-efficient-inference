<!--
LOCALIZED-SOURCE
id: 04-evaluation
language: zh-CN
-->

# 性能与生成效果 {#results}

## 测试配置

除特别说明外，测试均使用 MiniMax-H3 的 768p 配置，输出 24 fps H.264 视频和 32 kHz 双声道 AAC 音频。

| 实验 | 模型与任务 | 输出规格 | 采样 | GPU | 并行拓扑 |
|---|---|---|---|---:|---|
| TeleFuser 扩展性 | Base H3，T2VA | 1344 × 768，124 帧，5 秒 | 50 points / 49 DiT updates | 1 / 2 / 4 | 单卡 / TP2 / TP2 × Ulysses SP2 |
| 四卡框架对比 | Base H3，T2VA | 1344 × 768，124 帧，5 秒 | 50 points / 49 DiT updates | 4 | TP2 × Ulysses SP2；FastVideo 为 SP4 |
| FP8 smoothing | Base H3，T2VA | 1344 × 768，107 帧，4 秒 | 50 points / 49 DiT updates | 1 | 单卡 |
| Turbo LoRA | MiniMax-H3 Turbo，I2AV | 1344 × 768，124 帧，5 秒 | 9 points / 8 DiT updates | 1 | 单卡 |
| FastH3 Adapter | FastH3 dense，T2VA | 1344 × 768，124 帧，5 秒 | 5 points / 4 DiT updates | 1 | 单卡 |

扩展性实验关闭 feature cache 和 KV smoothing，用于单独测量并行收益。Smoothing、Turbo 和 FastH3 的具体低精度配置在对应小节列出。

## TeleFuser 扩展性

三组运行使用相同 prompt、seed、FP8 Linear 和 FP8 Sol-Attn。单卡直接执行；两卡使用 TP2 切分模型权重和计算；四卡在 TP2 之上增加 Ulysses SP2 切分长序列。

![TeleFuser MiniMax-H3 Base 1、2、4 卡扩展性能](assets/base-scaling.svg)

<div class="result-summary">
  <div><strong>提升 3.40 倍</strong><span>四卡相对单卡的去噪吞吐</span></div>
  <div><strong>降低 70.6%</strong><span>去噪时间：167.41 → 49.28 秒</span></div>
  <div><strong>降低 35.5%</strong><span>单卡最大采样峰值显存</span></div>
</div>

去噪时间从单卡的 167.41 秒降至两卡的 90.90 秒，再降至四卡的 49.28 秒。相邻两次扩展分别获得 1.84 倍和 1.84 倍加速；50-step 吞吐为 0.299、0.550 和 1.015 step/s。

## 四卡 Base H3 框架对比

四个框架采用相同的 Base H3 输出规格和 50-point schedule，并关闭 feature cache。SGLang、LightX2V 和 TeleFuser 使用 `TP2 × Ulysses SP2`，FastVideo 使用 `SP4`。FastVideo 与 SGLang 分别使用 [Base H3 官方示例](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_minimax_h3_t2v.py)和 [MiniMax-H3 官方 cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx)。

![四 GPU MiniMax-H3 Base 性能](assets/lightx2v-base-h3.svg)

<div class="result-summary">
  <div><strong>快 2.64 倍</strong><span>相比 LightX2V 的完整生成</span></div>
  <div><strong>快 2.19 倍</strong><span>相比 FastVideo 的完整请求</span></div>
  <div><strong>降低 40.3% / 37.3%</strong><span>相比 LightX2V / SGLang 的峰值显存</span></div>
</div>

TeleFuser 完整生成耗时 52.27 秒；LightX2V 为 137.76 秒，FastVideo 为 114.70 秒，SGLang 为 79.37 秒。在这一请求上，TeleFuser 相比三者分别快 2.64 倍、2.19 倍和 1.52 倍。TeleFuser 峰值显存为 42.5 GiB/GPU，LightX2V、FastVideo 和 SGLang 分别为 71.2、41.9 和 67.8 GiB。相对 LightX2V，TeleFuser 去噪时间从 129.22 秒降至 49.28 秒，50-point 吞吐提高 162.2%。

**Prompt：** `Steam rises from the ramen while the family talks in the background.`

<div class="video-pair" data-sync-group="base-h3">
  <figure>
    <figcaption>LightX2V</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="lightx2v-base-h3"></video>
  </figure>
  <figure>
    <figcaption>FastVideo</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="fastvideo-base-h3"></video>
  </figure>
  <figure>
    <figcaption>SGLang</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="sglang-base-h3"></video>
  </figure>
  <figure>
    <figcaption>TeleFuser</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="telefuser-base-h3"></video>
  </figure>
</div>

## Adapter 工作负载

Turbo LoRA 与 FastH3 使用不同的权重和采样协议，当前框架支持范围如下。

| 框架 | MiniMax-H3 Turbo LoRA | FastH3 Preview Adapter |
|---|---|---|
| TeleFuser | ✅ 已支持 | ✅ 已支持 |
| LightX2V | ✅ 已支持 | ❌ 未支持 |
| FastVideo | ❌ 未支持 | ✅ 已支持 |
| SGLang | ✅ 已支持 | ❌ 未支持 |

Turbo LoRA 对比 TeleFuser 与 LightX2V，FastH3 对比 TeleFuser 与 FastVideo。

### MiniMax-H3 Turbo LoRA

两套框架均使用 8-step v1.0 768p Adapter，DiT 常驻 GPU。TeleFuser 在构建 FP8 权重前合并 LoRA；对比不包含 CPU block offload。

![MiniMax-H3 Turbo Adapter 性能](assets/turbo-performance.svg)

<div class="result-summary">
  <div><strong>降低 26.7%</strong><span>相比 LightX2V 的去噪时间</span></div>
  <div><strong>提升 36.5%</strong><span>8-step 去噪吞吐</span></div>
  <div><strong>降低 12.3%</strong><span>进程峰值 GPU 显存</span></div>
</div>

**Prompt：** `Steam rises from the ramen while the family talks in the background.`

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

### FastH3 dense adapter

FastVideo 和 TeleFuser 使用相同 dense adapter、prompt 与 seed，实际执行四次 DiT，去噪期间均不使用 CPU offload。结果为一次 warm-up 后三次正式生成的中位数。

![FastH3 Adapter 对齐性能](assets/end-to-end.svg)

<div class="result-summary">
  <div><strong>降低 25.7%</strong><span>相比 FastVideo 的去噪时间</span></div>
  <div><strong>提升 34.6%</strong><span>实际 DiT forward 吞吐</span></div>
  <div><strong>降低 14.3%</strong><span>进程峰值 GPU 显存</span></div>
</div>

**Prompt：** `integrated_multimodal_description: A red fox runs through fresh snow at dawn. overall_soundscape: Fast pawsteps in snow, winter wind, and distant birds.`

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
