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

## 统一性能对比

图中以去噪吞吐表示速度，以峰值显存表示资源占用。Base H3 使用四卡；Adapter 暂保留当前已验证的单卡记录，四卡重跑后替换对应点。

![MiniMax-H3 Base 与 Adapter 性能对比](assets/all-workloads-performance.svg)

Base H3 对比采用 [FastVideo 官方示例](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_minimax_h3_t2v.py)、[SGLang 官方 cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx)以及匹配的 LightX2V/TeleFuser 配置。TeleFuser Base H3 去噪吞吐为 1.015 step/s，LightX2V 为 0.387，FastVideo 为 0.496；Turbo Adapter 将调度缩短到 8 次 DiT 更新后，吞吐为 0.306 step/s。

**Prompt：** `Steam rises from the ramen while the family talks in the background.`

<div class="video-grid video-grid-four" data-sync-group="base-h3">
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

| 框架 | MiniMax-H3 Turbo LoRA | FastH3 Preview Adapter |
|---|---|---|
| TeleFuser | ✅ 已支持 | ✅ 已支持 |
| LightX2V | ✅ 已支持 | ❌ 未支持 |
| FastVideo | ❌ 未支持 | ✅ 已支持 |
| SGLang | ✅ 已支持 | ❌ 未支持 |

Turbo LoRA 对比 TeleFuser 与 LightX2V，FastH3 对比 TeleFuser 与 FastVideo。

### MiniMax-H3 Turbo LoRA

两套框架均使用 8-step v1.0 768p Adapter，DiT 常驻 GPU。TeleFuser 在构建 FP8 权重前合并 LoRA；对比不包含 CPU block offload。



**Prompt：** `Steam rises from the ramen while the family talks in the background. Bright, warm indoor lighting illuminates every face and the room with natural skin tones. The man holds a pair of straight, rigid chopsticks that remain perfectly straight throughout the video and never bend.`

<div class="video-grid video-grid-two" data-sync-group="turbo">
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



**Prompt：** `integrated_multimodal_description: A red fox runs through fresh snow at dawn. overall_soundscape: Fast pawsteps in snow, winter wind, and distant birds.`

<div class="video-grid video-grid-two" data-sync-group="fasth3">
  <figure>
    <figcaption>FastVideo：FastH3</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="fastvideo-primary"></video>
  </figure>
  <figure>
    <figcaption>TeleFuser：FastH3</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="telefuser-primary"></video>
  </figure>
</div>
