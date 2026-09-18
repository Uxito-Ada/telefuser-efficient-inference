<!--
LOCALIZED-SOURCE
id: 04-evaluation
language: zh-CN
-->

# 性能与生成效果 {#results}

## 测试配置

| 实验 | 模型与任务 | 输出规格 | 采样 | GPU | 并行拓扑 |
|---|---|---|---|---:|---|
| TeleFuser 扩展性 | Base H3，T2VA | 1344 × 768，124 帧，5 秒，24 fps | 50 次去噪步 | 1 / 2 / 4 | 单卡 / TP2 / TP2 × Ulysses SP2 |
| 框架对比 | Base H3，T2VA | 1344 × 768，124 帧，5 秒，24 fps | 50 次去噪步 | 4 | 各框架原生分布式路径 |
| Turbo LoRA | MiniMax-H3 Turbo，T2VA | 1344 × 768，124 帧，5 秒，24 fps | 8 次去噪步 | 4 | TP2 × Ulysses SP2 |
| FastH3 Adapter | FastH3 dense，T2VA | 1344 × 768，124 帧，5 秒，24 fps | 4 次去噪步 | 4 | 各框架原生分布式路径 |
| FP8 smoothing | Base H3，T2VA | 1344 × 768，107 帧，4 秒，24 fps | 50 次去噪步 | 1 | 单卡 |

## 四卡统一性能对比

柱形表示峰值 GPU 显存，折线表示去噪吞吐。所有数据均使用四张 H100，且未启用 CPU offload。

![MiniMax-H3 Base 与 Adapter 性能对比](assets/all-workloads-performance.svg)

Base H3 使用 [FastVideo 官方示例](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_minimax_h3_t2v.py)与 [SGLang 官方 cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx)，并采用 LightX2V 和 TeleFuser 的对应示例。Turbo LoRA 对比 TeleFuser、SGLang 与 LightX2V，FastH3 对比 TeleFuser 与 FastVideo。

- **Base H3：** TeleFuser 的端到端吞吐相比 LightX2V、FastVideo 和 SGLang 分别提高 163.5%、119.4% 和 51.8%。峰值显存相比 LightX2V 和 SGLang 分别降低 40.3% 和 37.3%，与 FastVideo 的差异为 1.5%。
- **Turbo LoRA：** TeleFuser 的端到端吞吐相比 LightX2V 提高 58.8%，相比 SGLang 提高 0.3%；峰值显存相比 LightX2V 降低 42.3%，相比 SGLang 降低 26.3%。
- **FastH3：** TeleFuser 的端到端吞吐相比 FastVideo 提高 208.5%，峰值显存降低 37.4%。

## TeleFuser 扩展性

![TeleFuser Base H3 扩展性](assets/base-scaling.svg)

在测试的单卡、双卡和四卡配置中，量化后的分布式路径保持了接近线性的扩展，
同时降低了单卡显存占用。释放出的显存空间可以支持更大规模的视频生成请求，而
不改变模型和输出接口。

## 生成效果

| 框架 | Base H3 | MiniMax-H3 Turbo LoRA | FastH3 Preview Adapter |
|---|---|---|---|
| TeleFuser | ✅ 已支持 | ✅ 已支持 | ✅ 已支持 |
| LightX2V | ✅ 已支持 | ✅ 已支持 | ❌ 未支持 |
| FastVideo | ✅ 已支持 | ❌ 未支持 | ✅ 已支持 |
| SGLang | ✅ 已支持 | ✅ 已支持 | ❌ 未支持 |

**Base H3 与 Turbo LoRA Prompt：** `Steam rises from the ramen while the family talks in the background.`

**FastH3 Prompt：** `Steam rises from the ramen while the family talks in the background.`

<div class="video-matrix">
  <div></div>
  <div class="video-matrix-heading">Base H3</div>
  <div class="video-matrix-heading">Turbo LoRA</div>
  <div class="video-matrix-heading">FastH3</div>

  <div class="video-matrix-label">LightX2V</div>
  <figure><video controls playsinline preload="metadata" data-result-slot="lightx2v-base-h3"></video></figure>
  <figure><video controls playsinline preload="metadata" data-result-slot="turbo-lightx2v"></video></figure>
  <div class="video-matrix-empty">暂不支持</div>

  <div class="video-matrix-label">FastVideo</div>
  <figure><video controls playsinline preload="metadata" data-result-slot="fastvideo-base-h3"></video></figure>
  <div class="video-matrix-empty">暂不支持</div>
  <figure><video controls playsinline preload="metadata" data-result-slot="fastvideo-primary"></video></figure>

  <div class="video-matrix-label">SGLang</div>
  <figure><video controls playsinline preload="metadata" data-result-slot="sglang-base-h3"></video></figure>
  <figure><video controls playsinline preload="metadata" data-result-slot="turbo-sglang"></video></figure>
  <div class="video-matrix-empty">暂不支持</div>

  <div class="video-matrix-label">TeleFuser</div>
  <figure><video controls playsinline preload="metadata" data-result-slot="telefuser-base-h3"></video></figure>
  <figure><video controls playsinline preload="metadata" data-result-slot="turbo-telefuser"></video></figure>
  <figure><video controls playsinline preload="metadata" data-result-slot="telefuser-primary"></video></figure>
</div>

## FP8 attention smoothing

最早的非融合实现使去噪时间增加 11.7%，融合后开销降至 2.2%。在捕获的 MiniMax-H3 真实层上，K 的量化 MSE 降低 21.65%，attention 输出 MSE 降低 8.18%。

![MiniMax-H3 FP8 smoothing 性能](assets/smoothing-performance.svg)

| 配置（BF16 为 reference） | 视频 PSNR ↑ | 视频 SSIM ↑ | 音频 cosine ↑ | 频谱收敛误差 ↓ |
|---|---:|---:|---:|---:|
| FP8，不使用 smoothing | **19.63 dB** | **0.6712** | 0.8504 | 0.5055 |
| FP8，开启 smoothing | 19.27 dB | 0.6700 | **0.8891** | **0.4537** |

**Prompt:** `Locked-off cinematic wide shot of a red vintage tram gliding slowly through a snowy alpine village at sunrise. The tram remains rigid and geometrically consistent, its windows and wheels stay aligned. Light snow falls; soft rail sounds and distant church bells are synchronized with the scene. No people, no cuts, no camera movement.`

<div class="video-grid video-grid-three" data-sync-group="smoothing">
  <figure><figcaption>BF16 reference</figcaption><video controls playsinline preload="metadata" data-result-slot="bf16-quality"></video></figure>
  <figure><figcaption>FP8，不使用 smoothing</figcaption><video controls playsinline preload="metadata" data-result-slot="fp8-unsmoothed"></video></figure>
  <figure><figcaption>FP8，开启 smoothing</figcaption><video controls playsinline preload="metadata" data-result-slot="fp8-smoothed"></video></figure>
</div>

在视频后半段，smoothing 提高了车顶标识、受电弓连线和窗框对齐的时序稳定性；音频 cosine 从 0.850 提升至 0.889，频谱收敛误差从 0.506 降至 0.454。
