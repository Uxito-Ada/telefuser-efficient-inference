<!--
SECTION-CONTRACT
id: 04-evaluation
incoming_premise: TeleFuser combines precision, sparsity, smoothing, adapters, and distributed execution.
outgoing_question: What does this add to TeleFuser as a world-model runtime?
evidence: normalized records under experiments/h100-4gpu-e2e/raw and experiments/adapter-suite/raw
do_not_claim: Do not combine incomparable schedules or present invalid media as measurements.
-->

# Performance and Output Quality {#results}

## Evaluation matrix

| Experiment | Model and task | Output | Sampling | GPUs | Topology |
|---|---|---|---|---:|---|
| TeleFuser scaling | Base H3, T2VA | 1344 × 768, 124 frames, 5 s at 24 FPS | 50 denoising steps | 1 / 2 / 4 | local / TP2 / TP2 × Ulysses SP2 |
| Framework comparison | Base H3, T2VA | 1344 × 768, 124 frames, 5 s at 24 FPS | 50 denoising steps | 4 | framework-native distributed path |
| Turbo LoRA | MiniMax-H3 Turbo, T2VA | 1344 × 768, 124 frames, 5 s at 24 FPS | 8 denoising steps | 4 | TP2 × Ulysses SP2 |
| FastH3 adapter | FastH3 dense, T2VA | 1344 × 768, 124 frames, 5 s at 24 FPS | 4 denoising steps | 4 | framework-native distributed path |
| FP8 smoothing | Base H3, T2VA | 1344 × 768, 107 frames, 4 s at 24 FPS | 50 denoising steps | 1 | local |

## Unified four-GPU comparison

Peak GPU memory is shown as bars; denoising throughput is shown as the line.
Every point uses four H100 GPUs and disables CPU offload.

![MiniMax-H3 Base and adapter performance comparison](assets/all-workloads-performance.svg)

Base H3 uses the official [FastVideo example](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_minimax_h3_t2v.py) and [SGLang cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx), together with the corresponding LightX2V and TeleFuser examples. Turbo LoRA is compared across TeleFuser, SGLang, and LightX2V; FastH3 is compared across TeleFuser and FastVideo.

- **Base H3:** TeleFuser delivers 163.5% higher end-to-end throughput than LightX2V, 119.4% higher than FastVideo, and 51.8% higher than SGLang. Peak memory is 40.3% lower than LightX2V and 37.3% lower than SGLang, while remaining within 1.5% of FastVideo.
- **Turbo LoRA:** TeleFuser throughput is 58.8% higher than LightX2V and 0.3% higher than SGLang. Peak memory is 42.3% lower than LightX2V and 26.3% lower than SGLang.
- **FastH3:** TeleFuser delivers 208.5% higher end-to-end throughput than FastVideo while using 37.4% less peak GPU memory.

## TeleFuser scaling

![TeleFuser Base H3 scaling](assets/base-scaling.svg)

Across the tested one-, two-, and four-GPU points, the quantized distributed
path preserves near-linear scaling while lowering per-GPU memory relative to
dense execution. The resulting headroom supports larger video-generation
requests without changing the model or output contract.

## Generated output

| Framework | Base H3 | MiniMax-H3 Turbo LoRA | FastH3 Preview adapter |
|---|---|---|---|
| TeleFuser | ✅ Supported | ✅ Supported | ✅ Supported |
| LightX2V | ✅ Supported | ✅ Supported | ❌ Unsupported |
| FastVideo | ✅ Supported | ❌ Unsupported | ✅ Supported |
| SGLang | ✅ Supported | ✅ Supported | ❌ Unsupported |

**Base H3 and Turbo LoRA prompt:** `Steam rises from the ramen while the family talks in the background.`

**FastH3 prompt:** `Steam rises from the ramen while the family talks in the background.`

<div class="video-matrix">
  <div></div>
  <div class="video-matrix-heading">Base H3</div>
  <div class="video-matrix-heading">Turbo LoRA</div>
  <div class="video-matrix-heading">FastH3</div>

  <div class="video-matrix-label">LightX2V</div>
  <figure><video controls playsinline preload="metadata" data-result-slot="lightx2v-base-h3"></video></figure>
  <figure><video controls playsinline preload="metadata" data-result-slot="turbo-lightx2v"></video></figure>
  <div class="video-matrix-empty">Not supported</div>

  <div class="video-matrix-label">FastVideo</div>
  <figure><video controls playsinline preload="metadata" data-result-slot="fastvideo-base-h3"></video></figure>
  <div class="video-matrix-empty">Not supported</div>
  <figure><video controls playsinline preload="metadata" data-result-slot="fastvideo-primary"></video></figure>

  <div class="video-matrix-label">SGLang</div>
  <figure><video controls playsinline preload="metadata" data-result-slot="sglang-base-h3"></video></figure>
  <figure><video controls playsinline preload="metadata" data-result-slot="turbo-sglang"></video></figure>
  <div class="video-matrix-empty">Not supported</div>

  <div class="video-matrix-label">TeleFuser</div>
  <figure><video controls playsinline preload="metadata" data-result-slot="telefuser-base-h3"></video></figure>
  <figure><video controls playsinline preload="metadata" data-result-slot="turbo-telefuser"></video></figure>
  <figure><video controls playsinline preload="metadata" data-result-slot="telefuser-primary"></video></figure>
</div>

## FP8 attention smoothing

The first unfused implementation added 11.7% denoising overhead; fusion reduced
the final cost to 2.2%. On a captured MiniMax-H3 layer, K quantization MSE fell
by 21.65% and attention-output MSE by 8.18%.

![MiniMax-H3 FP8 smoothing performance](assets/smoothing-performance.svg)

| Configuration (BF16 reference) | Video PSNR ↑ | Video SSIM ↑ | Audio cosine ↑ | Spectral convergence error ↓ |
|---|---:|---:|---:|---:|
| FP8, unsmoothed | **19.63 dB** | **0.6712** | 0.8504 | 0.5055 |
| FP8, smoothing enabled | 19.27 dB | 0.6700 | **0.8891** | **0.4537** |

**Prompt:** `Locked-off cinematic wide shot of a red vintage tram gliding slowly through a snowy alpine village at sunrise. The tram remains rigid and geometrically consistent, its windows and wheels stay aligned. Light snow falls; soft rail sounds and distant church bells are synchronized with the scene. No people, no cuts, no camera movement.`

<div class="video-grid video-grid-three" data-sync-group="smoothing">
  <figure><figcaption>BF16 reference</figcaption><video controls playsinline preload="metadata" data-result-slot="bf16-quality"></video></figure>
  <figure><figcaption>FP8, unsmoothed</figcaption><video controls playsinline preload="metadata" data-result-slot="fp8-unsmoothed"></video></figure>
  <figure><figcaption>FP8, smoothing enabled</figcaption><video controls playsinline preload="metadata" data-result-slot="fp8-smoothed"></video></figure>
</div>

In the latter part of the clip, smoothing improves the temporal stability of
the tram's roof markings, overhead linkage, and window alignment. It also raises
audio cosine similarity from 0.850 to 0.889 and reduces spectral convergence
error from 0.506 to 0.454.
