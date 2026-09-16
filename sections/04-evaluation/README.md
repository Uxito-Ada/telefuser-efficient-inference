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

Unless noted otherwise, each run uses MiniMax-H3's 768p profile and writes 24 FPS H.264 video with 32 kHz stereo AAC audio.

| Experiment | Model and task | Output | Sampling | GPUs | Topology |
|---|---|---|---|---:|---|
| TeleFuser scaling | Base H3, T2VA | 1344 × 768, 124 frames, 5 s | 50 points / 49 DiT updates | 1 / 2 / 4 | local / TP2 / TP2 × Ulysses SP2 |
| Four-GPU frameworks | Base H3, T2VA | 1344 × 768, 124 frames, 5 s | 50 points / 49 DiT updates | 4 | TP2 × Ulysses SP2; FastVideo SP4 |
| FP8 smoothing | Base H3, T2VA | 1344 × 768, 107 frames, 4 s | 50 points / 49 DiT updates | 1 | local |
| Turbo LoRA | MiniMax-H3 Turbo, I2AV | 1344 × 768, 124 frames, 5 s | 9 points / 8 DiT updates | 1 | local |
| FastH3 adapter | FastH3 dense, T2VA | 1344 × 768, 124 frames, 5 s | 5 points / 4 DiT updates | 1 | local |

## Unified performance comparison
The figure uses denoising throughput as the speed metric and peak GPU memory as the capacity metric. Base H3 uses four GPUs; adapter points are labelled with their measured GPU count.

![MiniMax-H3 Base and adapter performance comparison](assets/all-workloads-performance.svg)

The Base H3 points use the official [FastVideo example](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/basic/basic_minimax_h3_t2v.py), [SGLang cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx), and matched LightX2V/TeleFuser configurations. TeleFuser reaches 1.015 denoising steps/s on Base H3, compared with 0.387 for LightX2V and 0.496 for FastVideo. The Turbo point reaches 0.306 steps/s after the adapter reduces the schedule to eight DiT updates.
**Prompt:** `Steam rises from the ramen while the family talks in the background.`

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

## Adapter workloads

Turbo LoRA and FastH3 use different weights and sampling contracts. Current framework support is:

| Framework | MiniMax-H3 Turbo LoRA | FastH3 Preview adapter |
|---|---|---|
| TeleFuser | ✅ Supported | ✅ Supported |
| LightX2V | ✅ Supported | ❌ Unsupported |
| FastVideo | ❌ Unsupported | ✅ Supported |
| SGLang | ✅ Supported | ❌ Unsupported |

The Turbo LoRA comparison covers TeleFuser and LightX2V; the FastH3 comparison covers TeleFuser and FastVideo.

### MiniMax-H3 Turbo LoRA

Both frameworks use the 8-step v1.0 768p adapter with resident DiT weights. TeleFuser merges the LoRA before creating FP8 weights. CPU block offload is excluded.



**Prompt:** `Steam rises from the ramen while the family talks in the background. Bright, warm indoor lighting illuminates every face and the room with natural skin tones. The man holds a pair of straight, rigid chopsticks that remain perfectly straight throughout the video and never bend.`

<div class="video-grid video-grid-two" data-sync-group="turbo">
  <figure>
    <figcaption>LightX2V: MiniMax-H3 Turbo</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="turbo-lightx2v"></video>
  </figure>
  <figure>
    <figcaption>TeleFuser: MiniMax-H3 Turbo</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="turbo-telefuser"></video>
  </figure>
</div>

### FastH3 dense adapter



**Prompt:** `integrated_multimodal_description: A red fox runs through fresh snow at dawn. overall_soundscape: Fast pawsteps in snow, winter wind, and distant birds.`

<div class="video-grid video-grid-two" data-sync-group="fasth3">
  <figure>
    <figcaption>FastVideo: FastH3</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="fastvideo-primary"></video>
  </figure>
  <figure>
    <figcaption>TeleFuser: FastH3</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="telefuser-primary"></video>
  </figure>
</div>
