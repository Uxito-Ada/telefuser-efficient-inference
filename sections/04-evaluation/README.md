<!--
SECTION-CONTRACT
id: 04-evaluation
incoming_premise: TeleFuser combines precision, sparsity, smoothing, adapters, and distributed execution.
outgoing_question: What does this add to TeleFuser as a world-model runtime?
evidence: normalized records under experiments/h100-4gpu-e2e/raw and experiments/adapter-suite/raw
do_not_claim: Do not combine incomparable schedules or present invalid media as measurements.
-->

# Performance and Output Quality {#results}

## Base H3 on four GPUs

The primary framework comparison runs MiniMax-H3 Base on four GPUs.
LightX2V and TeleFuser both use `TP2 x Ulysses SP2`, the same prompt and seed,
1344 x 768 output, 124 frames at 24 FPS, and the same 50-point schedule. Feature
cache is disabled. Each framework uses its validated optimized profile.

![Four-GPU MiniMax-H3 Base performance](assets/lightx2v-base-h3.svg)

<div class="result-summary">
  <div><strong>2.64x faster</strong><span>generation than LightX2V</span></div>
  <div><strong>2.62x faster</strong><span>denoising, or 162.2% higher step throughput</span></div>
  <div><strong>40.3% lower</strong><span>representative peak GPU memory</span></div>
</div>

Generation time falls from 137.76 to 52.27 seconds, while denoising falls from
129.22 to 49.28 seconds. This is the main end-to-end result: the external
baseline and TeleFuser use the same four-GPU parallel topology rather than
comparing a distributed path with a single-GPU run.

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

Both files contain a coherent 124-frame ramen scene with synchronized stereo
audio. The performance record uses the matched prompt and seed described above;
the displayed TeleFuser sample uses a separate, camera-stable presentation
prompt and is included for qualitative inspection rather than a paired quality
score. The benchmark uses one warm-up and one measured request; the memory
figure is the representative peak across otherwise idle GPUs because GPU 0 had
a fixed unrelated allocation during both runs.

## Adapter workloads

TeleFuser supports both MiniMax-H3 Turbo LoRA and FastH3's dense hybrid
adapter. Each is compared with the external framework that provides a working
reference path for that adapter, using matched model inputs, output shape, and
DiT work.

### MiniMax-H3 Turbo LoRA

The Turbo comparison uses one GPU, the 8-step v1.0 768p adapter, eight DiT
updates, and resident DiT weights in both frameworks. TeleFuser merges the LoRA
before low-precision execution.
CPU block-offload measurements are not included in the chart.

![MiniMax-H3 Turbo adapter performance](assets/turbo-performance.svg)

<div class="result-summary">
  <div><strong>26.7% lower</strong><span>denoise time than LightX2V</span></div>
  <div><strong>36.5% higher</strong><span>8-step denoise throughput</span></div>
  <div><strong>12.3% lower</strong><span>whole-process peak GPU memory</span></div>
</div>

<div class="video-pair" data-sync-group="turbo">
  <figure>
    <figcaption>LightX2V: MiniMax-H3 Turbo</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="turbo-lightx2v"></video>
  </figure>
  <figure>
    <figcaption>TeleFuser: MiniMax-H3 Turbo</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="turbo-telefuser"></video>
  </figure>
</div>

The performance workload remains prompt-matched. The displayed TeleFuser Turbo
sample uses a camera-stable presentation prompt: a level tripod shot, a subtle
push-in, and explicit suppression of orbit, roll, and spinning. This isolates
the adapter's subject motion without inviting unstable camera transforms.

### FastH3 dense adapter

For FastH3, both systems run one GPU with the same dense adapter, prompt,
seed, 1344 x 768 output, 124 frames, and four actual DiT evaluations. Each
framework uses its validated optimized profile, and neither path offloads the
DiT during denoising. Results are the median after one warm-up and three
measured generations.

![Matched FastH3 adapter performance](assets/end-to-end.svg)

<div class="result-summary">
  <div><strong>25.7% lower</strong><span>denoise time than FastVideo</span></div>
  <div><strong>34.6% higher</strong><span>actual DiT-forward throughput</span></div>
  <div><strong>14.3% lower</strong><span>whole-process peak GPU memory</span></div>
</div>

<div class="video-pair" data-sync-group="fasth3">
  <figure>
    <figcaption>FastVideo: FastH3</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="fastvideo-primary"></video>
  </figure>
  <figure>
    <figcaption>TeleFuser: FastH3</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="telefuser-primary"></video>
  </figure>
</div>

The chart compares the matched denoising region rather than whole-request
latency because the recorded frameworks used different prompt-conditioning
cache policies. All four adapter outputs above contain 124 H.264 frames at
1344 x 768 and 32kHz stereo AAC audio.

## Communication overlap

The four-GPU Base path also benefits from TeleFuser's communication-compute
overlap. In a separate five-case regression, it reduced mean request wall time
from 78.546 to 75.766 seconds, a **3.539%** improvement, while every generated
MP4 remained byte-identical. This optimization is part of the same distributed
execution path used by the main Base H3 result.
