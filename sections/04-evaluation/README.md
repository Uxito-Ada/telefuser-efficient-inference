<!--
SECTION-CONTRACT
id: 04-evaluation
incoming_premise: TeleFuser combines precision, sparsity, smoothing, adapters, and distributed execution.
outgoing_question: What does this add to TeleFuser as a world-model runtime?
evidence: normalized records under experiments/h100-4gpu-e2e/raw and PR37
do_not_claim: Do not combine incomparable E2E scopes or present excluded runs as measurements.
-->

# Results on H100

The primary comparison uses FastVideo's MiniMax-H3 FastH3 Dense/Data-Free
adapter path as the external baseline. Both systems run on one H100 80GB with
the same base checkpoint, adapter strength, prompt, seed, 1344 x 768 output,
124 frames at 24 FPS, and a five-point schedule with four DiT evaluations.
Each result is the median after one warm-up and three measured generations.

FastVideo uses BF16 Linear + FlashAttention 4. TeleFuser uses FP8 Linear + FP8
Sol-Attn with `tau=1.0`, two dense opening updates, two dense layers, KV
smoothing, and V correction. Neither path offloads the DiT during denoising.

![Matched MiniMax-H3 adapter performance](assets/end-to-end.svg)

<div class="result-summary">
  <div><strong>25.7% lower</strong><span>denoise time than FastVideo BF16 + FA4</span></div>
  <div><strong>34.6% higher</strong><span>actual DiT-forward throughput</span></div>
  <div><strong>14.3% lower</strong><span>whole-process peak GPU memory</span></div>
</div>

Median denoising time falls from 21.685 to 16.116 seconds. With four actual DiT
evaluations per video, throughput rises from 0.184 to 0.248 forwards per second.
Whole-process peak memory falls from 79.03 to 67.70 GiB. The memory reduction is
smaller than the FP8 weight ratio because text and media encoders, non-Linear
weights, activations, and workspaces remain part of the process peak.

The chart compares the matched denoising region rather than whole-request
latency because the recorded frameworks used different prompt-conditioning
cache policies.

## Generated output

Both outputs contain 124 frames of 1344 x 768 H.264 video and 32kHz stereo AAC
audio. The HTML players are synchronized for direct comparison.

<div class="video-pair" data-sync-group="primary">
  <figure>
    <figcaption>FastVideo: BF16 Linear + FlashAttention 4</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="fastvideo-primary"></video>
  </figure>
  <figure>
    <figcaption>TeleFuser: FP8 Linear + smoothed FP8 Sol-Attn</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="telefuser-primary"></video>
  </figure>
</div>

The earlier 50-step ablation provides the controlled quality comparison among
BF16, unsmoothed FP8, and smoothed FP8. The FastH3 pair above shows the final
adapter-enabled path producing complete video and audio under the performance
configuration.

## Multi-GPU execution

A separate Base H3 validation exercises TeleFuser's resident
`TP2 x Ulysses SP2` topology on four H100s. Across five prompt/seed cases,
communication-compute overlap reduced mean request wall time from 78.546 to
75.766 seconds, a **3.539%** improvement, while all five generated MP4 files
remained byte-identical.

This result is shown separately because it uses the 50-point Base H3 schedule,
whereas the external comparison uses the five-point FastH3 adapter. Together,
the two experiments cover the optimized single-GPU path and TeleFuser's
distributed execution capability.
