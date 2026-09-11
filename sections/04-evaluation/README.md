<!--
SECTION-CONTRACT
id: 04-evaluation
incoming_premise: The final path combines precision, sparsity, quality controls, adapters, and distributed execution.
outgoing_question: What can the result teach beyond these measurements?
evidence: normalized records under experiments/h100-4gpu-e2e/raw and PR37
do_not_claim: Do not combine incomparable E2E scopes or present excluded runs as measurements.
-->

# What the Complete Path Actually Buys

An external baseline is more useful than comparing TeleFuser with another
TeleFuser flag. We therefore use FastVideo's MiniMax-H3 FastH3 Dense/Data-Free
adapter path as the primary reference. Both outputs use one H100 80GB, the
same base checkpoint and adapter at strength 1.0, the same fox prompt and seed
1000, 1344 x 768 resolution, 124 frames at 24 FPS, and five scheduler sigma
points, which execute four DiT forwards. Each profile ran one full warm-up and
three measured generations.

FastVideo uses BF16 Linear and FlashAttention 4. Its H100 checkout discards the
merged LoRA state after the numerical merge so that the 80GB device can run the
official dense recipe; it does not change model weights or attention math.
TeleFuser uses tf-kernel FP8 Linear and FP8 Sol-Attn with `tau=1.0`, exact
routing, two dense opening updates and two dense layers, KV smoothing, and V
bias correction. Neither DiT is CPU-offloaded during the measured denoising
stage.

![Matched MiniMax-H3 adapter performance](assets/end-to-end.svg)

<div class="result-summary">
  <div><strong>25.7% lower</strong><span>denoise time than FastVideo BF16 + FA4</span></div>
  <div><strong>34.6% higher</strong><span>actual DiT-forward throughput</span></div>
  <div><strong>14.3% lower</strong><span>whole-process peak GPU memory</span></div>
</div>

The first result is the central systems claim. Moving both the dense DiT work
and sparse attention boundary to FP8 reduces median denoising time from 21.685
to 16.116 seconds. Because the distilled schedule performs four real DiT
forwards, throughput is reported as `4 / denoise_seconds`, not as five
"steps" divided by time. The five sigma points describe the schedule grid;
there are only four model evaluations.

Peak memory falls less than the two-to-one ratio suggested by FP8 parameter
size. Only the selected DiT Linear weights and QKV attention operands are FP8.
The text encoder, video and audio VAEs, non-Linear weights, live activations,
communication buffers, kernel workspaces, and allocator state do not all
halve. The measured 14.3% reduction is the end-to-end process peak, not a
weight-file estimate.

We deliberately do not publish an E2E speedup percentage for this pair. The
FastVideo warm-request path reused conditioning for the repeated prompt, while
the measured TeleFuser path encoded it on every request. Their MP4-close wall
times therefore answer different serving-cache questions even though their
denoise windows are matched. Reporting the ratio as a framework speedup would
reward a cache policy rather than the FP8 sparse execution path.

## Inspect the artifacts, not just the bars

Both selected files decode to 1344 x 768 H.264 video with 32kHz stereo AAC and
have the same 5.175-second container duration. They come from the same prompt,
seed, checkpoint, adapter, and schedule. The players are synchronized in the
HTML build so motion and audio events can be compared directly.

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

These videos are evidence of a valid generation path, not a claim of
pixel-level identity. The earlier 50-step ablation is the controlled quality
experiment because it compares BF16, unsmoothed FP8, and smoothed FP8 under one
runtime. Cross-framework media differences can also come from preprocessing,
operator ordering, and decoder details, so we keep them visible rather than
compressing them into one misleading similarity score.

## Where multi-GPU changes the answer

The single-H100 comparison isolates the final low-precision sparse path. The
distributed schedule was validated separately on four H100s using MiniMax-H3's
resident `TP2 x Ulysses SP2` profile. In the five prompt/seed cases published
with the communication-compute overlap work, the optimized path reduced mean
request wall time from 78.546 to 75.766 seconds, a 3.539% improvement, while
all five synchronized MP4 files remained byte-identical. The benefit is
modest but important: after reducing arithmetic, exposed all-to-all and layout
work become a larger fraction of latency, so lossless scheduling work still
matters.

We did not merge that four-GPU result into the external-baseline chart. It uses
a 50-point Base H3 protocol rather than the five-point FastH3 adapter protocol,
and it measures before/after communication scheduling inside TeleFuser. It is
evidence that Ulysses is exercised and its optimization preserves output, not
a substitute for a matched external comparison.

During this article refresh, only two unoccupied H100s were available. Fresh
two-GPU attempts were excluded: clean FastVideo exceeded 79GB during full VAE
decode, its official lazy-module LoRA route recursively materialized a second
FSDP transformer, and TeleFuser's resident two-GPU stage topology exceeded HBM
when the denoise worker onloaded its model. None of those partial runs appears
in the chart. Raw successful records and the exclusion log remain in the
[experiment directory](../../experiments/h100-4gpu-e2e/README.md).
