<!--
SECTION-CONTRACT
id: 07-evaluation
incoming_premise: The complete method now includes precision, sparsity, quality control, SP4, and adapter-correct weights.
outgoing_question: Which conclusions generalize beyond the measured H100/H3 contract?
evidence: Only new raw records under experiments/.
do_not_claim: Do not use failed, invalid, unmatched, single-GPU, or B200 runs in the headline comparison.
-->

# 7. Evaluating the Complete System

The preceding sections made one optimization path progressively executable.
Evaluating each implementation step on a different workload would not tell us
whether the final system is useful. This section therefore asks one primary
question:

> On the same four H100 GPUs, can the complete TeleFuser FP8 sparse path execute
> a matched MiniMax-H3 FastH3 request faster than a maintained external
> implementation while preserving acceptable video and audio quality?

The comparison uses an external baseline because an internal BF16 mode would
only isolate an ablation. The production question is whether the resulting
system is competitive with another optimized H3 runtime.

## Systems under test

**Primary baseline: FastVideo Dense/Data-Free, BF16 + FA4.** FastVideo maintains
the official FastH3 CUDA recipe and adapter path. The dense Data-Free adapter
provides a strict comparison because both frameworks can load the same
MiniMax-H3 base and the same released adapter. FA4 is the baseline's intended
H100 dense-attention backend. The run uses the maintained example with
configuration-only workload changes.

**Our system: TeleFuser FP8 Linear + FP8 Sol + smoothing + Ulysses SP4.** This
is the complete path described in the article: merged Dense/Data-Free adapter,
cached E4M3 Linear weights, shared QKV activation quantization, post-Ulysses
attention preparation, dynamic Sol routing, dense quality islands, K/V
smoothing, and V-bias correction.

**Related sparse reference: FastVideo VSA.** The trained VSA route is admitted
only if the maintained H100 path runs without source changes and produces valid
media at the matched output specification. Its result is contextual, not the
denominator of the headline speedup, because VSA contains learned gates and
therefore represents a different effective model.

**Official quality reference: MiniMax-H3 Diffusers.** The publisher-supported
BF16 route establishes expected output behavior. It is excluded from the SP4
performance chart unless it exposes a comparable four-GPU execution contract.

**Conditional LightX2V reference.** The published MiniMax-H3 Sol example is
included only if a clean official-environment reproduction produces meaningful
video and audio and reaches the matched workload through configuration alone.
An invalid output has no meaningful throughput and is recorded under excluded
results.

## Matched workload

| Dimension | Contract |
|---|---|
| Hardware | 4 x NVIDIA H100 80GB |
| Model | MiniMax-H3 |
| Adapter | FastH3 Dense/Data-Free, scale 1.0 |
| Task | Text-to-video-and-audio |
| Output | 1344 x 768, 124 frames, 24 FPS, 5 seconds |
| Sampling work | Five sigma points, four actual DiT forwards |
| Batch / concurrency | 1 / 1 |
| Prompt | One fixed benchmark prompt, stored verbatim in the experiment config |
| Seed | Fixed and recorded |
| Parallelism | Four GPUs for every headline performance row |
| Warm-up | One full generation including decode and file output |
| Repeats | Five measured requests; report median and full samples |
| E2E boundary | Prompt processing through synchronized MP4 close |

The primary prompt is held constant for timing so text length and conditioning
do not become uncontrolled variables. A separate prompt suite measures quality
coverage. Model loading, adapter merging, and first-time kernel compilation are
reported separately from warm serving latency; they are not hidden inside one
framework's result and excluded from another's.

## Metrics

End-to-end latency measures the user-visible request. Denoising time is retained
to explain where the speedup comes from but is not the headline denominator.
Throughput is reported as completed five-second videos per hour at concurrency
one:

\[
\text{videos/hour}=\frac{3600}{\operatorname{median}(t_{\mathrm{E2E}})}.
\]

We also report actual DiT forwards per second rather than calling five sigma
points five steps of transformer work. GPU memory is sampled through NVML at
100 ms intervals during every formal request. Both the largest per-GPU peak and
the aggregate four-GPU peak are retained. Framework-specific allocator metrics
may diagnose a phase but do not replace the common NVML boundary.

## End-to-end result

The following table is deliberately incomplete until all systems pass the
protocol and media-validity gate.

| System | 4-GPU mode | Median E2E | Videos/hour | DiT forwards/s | Max per-GPU memory | Aggregate peak |
|---|---|---:|---:|---:|---:|---:|
| FastVideo BF16 + FA4 | official distributed recipe | TBD | TBD | TBD | TBD | TBD |
| TeleFuser FP8 + Sol | Ulysses SP4 | TBD | TBD | TBD | TBD | TBD |
| FastVideo VSA | official H100 route, if valid | TBD | TBD | TBD | TBD | TBD |

The final prose will be generated from raw JSON:

> Relative to the matched FastVideo baseline, TeleFuser reduces median
> end-to-end latency by **TBD%**, increases completed-video throughput by
> **TBD%**, and changes maximum per-GPU peak memory by **TBD%**. Denoising
> accounts for **TBD%** of the end-to-end improvement.

One figure will show the same experiment with narrow grouped bars for E2E
latency, videos/hour, and per-GPU memory. It will not combine unrelated units on
one axis or import historical measurements.

<!-- RESULT_FIGURE_TBD: experiments/h100-sp4-e2e/figures/end-to-end.svg -->

## Quality suite

Speed is accepted only after the final configuration passes a prompt suite
chosen before generation. The suite covers:

| Category | Failure being tested |
|---|---|
| Human face and hands | identity drift and local structure |
| Fast subject motion | temporal breakup and route instability |
| Camera motion | global geometry and background consistency |
| Multiple interacting objects | counting and spatial relationships |
| Low light and high contrast | FP8 outliers and fine detail |
| Material and fluid motion | high-frequency temporal behavior |
| Speech or singing | mouth/audio correspondence |
| Impact and environmental sound | event timing and soundscape alignment |

Each prompt uses fixed seeds and generates paired external-BF16 and TeleFuser
outputs. Tensor MSE, cosine, and SQNR diagnose the quantized attention boundary.
LPIPS/SSIM and temporal features measure paired trajectory divergence.
Prompt-video and prompt-audio scores test semantic alignment. Codec validity,
frame count, duration, audio channels, black frames, and non-finite samples are
hard acceptance checks.

These metrics answer different questions. A diffusion sample may diverge from
the BF16 pixels while remaining perceptually valid, so paired similarity is not
presented as an absolute quality score. The actual media remains the primary
evidence.

## Direct video evidence

The final HTML article embeds original MP4 files. Players are presented in
matched pairs and synchronized by page JavaScript; no frame collage stands in
for motion or audio.

<div class="video-pair" data-sync-group="primary">
  <figure>
    <figcaption>FastVideo BF16 + FA4</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="fastvideo-primary"></video>
  </figure>
  <figure>
    <figcaption>TeleFuser FP8 + Sol + smoothing, SP4</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="telefuser-primary"></video>
  </figure>
</div>

<p><strong>Video result:</strong> TBD--new experiment required.</p>
<p><strong>Audio result:</strong> TBD--new experiment required.</p>

The article-level quality conclusion is filled only after every predefined
prompt has a valid result. A single attractive clip cannot establish stability.
Once the result is available, its interpretation must remain within the
hardware, model, adapter, and prompt-suite boundaries defined here.
