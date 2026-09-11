<!--
SECTION-CONTRACT
id: 03-quality-and-scale
incoming_premise: The shared FP8 sparse path is faster but perturbs a recurrent denoising trajectory.
outgoing_question: What performance and quality does the complete path deliver?
evidence: PR40 quality suite, PR37 overlap work, and distributed benchmark
do_not_claim: One prompt or one full-reference metric proves perceptual equivalence.
-->

# Preserving Quality While Scaling

Diffusion models reuse each prediction in the next denoising update. Small
attention errors can accumulate into visible changes in texture, motion, or
audio-video alignment. Combining FP8 rounding with sparse routing makes this
quality problem more important, not less.

## Attention smoothing

Profiles from real MiniMax-H3 layers showed that K and V are not always
centered around zero. Symmetric FP8 represents zero directly and scales around
it, so a large mean consumes dynamic range that could otherwise represent the
variation attention needs.

TeleFuser uses two attention identities to remove that offset before FP8
quantization without changing the corresponding higher-precision operation:

~~~python
K_centered = K - K.mean(dim=sequence)
softmax(Q @ K_centered.T) == softmax(Q @ K.T)

V_mean = V.mean(dim=sequence)
V_centered = V - V_mean
attention(Q, K, V) == attention(Q, K, V_centered) + V_mean
~~~

K can be centered because softmax cancels a constant shift in each query row.
V can be centered because attention probabilities sum to one; its mean is
restored at the output. TeleFuser also corrects the small residual mean caused
by FP8 rounding. KV smoothing and V correction are enabled by default for the
optimized MiniMax-H3 path.

The mechanism was checked on a captured MiniMax-H3 attention layer with a
32,626-token context. Centering reduced K quantization MSE by **21.65%** and
attention-output MSE by **8.18%**, while output SQNR improved from 29.23 to
29.60 dB.

The first smoothing implementation added separate tensor passes and increased
denoising time by 11.7%. Integrating the work into the existing attention path
reduced the final overhead to 2.2%. The smoothed path still delivered 39.4%
higher denoising throughput than BF16 Linear + FlashAttention 4 in the 50-step
quality experiment, with 42.6% lower peak allocated memory.

![Performance cost and retained speedup of attention smoothing](assets/smoothing-performance.png)

For the same prompt and seed, smoothing improved all reported video trajectory
metrics: frame cosine rose from 0.87488 to 0.87729, PSNR from 14.695 to 14.800
dB, and mean SSIM from 0.5464 to 0.5659. The measured audio distances were
mixed, so the demonstrated quality gain is strongest on the video trajectory.
The synchronized outputs below make the visual and audio differences directly
inspectable.

<div class="video-grid video-grid-three" data-sync-group="smoothing">
  <figure>
    <figcaption>BF16 Linear + FlashAttention 4</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="bf16-quality"></video>
  </figure>
  <figure>
    <figcaption>FP8 Linear + FP8 Sol, unsmoothed</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="fp8-unsmoothed"></video>
  </figure>
  <figure>
    <figcaption>FP8 Linear + FP8 Sol, KV smoothing + V correction</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="fp8-smoothed"></video>
  </figure>
</div>

## Quality-aware sparsity

Smoothing improves low-precision representation; sparse attention still needs
to decide where full attention matters. Early denoising updates establish
global structure, and some layers are more sensitive to missing long-range
interactions. TeleFuser therefore supports a dense opening window and selected
dense layers before Sol-Attn handles the remaining work. The evaluated FastH3
profile uses two dense opening updates and two dense layers, followed by
`tau=1.0` exact sparse routing.

This policy keeps the most quality-sensitive regions dense while retaining the
benefit of sparse attention through most of the DiT execution. Dense steps,
dense layers, threshold mode, and `tau` remain configurable for other
MiniMax-H3 schedules.

## The same semantics on multiple GPUs

Sequence parallelism changes which part of attention each rank owns. TeleFuser
computes smoothing statistics after Ulysses has established the full-sequence,
local-head view, so distributed execution preserves the single-GPU attention
semantics. On four H100s, tensor parallelism can split the wide transformer
work while Ulysses splits the long sequence.

As FP8 and sparsity reduce arithmetic time, communication becomes more visible.
TeleFuser therefore supports overlapping Ulysses communication with attention
compute. This turns multi-GPU execution into part of the optimization rather
than a scaling wrapper around it.
