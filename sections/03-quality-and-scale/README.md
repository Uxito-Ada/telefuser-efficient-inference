<!--
SECTION-CONTRACT
id: 03-quality-and-scale
incoming_premise: The shared FP8 sparse path is faster but perturbs a recurrent denoising trajectory.
outgoing_question: What performance and quality does the complete path deliver?
evidence: PR40 quality suite, PR37 overlap work, and distributed benchmark
do_not_claim: One prompt or one full-reference metric proves perceptual equivalence.
-->

# Quality Controls and Multi-GPU Scaling

Diffusion models feed each prediction into the next denoising update, so FP8
and sparse attention must preserve a stable trajectory as well as reduce
compute. TeleFuser addresses this inside its FP8 path and in the policy that
selects sparse work.

## Quality-aware FP8 attention

Profiling real MiniMax-H3 layers revealed that K and V can have clear non-zero
mean distributions. That offset reduces the useful range of symmetric FP8.
TeleFuser handles it as an attention-specific form of asymmetric quantization:
K and V are centralized before FP8 compute, and the equivalent shift is
compensated in the attention output.

The centralization and correction were fused into the existing FP8 Sol-Attn
path so quality does not require a separate performance mode. The initial
unfused implementation added 11.7% denoising overhead; fusion reduced the final
cost to 2.2%. On a captured MiniMax-H3 layer, K quantization MSE fell by 21.65%
and attention-output MSE by 8.18%. KV smoothing and V correction are enabled by
default in the optimized profile.

In the 50-step quality experiment, the resulting FP8 path remained 39.4%
faster in denoising than BF16 Linear + FlashAttention 4 and used 42.6% less peak
allocated memory. Frame cosine, PSNR, and mean SSIM all improved over
unsmoothed FP8; audio-distance results were mixed. The synchronized outputs
below show the complete video and audio result.

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
    <figcaption>FP8 Linear + FP8 Sol, quality-aware FP8</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="fp8-smoothed"></video>
  </figure>
</div>

## Quality-aware sparsity

Some denoising updates and transformer layers are more sensitive to missing
long-range interactions. TeleFuser supports a dense opening window and selected
dense layers before Sol-Attn handles the remaining work. The evaluated FastH3
profile uses two dense opening updates and two dense layers, followed by
`tau=1.0` exact sparse routing. Dense steps, dense layers, threshold mode, and
`tau` remain configurable for other MiniMax-H3 schedules.

## The same path on multiple GPUs

TeleFuser combines Ulysses sequence parallelism with tensor parallelism for
long-sequence, wide-transformer execution. Quality statistics are derived from
the attention view established by Ulysses, keeping the single- and multi-GPU
paths consistent. The runtime also supports overlapping Ulysses communication
with attention compute, which becomes increasingly valuable after FP8 and
sparsity reduce arithmetic time.
