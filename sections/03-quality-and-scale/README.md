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

The quality case uses a locked camera on a tram moving through snow. The rigid
body, aligned windows, rails, and pantograph make temporal geometry directly
visible throughout the clip.

| Model | Resolution and frames | Sampling | GPUs | Case / seed |
|---|---|---|---:|---|
| MiniMax-H3 Base, T2VA | 1344 × 768, 107 frames, 4 s at 24 FPS | 50 points / 49 DiT updates | 1 × H100 | snow tram / 17 |

![MiniMax-H3 FP8 smoothing performance on one GPU](assets/smoothing-performance.svg)

Smoothed FP8 raises denoise throughput by 37.2% over BF16 Linear +
FlashAttention 4 and reduces peak allocated memory by 42.6%. The fused
correction adds 2.1% denoise time over raw FP8.

| Configuration (BF16 reference) | Video PSNR ↑ | Video SSIM ↑ | Audio cosine ↑ | Spectral convergence error ↓ |
|---|---:|---:|---:|---:|
| FP8, unsmoothed | **19.63 dB** | **0.6712** | 0.8504 | 0.5055 |
| FP8, smoothing enabled | 19.27 dB | 0.6700 | **0.8891** | **0.4537** |

<div class="video-grid video-grid-three" data-sync-group="smoothing">
  <figure>
    <figcaption>BF16 reference</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="bf16-quality"></video>
  </figure>
  <figure>
    <figcaption>FP8, unsmoothed</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="fp8-unsmoothed"></video>
  </figure>
  <figure>
    <figcaption>FP8, smoothing enabled</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="fp8-smoothed"></video>
  </figure>
</div>

In the latter part of the clip, the unsmoothed output shows less stable roof
markings, overhead linkage, and window alignment than the smoothed output.

## Tuned defaults with configurable sparsity

Models and generation tasks differ in which attention regions are sensitive to
sparsity. TeleFuser exposes the dense window, dense layers, threshold mode, and
sparsity strength for further tuning on new workloads. Its MiniMax-H3 defaults
have already been validated for performance and output quality, so users do not
need to select these parameters manually.

## The same path on multiple GPUs

TeleFuser specializes its SP kernels for FP8, Sol-Attn, and world-model video
generation. Each GPU quantizes FP8 QKV and runs Sol-Attn on its local attention
layout after Ulysses All-to-All. Three-dimensional video-token reordering runs
before sequence partitioning; scalar timesteps remain replicated, while
per-token timesteps are sharded with the video tokens. Ulysses communication is
also overlapped with attention compute to reduce multi-GPU overhead.
