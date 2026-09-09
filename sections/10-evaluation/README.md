<!--
SECTION-CONTRACT
id: 10-evaluation
role: Define measurement semantics and synthesize results without combining unmatched workloads.
sources: All case-study metrics, PR descriptions, and media manifests.
edit_scope: Any new row must name its workload and metric boundary.
-->

# 10. Evaluation

The six pull requests were developed at different points in the stack and do
not share one universal benchmark. This section makes the comparison boundary
explicit before summarizing results.

## Measurement semantics

| Metric | Definition in this article |
|---|---|
| Denoising time | Time inside the iterative denoiser, excluding model load and MP4 serialization |
| Generation/E2E time | Pipeline time as defined by the local case; component reuse and file output are stated nearby |
| `step/s` | Configured scheduler points divided by denoising time |
| DiT updates/s | Actual transformer evaluations divided by denoising time |
| Peak allocated | `torch.cuda.max_memory_allocated()` within the measured phase |
| Sampled peak | Maximum NVML process allocation sampled during the formal request |
| Quality metric | Similarity to a same-seed reference trajectory, not an absolute human preference score |

For Base MiniMax-H3, 50 configured points correspond to 49 denoiser updates in
some schedulers; the historical PR charts report `50 / denoising_seconds`.
FastH3 similarly exposes five scheduler points while executing four transformer
updates. The article preserves each published convention and names actual
updates when comparing distilled workloads.

## Evidence matrix

| Case | Matched workload | Primary result | Quality evidence |
|---|---|---|---|
| PR #16 | Qwen-Image, 1328 x 1328, 16 steps, 1 x H100 | NF4: 47.4% less peak memory than BF16 | Five-profile image grid |
| PR #25 | H3 FL2VA, 768p, 5 s, 50 steps, 1 x H100 | tf-kernel FP8: 5.6% lower denoise time than TorchAO FP8 | Four synchronized video/audio outputs |
| PR #30 | H3, 1344 x 768, 124 frames, 50 steps, 1 x H100 | FP8 Sol: 65.0% higher throughput and 41.9% less peak memory than BF16 Dense | H3 and Wan 2 x 2 media ablations |
| PR #35 | H3, 4 x H100, TP2 x SP2, 50 points | 53.2% higher throughput than TeleFuser BF16 FA4; 2.62x LightX2V denoise speed | Three-framework output comparison |
| PR #40 | H3, 1344 x 768, 107 frames, 50 steps, 1 x H100 | 8.18% lower dense attention-output MSE for 2.11% throughput cost | Frame, audio, and tensor metrics |
| PR #44 Turbo | H3 Turbo, 124 frames, 8 updates, 1 x H100 | 36.5% higher throughput than resident LightX2V | Two framework outputs |
| PR #44 FastH3 | Dense adapter, 124 frames, 4 updates, 1 x H100 | 57.2% higher scheduler-point throughput than FastVideo | Two framework outputs |

The table is a map of local Pareto improvements, not a multiplication chain.
For example, the four-H100 result already includes FP8 Linear, FP8 attention,
sparsity, parallel communication, and model-specific guards.

## Performance protocol by case

**Warm versus cold.** PR #30 reports clean-process behavior including first
kernel/JIT execution. PR #35 and PR #44 perform a full warmup before formal
requests. PR #40 uses cold processes without warmup but repeats each FP8
profile. These policies answer different deployment questions and should not be
plotted on one axis.

**Memory.** Early cases use PyTorch peak allocated memory. Distributed and
cross-framework cases also sample process memory because each framework manages
component residency differently. PR #44 resets the peak only after adapter
merge, FP8 materialization, garbage collection, and CUDA cache clearing to
exclude temporary BF16-to-FP8 overlap.

**Cross-framework baselines.** External frameworks are run through official
examples with configuration-only workload changes wherever possible. If a
baseline cannot produce valid output or OOMs, the failure is reported and no
performance number is assigned. CPU-offload profiles are excluded from the
final PR #44 charts because PCIe transfer changes both the latency and memory
question.

## Quality protocol

Quality validation progresses from cheap to expensive:

1. **Kernel invariants:** shape, dtype, finite values, padding, tail masks,
   route-length correction, split-KV reduction, and fallback behavior.
2. **Tensor error:** MSE, MAE, max error, cosine, relative L2, and SQNR at a
   captured real model boundary.
3. **Decoded trajectory similarity:** sampled-frame PSNR/SSIM/cosine and audio
   cosine, SI-SDR, and spectral distance.
4. **Media inspection:** coherent subjects, motion, prompt alignment, absence
   of black/corrupt frames, valid H.264/AAC streams, and synchronized audio.

Diffusion trajectories can diverge while both outputs remain high quality.
Consequently, a low PSNR against BF16 is evidence of numerical divergence, not
proof that the generated video is perceptually bad. The media files are kept
beside every case so readers can inspect that distinction.

## Reproducibility checklist

- Hardware model and GPU count are stated for every primary chart.
- Resolution, frame count, sampling work, seed, and prompt are recorded.
- Raw JSON is checked in next to the figure it supports.
- Final media is linked both locally and through immutable PR attachments.
- Rejected settings and failed baselines remain documented.
- TeleFuser unit, integration, GPU, lint, and documentation checks are reported
  in the source PRs.

The exact commands remain in each TeleFuser PR and its benchmark README. This
repository is an evidence package and narrative layer, not a fork of the
runtime implementation.
