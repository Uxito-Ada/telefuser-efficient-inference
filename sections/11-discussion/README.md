<!--
SECTION-CONTRACT
id: 11-discussion
role: Interpret the cross-case result, preserve limitations, and identify next work.
sources: All cases plus the related-work study.
edit_scope: Clearly mark proposals as future work rather than measured results.
-->

# 11. Discussion

## What actually composes

The six cases expose five optimization layers:

1. **Capacity:** compressed Linear weights make a model fit.
2. **Compute:** FP8 GEMMs and sparse QK/PV reduce denoising work.
3. **Distribution:** Ulysses changes tensor ownership and increases aggregate
   bandwidth.
4. **Numerics:** dense islands and smoothing recover quality.
5. **Lifecycle:** adapter merge, cache construction, process launch, and
   component residency determine whether the optimized graph is executable.

The layers compose only through explicit boundaries. FP8 Linear returns BF16
Q/K/V because norm and RoPE remain BF16. FP8 attention starts after those
operators. Ulysses communicates BF16 because the destination rank defines the
correct attention scale domain. Adapter merge precedes both cache creation and
parallel sharding.

## Why FP8 memory is not simply half of BF16

Only the tensors actually quantized can approach a two-to-one storage ratio.
Peak memory also includes unquantized encoders and VAEs, activation workspaces,
allocator reservation, route state, and phase overlap. In PR #44, the FP8 DiT
is correctly materialized and source BF16 weights are released; text encoding,
not denoising, produces the whole-process peak. A dtype claim must therefore be
paired with a tensor inventory and a phase timeline.

## Why sparse attention can use more memory

Sol reduces exact QK/PV work, but its router needs K/V summaries, thresholds,
bitmasks, LSE state, and optionally split-KV partial outputs. PR #30's
MiniMax-H3 FP8 Sol profile uses 6.1% more peak allocated memory than FP8 Dense
while delivering 47.1% more throughput. The correct description is
compute-efficient sparse attention, not free memory compression.

## Why exact quality fixes still need fusion

K/V centering is algebraically neutral before quantization, but exact sequence
statistics are global reductions. A naive implementation introduces enough
launch and memory traffic to weaken the end-to-end gain. PR #40 recovers 23.1%
at the combined smoothing boundaries through fusion, yet still pays a measured
2.11% throughput cost. Correctness supplies the transformation; kernel
engineering determines whether it is deployable.

## Limitations

- Native FP8 Sol is validated on NVIDIA Hopper SM90, noncausal self-attention,
  equal Q/K/V shapes, and head dimension 128. Other architectures use
  fallbacks.
- Ring and Ulysses-ring do not use native Sol because the current contract lacks
  a distributed log-sum-exp merge.
- PR #40 quality metrics cover one prompt and seed. They support mechanism
  analysis, not a population-level perceptual claim.
- Attention smoothing improves the measured video metrics but not the reported
  audio metrics.
- Cross-framework runtimes differ in allocators, component lifetime, seed
  mapping, compilation, and caching. The article matches workloads and states
  remaining differences rather than claiming bitwise equivalence.
- Dense FastH3 adapters are supported; VSA replacement-gate adapters are
  rejected until TeleFuser has a compatible learned sparse backend.
- Several source PRs were measured on evolving PyTorch/CUDA revisions. Raw
  snapshots from different environments are not merged into one ranking.

## Open directions

**Fuse statistics deeper.** The remaining smoothing reduction could be
combined with the attention preparation pipeline if exact accumulation order
and reuse permit it.

**Distributed Sol beyond Ulysses.** Exposing partial max, sum, and output state
would make a ring-compatible sparse merge possible, but routing balance and
approximation correction would need a distributed contract.

**Blackwell formats.** Recent NVFP4 and FlashAttention-4 results suggest a new
co-design point for block scaling and asymmetric tensor-core pipelines. The
current H100 evidence should not be projected onto B200 without re-tuning.

**Learned sparse adapters.** Supporting FastH3 VSA means implementing its
compression gates, not ignoring them. The loader's explicit rejection creates
a safe boundary for that future backend.

**Population-level multimodal evaluation.** A publishable quality study should
add prompt suites, multiple seeds, temporal and audio perceptual metrics, and
human preference, while retaining operator-level error to diagnose regressions.
