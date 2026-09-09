<!--
SECTION-CONTRACT
id: 00-abstract
incoming_premise: The reader has no project context.
outgoing_question: Why does world-model quality create this systems problem?
evidence: Only the final matched 4xH100 experiment may fill headline values.
do_not_claim: No historical PR metric is a headline result.
-->

# Fast and Faithful World-Model Inference

## Co-designing FP8, sparse attention, and sequence parallelism for MiniMax-H3

World models are expensive for the same reason they are useful. A generated
scene must remain visually detailed while people, objects, cameras, and sound
evolve coherently over time. MiniMax-H3 addresses that requirement with a large
diffusion transformer (DiT) that repeatedly processes a long joint
video--audio context. The resulting inference workload is dominated by dense
Linear and attention computation. Reducing resolution or denoising work makes
the system cheaper by relaxing the output requirement; our goal is to reduce
cost without changing that requirement.

FP8 quantization and sparse attention attack complementary parts of this cost.
FP8 lowers the precision of projection and matrix-multiplication work, while
sparsity avoids evaluating attention blocks that contribute little. In
practice, however, they are not independent switches. General quantization
libraries do not expose the layout, scale, and routing contract required by a
sparse attention kernel. Hardware-specific formats and kernels have uneven
architecture coverage. Applying both approximations naively can also turn
small, data-dependent errors into visible drift over a denoising trajectory.
The conflict becomes sharper under sequence parallelism, where an all-to-all
collective changes which tokens and heads define a local quantization domain.

This article presents an H100-oriented inference path that treats precision,
sparsity, and distribution as one numerical contract. It shares activation
quantization across QKV projections, prepares attention operands only after
normalization, rotary embedding, and Ulysses redistribution, and executes
dynamic Sol routing with FP8 QK/PV in an SM90 mainloop. Dense islands protect
sensitive denoising regions. Algebraically equivalent K/V centering and V-bias
correction reduce quantization error, while fused boundaries limit their
runtime cost. Adapter updates are merged before FP8 materialization, and
device-local caches are created after worker launch so the same path remains
valid for distilled MiniMax-H3 variants.

We evaluate the complete system, rather than isolated pull requests, on four
NVIDIA H100 GPUs. The primary baseline is the maintained FastVideo FastH3
Dense/Data-Free BF16+FA4 path with a matched checkpoint, adapter, output,
sampling work, and end-to-end timing boundary. The final measurements will
report:

- end-to-end latency reduction: **TBD--new experiment required**;
- single-request throughput improvement: **TBD--new experiment required**;
- peak GPU-memory change: **TBD--new experiment required**; and
- quality-suite acceptance: **TBD--new experiment required**.

Those numbers are intentionally blank in this draft. Before measuring them, we
first need to explain why the expensive computation exists and why simpler
optimizations do not preserve the same output contract.
