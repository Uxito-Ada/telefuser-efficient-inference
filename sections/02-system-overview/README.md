<!--
SECTION-CONTRACT
id: 02-system-overview
incoming_premise: FP8 and sparsity need one hardware-aware model contract.
outgoing_question: How does TeleFuser keep the combined approximation stable?
evidence: TeleFuser PRs 16, 25, 30, 35, and 44
do_not_claim: TeleFuser invented Sol-Attn or makes all H3 operators FP8.
-->

# Q-SPA in TeleFuser {#q-spa-system}

The MiniMax-H3 work extends TeleFuser at three levels: dense DiT compute,
long-sequence attention, and model-scale execution. Users select one supported
inference profile that brings these capabilities together.

## Layout-aware FP8 sparse attention

TeleFuser applies FP8 to the DiT projections and MLPs, then carries the same
precision into Sol-Attn. Sparse block indices, quantization scales, and the
tiles consumed by the attention kernel are kept in the same layout contract.
Selected QKV blocks can therefore remain in FP8 instead of returning to a BF16
attention backend.

The implementation retains validated dense fallbacks for unsupported cases.

## Model variants

MiniMax-H3 is used both as a base model and with acceleration or style adapters.
TeleFuser supports the official Turbo LoRA path as well as FastH3-style LoRA
and dense adapters. Adapter changes are incorporated into the effective model
before its reusable FP8 representation is created, ensuring that low-precision
execution represents the requested model rather than the base checkpoint.

A distilled adapter can reduce the number of DiT evaluations, while Q-SPA
reduces the cost of each evaluation.

## Distributed execution for long sequences

For larger deployments, Ulysses sequence parallelism divides long attention
work across GPUs and tensor parallelism divides wide transformer layers.
TeleFuser supports two- and four-GPU MiniMax-H3 topologies, including
`TP2 x Ulysses SP2`, and can overlap Ulysses communication with attention
compute.

The result is one attention path spanning low-precision dense compute, sparse
low-precision attention, and distributed execution. Because FP8 rounding and
sparsity affect the same denoising trajectory, quality preservation is built
into this path as well.
