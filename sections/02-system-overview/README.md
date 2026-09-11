<!--
SECTION-CONTRACT
id: 02-system-overview
incoming_premise: FP8 and sparsity need one hardware-aware model contract.
outgoing_question: How does TeleFuser keep the combined approximation stable?
evidence: TeleFuser PRs 16, 25, 30, 35, and 44
do_not_claim: TeleFuser invented Sol-Attn or makes all H3 operators FP8.
-->

# TeleFuser's Optimization Stack

The MiniMax-H3 work extends TeleFuser at three levels: dense DiT compute,
long-sequence attention, and model-scale execution. Users select one supported
inference profile that brings these capabilities together.

## H100-native FP8 sparse attention

TeleFuser applies FP8 to the DiT projections and MLPs, then carries the same
low-precision objective into attention with an SM90 implementation of
Sol-Attn. Sol-Attn selects important attention regions online; the TeleFuser
path combines that sparsity with FP8 QKV compute instead of returning to a
BF16 attention backend.

Owning both precision and sparsity in the same attention implementation removes
a common integration gap: the output of a quantized transformer no longer has
to pass through repeated conversions before sparse attention can use it. The
implementation retains validated dense fallbacks for unsupported cases.

## Adapter-aware low-precision inference

MiniMax-H3 is used both as a base model and with acceleration or style adapters.
TeleFuser supports the official Turbo LoRA path as well as FastH3-style LoRA
and dense adapters. Adapter changes are incorporated into the effective model
before its reusable FP8 representation is created, ensuring that low-precision
execution represents the requested model rather than the base checkpoint.

This support matters beyond compatibility. A distilled adapter can reduce the
number of DiT evaluations, while FP8 and sparsity reduce the cost of each
evaluation. TeleFuser can apply both forms of acceleration in the same request.

## Distributed execution for long sequences

For larger deployments, Ulysses sequence parallelism divides long attention
work across GPUs and tensor parallelism divides wide transformer layers.
TeleFuser supports two- and four-GPU MiniMax-H3 topologies, including
`TP2 x Ulysses SP2`, and can overlap Ulysses communication with attention
compute.

The result is a single optimization stack spanning fewer model evaluations,
lower-precision dense compute, sparse low-precision attention, and distributed
execution. Because FP8 rounding and sparsity affect the same denoising
trajectory, quality preservation is built into this path as well.
