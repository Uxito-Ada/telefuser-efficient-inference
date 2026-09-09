<!--
SECTION-CONTRACT
id: 02-motivation
role: Turn observed deployment failures into the research questions.
sources: Negative and positive results from PRs 16, 25, 30, 35, 40, and 44.
edit_scope: Preserve the distinction between observed evidence and hypotheses.
-->

# 2. Motivation

The project began with a simple deployment request: run models without native
quantized checkpoints on available GPUs. The experiments quickly exposed five
gaps between enabling an optimization and obtaining an efficient system.

## Gap 1: compression is not acceleration

In the Qwen-Image case, online TorchAO FP8 reduced peak allocated memory from
59.47 GiB to 40.48 GiB but increased mean latency from 6.02 s to 8.35 s. BNB
NF4 reduced memory further to 31.30 GiB but also increased latency to 7.75 s.
The result is useful, but it rejects the assumption that a smaller dtype
automatically makes the workload faster. Dynamic quantization, dispatch, and
unsupported matrix shapes remain visible.

## Gap 2: quantized weights leave attention untouched

FP8 Linear produces BF16 Q/K/V. If attention immediately consumes those tensors
through a BF16 backend, the long-sequence QK and PV products still dominate.
Conversely, sparse BF16 attention reduces QK/PV work but retains BF16 model
weights. The first main research question is therefore:

> Can weight/activation quantization and dynamic attention sparsity share an
> FP8 data path without paying dequantization and materialization overhead
> between them?

## Gap 3: distributed layout changes scale semantics

The scale for a Q/K block is meaningful only for the tokens and heads that the
local kernel consumes. Quantizing before a Ulysses all-to-all makes each rank
compute scales over a sequence shard that is later rearranged. The second
question is:

> Where should FP8 preparation occur so that communication, scale granularity,
> and sparse routing describe the same local tensor?

## Gap 4: numerical validity is not perceptual validity

Early all-layer FP8 attention profiles produced visibly degraded videos even
when kernels completed and outputs were finite. Dense prefixes and partial
layer ranges recovered quality, but the final FP8 boundary still introduced
measurable error. This motivates a third question:

> Which precision-preserving transformations are mathematically neutral before
> quantization, and can their runtime cost be fused away?

## Gap 5: optimized weights are mutable

MiniMax-H3 Turbo and FastH3 are distributed as adapters over the same base
model, but their formats differ. Turbo uses standard low-rank tensors, while
FastH3 combines hundreds of low-rank pairs with exact residual deltas. A VSA
variant also contains learned replacement gates. This creates the fourth
question:

> Can the runtime merge all supported updates into the BF16 source of truth,
> quantize the final adapted weights once, and reject unsupported semantics
> rather than silently generating with an incomplete model?

These questions define the optimization ladder evaluated in the next sections.
