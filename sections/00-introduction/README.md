<!--
SECTION-CONTRACT
id: 00-introduction
incoming_premise: none
outgoing_question: Why do low precision and sparse attention need one execution design?
evidence: experiments/h100-4gpu-e2e/raw/summary.json
do_not_claim: Do not generalize beyond MiniMax-H3 on H100.
-->

# Fast and Faithful MiniMax-H3 Inference on H100

*Making FP8, sparse attention, adapters, and multi-GPU execution work as one system*

MiniMax-H3 is a useful stress test for efficient AI systems. A single diffusion
transformer produces a high-resolution video together with synchronized audio.
The model must preserve appearance, motion, temporal reasoning, speech, and
event-aligned sound across many denoising updates. Those requirements are what
make the output compelling, and also what make inference expensive.

The obvious remedies attack different parts of the workload. FP8 reduces the
cost of the large projections and feed-forward layers. Sparse attention avoids
evaluating token pairs that contribute little to the output. Sequence
parallelism divides a long multimodal sequence across GPUs. Distilled adapters
reduce the number of transformer evaluations.

The difficult part is not enabling those features one at a time. It is keeping
them useful when they meet.

A generic quantization wrapper may support Linear layers but not sparse QK/PV
attention. A sparse kernel may assume BF16 inputs or a layout that forces Q, K,
and V to be converted and quantized again. A statistic computed before
sequence-parallel redistribution describes the wrong tensor. An adapter merged
after FP8 caching leaves the cache representing the base model rather than the
model the user requested. Each mistake is local; their effects accumulate
through the denoising trajectory.

This is the systems problem we addressed in TeleFuser. The resulting
MiniMax-H3 path combines:

- cached FP8 Linear weights and dynamic FP8 activations;
- an H100-native FP8 Sol-Attn kernel with online sparse routing;
- attention smoothing and selective dense work to stabilize quality;
- Ulysses sequence parallelism, tensor parallelism, and communication overlap;
- base, Turbo LoRA, and FastH3-style adapter loading before quantization.

The rest of this post follows the order in which those constraints arise. We
first explain why FP8 and sparsity cannot be composed as independent switches.
We then follow one attention tensor through the H100 execution path, add the
quality controls demanded by that approximation, and finally distribute the
same semantics across GPUs. The evaluation separates operator-level diagnosis
from the result that can be compared honestly: a matched denoising comparison
against another maintained MiniMax-H3 runtime, a separately scoped multi-GPU
validation, and the generated video and audio available to inspect.

The goal is not a collection of isolated kernel wins. It is a faster
MiniMax-H3 request whose output remains worth generating.
