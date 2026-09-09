<!--
SECTION-CONTRACT
id: 08-discussion
incoming_premise: The final experiment has tested the complete system against external references.
outgoing_question: none
evidence: Final experiment plus explicit implementation boundaries.
do_not_claim: Pending results are hypotheses, not findings.
-->

# 8. Discussion

The system in this article follows from one constraint: improve inference
without weakening the MiniMax-H3 output request. That constraint is what made
FP8 and sparse attention necessary together, exposed their incompatible
interfaces, required a dedicated SM90 path, and turned numerical quality,
parallel layout, and adapter loading into parts of the same design.

Several broader lessons follow from the method. Their measured magnitude
remains pending until the flagship experiment is complete.

## A dtype is not an execution strategy

FP8 names a representation, not a guaranteed kernel, layout, or speedup.
Hardware support, scale granularity, tensor shape, fusion, and framework
coverage determine whether lower precision reduces wall time. A generic Linear
quantizer can be correct and still leave attention untouched. A Blackwell-
oriented MXFP8 or NVFP4 result cannot be assumed to transfer to H100. The
relevant unit of design is the end-to-end operator path on a named
architecture.

## Approximation methods share one error budget

Quantization and sparsity are often evaluated separately, but the composed
system does not experience separate errors. Rounded Q/K values influence sparse
routing, sparse routing determines which rounded values receive exact work, and
their output re-enters the denoising trajectory. Dense islands and smoothing
are therefore not optional polish after optimization; they allocate and reshape
the common error budget.

This also explains why operator MSE and generated quality must be reported
together. Operator error helps identify a mechanism. Video and audio reveal
whether that mechanism matters after dozens of layers, multiple updates, and
decoding.

## Distribution is part of numerical semantics

Sequence parallelism is usually introduced as a performance technique. Here it
also decides which tokens and heads define a scale, mean, block, and sparse
route. Moving quantization across an all-to-all can change the represented
function even if every local kernel is individually correct. Distributed
layout must be treated as part of an operator's numerical contract.

## Model state extends beyond the checkpoint

An optimized deployment contains derived state: packed weights, scales,
compiled kernels, sparse metadata, and device-local caches. An adapter mutates
the model that this state represents. Correctness requires explicit invalidation
or, as in this path, merge-before-materialize ordering. Process launch adds a
second ownership boundary: serializable policy can cross spawn, but CUDA
tensors and Python module objects must be created or resolved by the worker
that owns them.

## Limitations

- The native FP8 Sol path targets NVIDIA H100/SM90, E4M3, non-causal
  self-attention, and the MiniMax-H3 head geometry. It is not a portable
  low-precision attention implementation.
- Ulysses SP4 is evaluated because it gives each rank the complete sequence for
  local heads. Ring-compatible sparse routing would require a new distributed
  summary and online-softmax merge.
- The strict performance claim uses the Dense/Data-Free adapter. VSA uses
  learned gates and is reported only as related context.
- Whole-process peak memory includes text encoding and media decoding, so it
  need not follow the DiT weight compression ratio.
- The quality suite samples predefined prompts and seeds but cannot replace a
  large blinded human preference study.
- Warm serving latency does not describe cold model load, adapter merge, or
  first kernel compilation; those costs are reported separately.

## Conclusion

High-quality world-model inference is not accelerated by choosing between
quantization, sparsity, and parallelism. The useful operating point comes from
making them agree on the same tensor and model:

- quantize the work that remains;
- sparsify the work that need not be exact;
- preserve sensitive computation through mathematically justified controls;
- distribute before defining local scales and routes; and
- materialize optimized weights only after the model has reached its final
  adapted state.

On the matched four-H100 MiniMax-H3 workload, the complete path delivers
**TBD--new experiment required** end-to-end acceleration with
**TBD--new experiment required** quality outcome. Those final values will be
inserted from raw experiment artifacts, not reconstructed from the development
history.
