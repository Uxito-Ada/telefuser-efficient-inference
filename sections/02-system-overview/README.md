<!--
SECTION-CONTRACT
id: 02-system-overview
incoming_premise: FP8 and sparsity need a shared, hardware-aware attention boundary.
outgoing_question: How can this faster path preserve the statistics expected by MiniMax-H3?
evidence: TeleFuser PRs 16, 25, 30, 35, and 44
do_not_claim: TeleFuser invented Sol-Attn or makes all H3 operators FP8.
-->

# Building the H100 Execution Path

![TeleFuser MiniMax-H3 execution overview](assets/overview.svg)

The optimized request begins before the first token reaches attention. It
starts by defining which model is actually being executed.

## Prepare the effective weights

MiniMax-H3 can run as the base model or with a released Turbo or FastH3-style
adapter. An adapter is part of the model identity, not a post-processing
effect. TeleFuser loads the base checkpoint on CPU, applies low-rank and dense
adapter deltas in higher precision, and only then creates FP8 weight caches.

The order is:

~~~text
base BF16 weights
  -> merge the selected adapter
  -> quantize the effective weight
  -> build the device-local FP8 cache
  -> release superseded storage
~~~

Reversing the middle two operations would run stale base-model weights.
Keeping both copies after the merge would make an FP8 model look artificially
large in peak-memory measurements.

Multi-process execution adds another constraint. CUDA tensors created in a
parent process cannot be inherited safely by workers started with
`multiprocessing.spawn`. TeleFuser therefore starts multi-GPU MiniMax-H3 from
CPU weights and materializes each FP8 cache lazily inside the worker that owns
the target GPU. The Python module that exposes the external kernel is resolved
at runtime rather than stored inside every `FP8Linear` object, so the model
configuration remains pickleable during worker launch.

## Keep dense transformer work in FP8

The dominant projections and MLPs use cached FP8 weights with dynamically
quantized activations. Higher-precision source weights are preparation state,
not part of the steady-state data path. This distinction matters when reporting
memory: parameter compression is real, but the end-to-end peak also includes
text encoding, VAE state, activations, communication buffers, CUDA workspaces,
and allocator reservations. FP8 should not be advertised as "half the total
GPU memory" unless all those components are also halved.

## Quantize QKV where the kernel can consume it

Q/K normalization and rotary embedding remain in higher precision because
their reductions and phase transforms are sensitive. After Ulysses
redistribution, Q, K, and V have their final local-head/full-sequence meaning.
TeleFuser prepares them jointly, producing FP8 tensors and scale metadata in
the layout expected by Sol-Attn.

The custom SM90 kernel then owns the complete sparse attention operation:

1. summarize candidate key blocks and select them online;
2. execute the selected QK tiles in FP8;
3. apply the sparse softmax path and correction terms;
4. execute the selected probability-value tiles;
5. write the corrected output once.

TMA moves tiled data while WGMMA executes tensor-core matrix operations. More
important than the instruction names, the kernel avoids bouncing through
generic attention formats between routing and compute. Dense-prefix
replacement and sparse-output correction are merged into the same output pass,
so enabling quality controls does not require writing and rereading a full
attention tensor.

Unsupported shapes retain a validated fallback. Hardware specialization is
useful only when it fails explicitly; silently accepting an untested shape is
not portability.

This path solves the execution problem, but it also compounds two
approximations. The next step is to make the kernel preserve the attention
statistics that later denoising blocks expect.
