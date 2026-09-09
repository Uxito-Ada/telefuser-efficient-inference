<!--
SECTION-CONTRACT
id: 02-composition-gap
incoming_premise: The fixed-quality workload requires both low precision and fewer attention operations.
outgoing_question: What shared H100 execution path can make them one operation?
evidence: H3 operator graph, backend capability matrix, and rejected-path logs.
do_not_claim: Do not say all TorchAO FP8 is unavailable on H100.
-->

# 2. Two Necessary Optimizations That Do Not Compose

Suppose we quantize every eligible MiniMax-H3 Linear layer. The Q, K, and V
projections may use FP8 tensor cores, but normalization and rotary position
embedding still operate on BF16 tensors. A conventional attention backend then
consumes BF16 Q/K/V, so the longest QK and PV products remain outside the FP8
path.

Now suppose we instead enable sparse attention. Sol-Attn can avoid exact work
for low-importance blocks, but its projections and feed-forward layers are
unchanged. The model still carries BF16 weights and executes dense BF16 GEMMs
around a less expensive attention operator.

Running both configurations in the same process does not close the gap. The
boundary typically becomes:

```text
hidden state
  -> quantize for Q projection -> FP8 GEMM -> BF16 Q
  -> quantize for K projection -> FP8 GEMM -> BF16 K
  -> quantize for V projection -> FP8 GEMM -> BF16 V
  -> norm and RoPE
  -> transpose / summarize / rescale for sparse attention
  -> quantize again inside the attention backend
```

The hidden state is quantized three times. Q/K/V are materialized in an
intermediate layout, transformed, and quantized again. Kernel launches and
HBM traffic accumulate around the operation that was supposed to get cheaper.
More importantly, the Linear backend and attention backend choose scales over
different tensor domains. They are adjacent features in a configuration file,
not one low-precision data path.

## The hardware gap behind the API gap

Low-precision kernels are hardware-aware by construction. A numerical format
is only useful when the target tensor cores, instruction set, data movement,
tile shape, and software interface support it. Names such as FP8, MXFP8, and
NVFP4 do not describe interchangeable implementations.

On H100 (SM90), generic E4M3 Linear paths are available, including selected
TorchAO configurations. But a general Linear API does not expose Sol's block
summaries, dynamic route decisions, online-softmax state, or attention-specific
Q/K/V scale layout. MXFP8 and NVFP4 optimized routes increasingly target
Blackwell capabilities, while several high-performance attention kernels are
distributed through interfaces that cannot be extended with an independent
sparse mainloop. A kernel may therefore be excellent on its supported shape
and architecture yet unusable for this particular composition.

Weight-only NF4 illustrates a different mismatch. It can reduce model storage,
but dequantized BF16 compute does not create an FP8 QK/PV path. Treating it as a
substitute because both methods are called quantization confuses capacity with
compute throughput.

The engineering question is consequently narrower and harder than selecting a
library:

> Which process should own scaling, layout, sparsity, and accumulation for
> MiniMax-H3 attention on SM90?

If ownership is split across opaque operators, every boundary can reformat or
requantize the tensor. If one operator owns everything, it must also preserve
the exact dense contribution, the approximate summary contribution, and the
softmax normalization correctly.

## Distribution makes a local mismatch global

The same ownership problem appears under sequence parallelism. Before a
Ulysses all-to-all, a rank holds a subset of tokens across all heads. After the
collective, it holds the full sequence for a subset of heads. A scale computed
before communication describes a different tensor from the one consumed by the
attention kernel. Likewise, a sparse route selected from an incomplete
sequence does not represent the final local attention problem.

This observation gives us the central design rule used throughout the system:

> The final consumer of a tensor must define its quantization and approximation
> domain. Communication and exact nonlinear transforms happen first; low-
> precision preparation and sparse routing happen where the kernel has its
> final local sequence and heads.

Following that rule requires more than glue code. We need an SM90 path that
shares QKV activation work, chooses attention-specific scales after norm and
RoPE, and fuses dynamic sparse attention without exposing incompatible
intermediate formats. That path is the subject of the next section.
