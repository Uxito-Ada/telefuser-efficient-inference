<!--
SECTION-CONTRACT
id: 03-sm90-fp8-sol
incoming_premise: Existing Linear and sparse-attention APIs do not share a low-precision tensor contract on H100.
outgoing_question: Once both computations are approximate, how is trajectory quality protected?
evidence: Kernel contracts, H3 tensor captures, operator tests, and new profiler traces.
do_not_claim: Attribute Sol routing to Sol-Attn; claim only the TeleFuser integration and SM90 FP8 implementation.
-->

# 3. Building One FP8 Sparse Path for SM90

The composition gap gives us a more useful starting point than the instruction
to "use FP8." The final attention consumer must own the precision and layout of
its operands. Everything that determines their exact value must happen before
that boundary; everything that exists only to serve the sparse kernel should be
fused after it.

For MiniMax-H3, the resulting path is:

```text
BF16 hidden state
  -> one dynamic activation quantization
  -> cached FP8 Q, K, and V projection GEMMs
  -> BF16 QK normalization and RoPE
  -> final local sequence/head layout
  -> attention-aware FP8 preparation
  -> Sol routing + QK + online softmax + PV
  -> BF16 corrected output
```

The BF16 segment is intentional. QK normalization and rotary embedding are
small relative to the surrounding GEMMs, are sensitive nonlinear transforms,
and change the final Q/K values. Quantizing before them would either force the
operators to consume an unsuitable format or require immediate dequantization.
Quantizing after them makes the scale describe exactly what attention sees.

## Reusing work across QKV projections

Q, K, and V are three different weight matrices applied to the same hidden
state. A generic dynamic FP8 Linear wrapper has no reason to know that they form
one logical operation, so it quantizes the activation independently for each
projection. MiniMax-H3 pays the same reduction and conversion cost three times.

TeleFuser moves activation preparation to the QKV group. The hidden state is
quantized once with its row-wise scale, and the resulting activation and scale
are reused by the three weight GEMMs. Each projection retains its own cached
E4M3 weight and weight scale. This optimization follows from data reuse, not
from a new numerical approximation: the three independent wrappers would have
computed the same activation quantization for the same input.

The cache matters for the same reason. Online quantization begins from a
standard BF16 checkpoint, but the static weight does not need to be converted
on every denoising update. We materialize the FP8 weight once, retain the scale
required by the GEMM, and later release the superseded BF16 source. Runtime
activation quantization remains dynamic because the hidden-state distribution
changes with the prompt and timestep.

## Attention needs different scale geometry

The projection GEMM's activation scale is not automatically a good attention
scale. Sol routes and computes 64-token blocks, so Q and K are prepared at a
granularity aligned with those blocks. For batch \(b\), head \(h\), and
token block \(i\):

\[
s^Q_{bhi}=\frac{\max |Q_{bhi}|}{448}, \qquad
s^K_{bhi}=\frac{\max |K_{bhi}|}{448}.
\]

V participates in the probability-times-value product. Its useful dynamic
range varies by channel, so it uses a per-batch, per-head, per-channel scale
over the token dimension:

\[
s^V_{bhd}=\frac{\max_t |V_{bthd}|}{448}.
\]

The preparation kernel also writes the token-contiguous backing layout expected
by the PV operand. Performing scaling and layout production together avoids a
standalone transpose that would move the full live QKV tensor through HBM.
Tail tokens are padded and masked so the 64-token tile contract never changes
the logical sequence length.

## Sparse routing belongs inside the attention mainloop

[Sol-Attn](https://nvlabs.github.io/Sana/Sol-Attn/) summarizes K/V blocks and
uses a query-dependent proxy to decide which blocks deserve exact attention.
Blocks above a threshold such as \(\mu + \tau\sigma\) enter the exact route;
the remaining blocks contribute through a compact summary correction. Unlike a
hard top-k drop, the approximate route still participates in the same
online-softmax normalization.

The SM90 implementation uses E4M3 QK and PV operations with FP32 accumulation
and BF16 output. TMA moves tiled operands, while WGMMA executes tensor-core
matrix products. Routing, exact QK, summary contribution, online max/sum
updates, and PV accumulation remain in the kernel's tiled loop. The
implementation does not materialize the quadratic attention matrix or export a
global route mask to memory.

This ownership is the reason for implementing a dedicated path instead of
wrapping an opaque dense FP8 operator:

| Responsibility | Generic operator boundary | Unified FP8 Sol path |
|---|---|---|
| QKV activation preparation | Three independent calls | One shared conversion |
| Attention scale domain | Hidden behind another backend | Matches Sol tiles |
| Sparse route | Separate mask or unsupported | CTA-local mainloop state |
| Softmax normalization | Dense operator owned | Exact and summary routes merged |
| Layout conversion | Standalone materialization | Written by preparation kernel |
| Architecture contract | Backend-dependent | Explicit SM90, E4M3, head-dim 128 |

## Fallbacks are part of correctness

The native path has a deliberately narrow contract: CUDA SM90, non-causal
self-attention, supported Q/K/V shapes, and head dimension 128. Inputs outside
that contract use a validated fallback or fail with an actionable message.
Silently selecting a nearby kernel would be dangerous because a different mask,
scale convention, or route correction can produce plausible but incorrect
media.

At this point the system has removed redundant quantization boundaries and can
execute both low-precision dense GEMMs and sparse attention on H100. It has
also combined two sources of approximation inside a recurrent denoising
process. The next question is no longer whether the kernel runs, but where its
error is safe.
