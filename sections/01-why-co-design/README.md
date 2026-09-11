<!--
SECTION-CONTRACT
id: 01-why-co-design
incoming_premise: MiniMax-H3 quality is a product constraint, while its DiT has more than one bottleneck.
outgoing_question: Where should precision conversion, sparse routing, and communication meet?
evidence: MiniMax-H3 execution profile and H100 kernel support
do_not_claim: General FP8 support implies an FP8 sparse-attention path.
-->

# One Model, Two Bottlenecks

Most of MiniMax-H3's denoising time sits in a large DiT, but "optimize the DiT"
is not one operation. Its dense projections and MLPs are dominated by matrix
multiplication. Its attention cost grows with the number of visual, audio, and
conditioning tokens. Making only the GEMMs cheaper leaves the token-pair work;
making only attention sparse leaves most model weights and activations in
BF16.

That is why FP8 and sparse attention are complementary:

| Technique | Reduces | Does not reduce |
|---|---|---|
| FP8 Linear | weight traffic and dense GEMM cost | the number of attention pairs |
| FP8 attention | QK/PV precision and bandwidth | dense attention's pair count |
| Sol-Attn | selected attention blocks | projection and MLP cost |
| Ulysses SP | per-rank sequence work and state | total work or communication |

The table looks modular. The actual tensors are not.

Consider the QKV path. The projection output is normalized, rotary position
embedding changes Q and K, sequence parallelism redistributes heads and
sequence shards, and sparse attention consumes a tiled layout plus routing
metadata. If a framework quantizes at every feature boundary, one attention
call can take the following route:

~~~text
FP8 Linear output
  -> BF16 normalization and RoPE
  -> per-rank Q/K/V quantization
  -> Ulysses all-to-all
  -> layout conversion
  -> sparse kernel quantization
  -> attention
~~~

The repeated conversion costs time, but the semantic problem is worse.
Per-tensor means and scales derived before all-to-all describe a sequence
shard. After redistribution, each rank owns different local heads over the
complete sequence. The scale is now attached to a tensor it did not summarize.
A path can therefore be individually "correct" at every API boundary and still
be numerically inconsistent as a system.

## The hardware boundary is part of the design

H100 makes this composition problem concrete. Hopper tensor cores support
useful FP8 execution, but a framework-level FP8 label does not provide every
kernel needed by MiniMax-H3. Linear libraries, dense attention, sparse
attention, routing, and fused output correction have different shape and
layout requirements. Newer MXFP8 and NVFP4 recipes can also depend on
Blackwell-specific hardware paths; their existence does not make them an SM90
solution.

We therefore use the quantization boundary as an architectural boundary.
Sensitive transforms stay in higher precision. Q, K, and V are quantized only
after they have their final per-rank attention semantics, and their scales are
passed directly to the consumer kernel. Sparse routing and FP8 QK/PV are owned
by the same H100 implementation.

This decision narrows the interface between features. It also creates a new
question: once low precision and sparsity perturb the same attention result,
how do we keep that combined approximation from steering the diffusion
trajectory away from a useful video?
