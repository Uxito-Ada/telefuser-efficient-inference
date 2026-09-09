<!--
SECTION-CONTRACT
id: 01-background
role: Explain the cost model and prior techniques without claiming TeleFuser invented them.
sources: MiniMax H3, Sol-Attn, SageAttention2, xDiT, FlashAttention-4, VSA.
edit_scope: Concepts and related work only; case-specific measurements belong later.
-->

# 1. Background

## Why video DiTs are a systems problem

Diffusion transformers repeatedly evaluate a large transformer while denoising
a latent video. Compared with image generation, the temporal axis makes the
attention sequence much longer. MiniMax-H3 adds another systems dimension: one
model produces video and synchronized stereo audio and supports multimodal
conditioning. In the 768p workload studied here, a live attention tensor has
32,626 tokens, 56 heads, and head dimension 128.

The repeated transformer contains two dominant matrix-multiplication families:

1. **Linear GEMMs** in QKV projections, output projections, and feed-forward
   networks.
2. **Attention GEMMs** in
   \(\operatorname{softmax}(QK^T / \sqrt d)V\).

They have different data layouts, scaling requirements, and fusion
opportunities. A 2-D FP8 Linear kernel does not automatically accelerate the
online softmax or the QK/PV path. This distinction drives the architecture in
the later case studies.

## Four complementary efficiency axes

**Low-precision Linear.** Hopper Tensor Cores can accelerate FP8 E4M3 GEMMs.
Dynamic W8A8 inference stores weights in FP8 and quantizes each activation row
at runtime. NF4 instead provides aggressive weight-only compression with BF16
compute. FP8 usually targets throughput and memory together; NF4 primarily
targets capacity.

**Sparse attention.** [Sol-Attn](https://nvlabs.github.io/Sana/Sol-Attn/)
routes important key-value blocks to exact attention and approximates the
remaining blocks inside the same online-softmax state. Unlike a hard drop mask,
the summary route retains a compressed contribution. This makes the sparsity
budget dynamic and avoids materializing a global routing map.

**Distributed sequence ownership.** [xDiT](https://arxiv.org/abs/2411.01738)
shows why diffusion inference benefits from composable parallel dimensions.
Ulysses sequence parallelism applies all-to-all communication so each rank
receives the full sequence for a subset of heads. Ring attention instead keeps
sequence shards local and circulates K/V blocks while merging online-softmax
statistics. Those layouts are mathematically equivalent for dense attention,
but they expose different local tensors to a quantizer and sparse router.

**Model compression and adaptation.** A low-rank adapter changes the effective
weight from \(W\) to \(W + \Delta W\). If a runtime has already cached an FP8
copy of \(W\), applying the adapter afterward leaves the optimized kernel with
stale weights. Adapter support therefore belongs in the quantization lifecycle,
not only in model loading.

## Precision preservation is part of the kernel

Quantized attention is not defined only by a dtype. Scale granularity,
accumulator precision, outlier handling, and algebraic transforms determine the
error. [SageAttention2](https://arxiv.org/abs/2411.10958), for example, combines
low-precision QK/PV kernels with smoothing and fine-grained quantization.
[FlashAttention-4](https://arxiv.org/abs/2603.05451) similarly illustrates that
algorithm and pipeline design must follow the asymmetry of the target hardware.

For diffusion, a small local error can perturb a trajectory and produce a
visually different but still valid sample. We therefore evaluate three levels:

- operator error against a high-precision reference;
- end-to-end performance and peak memory; and
- decoded image, video, and audio evidence.

No one level substitutes for the others.
