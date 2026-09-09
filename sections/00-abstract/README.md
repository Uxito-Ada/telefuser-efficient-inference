<!--
SECTION-CONTRACT
id: 00-abstract
role: State the thesis, contributions, and only cross-case headline results.
sources: TeleFuser PRs 16, 25, 30, 35, 40, and 44.
edit_scope: Update after any canonical claim changes; do not add unmatched aggregate speedups.
-->

# From Quantization to Quality

## Building an Efficient Video-DiT Inference Stack in TeleFuser

### Abstract

Video diffusion transformers combine large model weights, long spatiotemporal
sequences, iterative denoising, and expensive media decoders. Optimizing one
operator rarely optimizes the system: weight-only compression may save memory
without accelerating attention; sparse attention may reduce FLOPs while adding
workspace; sequence parallelism changes the tensor over which quantization
scales are valid; and a LoRA adapter invalidates a cached quantized weight.

This article presents the evolution of an efficient inference stack in
[TeleFuser](https://github.com/Tele-AI/TeleFuser) through six implementation
case studies. We first introduce online FP8 and NF4 quantization for image and
video DiTs, then specialize it for the 35B-class multimodal MiniMax-H3
transformer. We carry FP8 across the attention boundary with block-scaled Q/K,
channel-scaled V, and a Hopper CuTe Sol-Attn mainloop. We compose that kernel
with Ulysses sequence parallelism, recover part of the quantization error with
an attention-equivalent K/V centering transform, and merge both standard and
hybrid FastH3 adapters before optional FP8 materialization.

The evaluation uses generated images and synchronized video-audio outputs in
addition to latency, throughput, memory, kernel error, and trajectory-similarity
metrics. On a single H100, MiniMax-H3 FP8 Linear + FP8 Sol raises denoising
throughput by 65.0% over BF16 Dense while using 41.9% less peak allocated
memory. On four H100s, TP2 x Ulysses SP2 FP8 Sol raises throughput by 53.2% over
the matched TeleFuser BF16 + FlashAttention-4 baseline and reaches 2.62x the
denoising speed of the deployed LightX2V comparison. Attention smoothing
reduces dense attention-output MSE by 8.18% at a 2.11% throughput cost.
Finally, the adapter path improves throughput by 36.5% for MiniMax-H3 Turbo
relative to LightX2V and by 57.2% for FastH3 relative to FastVideo on their
matched single-H100 workloads.

The central result is methodological: efficient generative inference is a
contract across numerical representation, kernel layout, distributed tensor
ownership, model lifecycle, and quality validation. An optimization is not
complete until those contracts agree.
