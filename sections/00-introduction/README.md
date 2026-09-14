<!--
SECTION-CONTRACT
id: 00-introduction
incoming_premise: none
outgoing_question: Why do low precision and sparse attention need one design?
evidence: experiments/h100-4gpu-e2e/raw/summary.json
do_not_claim: Do not generalize performance beyond the evaluated MiniMax-H3 configurations.
-->

# Q-SPA: Quantized Sparse-Parallel Attention for World Models with TeleFuser

[TeleFuser](https://github.com/Tele-AI/TeleFuser) is an open-source streaming
inference and serving framework for real-time world models and multimodal
generation. It brings model execution, distributed GPU inference, stateful
serving, and streaming delivery into one runtime. This post introduces Q-SPA:
a quantized, sparse, and parallel attention path for compute-intensive
diffusion transformers.

We use MiniMax-H3 as the proving ground. Its DiT jointly generates
high-resolution video and synchronized audio, so acceleration cannot come at
the cost of unstable motion, lost visual detail, or broken audio events. The
model also combines large dense layers, long-sequence attention, optional
adapters, and multi-GPU execution. It exposes exactly the interactions an
efficient world-model runtime must handle.

TeleFuser now brings three execution dimensions together:

- FP8 Linear and layout-aware FP8 attention;
- Sol-Attn sparsity with selective dense computation;
- Ulysses sequence parallelism, tensor parallelism, and communication overlap.

Three observations motivate this co-design. First, quantization and sparsity
are complementary rather than exclusive: when the FP8 representation is
consumed directly by the sparse attention kernel, their gains compound. Second,
general-purpose quantization paths can be unstable on world-model DiTs;
attention layouts, activation ranges, and hardware-specific kernels require a
targeted reconstruction of the execution path rather than a drop-in quantizer.
Third, fitting a model on one GPU does not remove the value of distribution.
DiT denoising is compute-bound, so sequence/tensor parallelism can spread the
work and overlap communication even when the weights fit in device memory.

We evaluate the same execution path on Base H3, Turbo LoRA, and FastH3 model
variants.

On the matched four-GPU Base H3 workload, with both frameworks running
`TP2 x Ulysses SP2`, TeleFuser generates a video **2.64x faster** than LightX2V
while using **40.3% less** representative peak GPU memory. We then test both
supported adapter families against their working external baselines and embed
the generated video and audio for direct comparison.

The rest of this post follows four questions:

1. Why do quantization and block-sparse attention conflict in existing kernels?
2. How does Q-SPA make their layouts compatible?
3. How does attention smoothing recover quality without giving back the speed?
4. How does the same attention path scale across GPUs?

We close with matched performance, memory, tensor-error, and media results.
