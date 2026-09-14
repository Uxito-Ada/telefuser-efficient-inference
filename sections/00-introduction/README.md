<!--
SECTION-CONTRACT
id: 00-introduction
incoming_premise: none
outgoing_question: Why do low precision and sparse attention need one design?
evidence: experiments/h100-4gpu-e2e/raw/summary.json
do_not_claim: Do not generalize performance beyond the evaluated MiniMax-H3 configurations.
-->

# Q-SPA: Efficient World Model Inference with TeleFuser

*Quality-Aware Quantization, Sparse Attention, Parallelism, and Adapters*

[TeleFuser](https://github.com/Tele-AI/TeleFuser) is an open-source streaming
inference and serving framework for real-time world models and multimodal
generation. It brings model execution, distributed GPU inference, stateful
serving, and streaming delivery into one runtime. This post introduces Q-SPA,
an optimization stack that combines quality-aware FP8 quantization, sparse
attention, parallelism, and adapters for compute-intensive diffusion
transformers.

We use MiniMax-H3 as the proving ground. Its DiT jointly generates
high-resolution video and synchronized audio, so acceleration cannot come at
the cost of unstable motion, lost visual detail, or broken audio events. The
model also combines large dense layers, long-sequence attention, optional
adapters, and multi-GPU execution. It exposes exactly the interactions an
efficient world-model runtime must handle.

TeleFuser now brings those pieces together with:

- FP8 Linear compute and a hardware-aware FP8 Sol-Attn path;
- quality-aware FP8 attention and selective dense computation;
- base, Turbo LoRA, and FastH3-style adapter support;
- Ulysses sequence parallelism, tensor parallelism, and communication overlap.

On the matched four-GPU Base H3 workload, with both frameworks running
`TP2 x Ulysses SP2`, TeleFuser generates a video **2.64x faster** than LightX2V
while using **40.3% less** representative peak GPU memory. We then test both
supported adapter families against their working external baselines and embed
the generated video and audio for direct comparison.

The rest of this post follows four questions:

1. Why do FP8 and sparse attention need to be designed together?
2. What did TeleFuser add to make that combination practical in one runtime?
3. How does attention smoothing recover quality without giving back the speed?
4. How do adapters and multi-GPU execution fit into the same optimized path?

We close with matched performance, memory, tensor-error, and media results.
