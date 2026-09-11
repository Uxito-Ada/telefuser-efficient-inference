<!--
SECTION-CONTRACT
id: 00-introduction
incoming_premise: none
outgoing_question: Why do low precision and sparse attention need one design?
evidence: experiments/h100-4gpu-e2e/raw/summary.json
do_not_claim: Do not generalize beyond MiniMax-H3 on H100.
-->

# TeleFuser: Fast and Faithful World Model Inference

*Co-designing FP8, sparse attention, adapters, and multi-GPU execution on H100*

[TeleFuser](https://github.com/Tele-AI/TeleFuser) is an open-source streaming
inference and serving framework for real-time world models and multimodal
generation. It brings model execution, distributed GPU inference, stateful
serving, and streaming delivery into one runtime. This post focuses on a major
addition to its inference optimization stack: a quality-aware FP8 sparse path
for compute-intensive diffusion transformers.

We use MiniMax-H3 as the proving ground. Its DiT jointly generates
high-resolution video and synchronized audio, so acceleration cannot come at
the cost of unstable motion, lost visual detail, or broken audio events. The
model also combines large dense layers, long-sequence attention, optional
adapters, and multi-GPU execution. It exposes exactly the interactions an
efficient world-model runtime must handle.

TeleFuser now brings those pieces together with:

- FP8 Linear compute and an H100-native FP8 Sol-Attn path;
- attention smoothing and selective dense computation for output quality;
- base, Turbo LoRA, and FastH3-style adapter support;
- Ulysses sequence parallelism, tensor parallelism, and communication overlap.

On the matched single-H100 FastH3 adapter workload, this path reduces denoising
time by **25.7%**, raises DiT throughput by **34.6%**, and lowers peak GPU memory
by **14.3%** against FastVideo BF16 Linear + FlashAttention 4. The generated
video and audio are embedded below alongside the baseline.

The rest of this post follows four questions:

1. Why do FP8 and sparse attention need to be designed together?
2. What did TeleFuser add to make that combination practical on H100?
3. How does attention smoothing recover quality without giving back the speed?
4. How do adapters and multi-GPU execution fit into the same optimized path?

We close with matched performance, memory, tensor-error, and media results.
