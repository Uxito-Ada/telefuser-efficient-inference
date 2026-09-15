<!--
SECTION-CONTRACT
id: 05-lessons
incoming_premise: TeleFuser's combined path improves performance while retaining measurable output quality.
outgoing_question: none
evidence: final benchmark and quality records
do_not_claim: Performance portability beyond the tested MiniMax-H3 configurations.
-->

# Q-SPA in TeleFuser

Q-SPA addresses three connected problems in TeleFuser:

- aligning FP8 scale groups with dynamically selected attention blocks;
- controlling error when FP8 and sparsity affect the same denoising trajectory;
- running sparse FP8 attention with Ulysses SP, tensor parallelism, and
  communication overlap.

TeleFuser can also run Base H3 or merge a Turbo LoRA or FastH3 adapter before
creating the FP8 weights. The adapter defines the model and sampling schedule;
Q-SPA reduces the cost of each DiT evaluation.

On the matched four-GPU Base H3 workload, TeleFuser completes generation in
52.27 seconds. It reduces latency by 62.1% against LightX2V and 34.1% against
SGLang, with 40.3% and 37.3% lower reported peak memory. The Turbo LoRA and
FastH3 runs also outperform their LightX2V and FastVideo baselines. Tensor
profiles, video, and synchronized audio accompany the performance results.

## Insights

- FP8 and sparse attention should be designed as one path. Applying either in
  isolation leaves substantial DiT work on the table; matching the quantized
  representation to the sparse kernel allows both optimizations to contribute.
- World-model quality is not guaranteed by a generic quantization wrapper.
  The long-sequence attention layout and activation statistics make a
  hardware-specific, quality-aware implementation necessary.
- Distributed execution remains useful after the model fits on one GPU.
  MiniMax-H3 spends most of its denoising time in compute-bound DiT blocks, so
  Ulysses SP and tensor parallelism reduce per-device work and expose overlap
  opportunities. Denoising takes 167.41 seconds on one GPU, 90.90 seconds on
  two, and 49.28 seconds on four; four-GPU throughput reaches 3.40x the
  single-GPU result.

## Further reading

- [TeleFuser](https://github.com/Tele-AI/TeleFuser)
- [MiniMax-H3 model and official pipeline](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [Sol-Attn: on-the-fly attention sparsification](https://nvlabs.github.io/Sana/Sol-Attn/)
- [FastVideo MiniMax-H3 cookbook](https://haoailab.com/FastVideo/cookbook/minimax-h3/)
- [LightX2V MiniMax-H3 examples](https://github.com/ModelTC/LightX2V/tree/main/scripts/minimax_h3)
- [SGLang MiniMax-H3 cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx)
- [TorchAO quantized inference workflows](https://docs.pytorch.org/ao/stable/workflows/inference.html)
