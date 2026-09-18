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

TeleFuser can run Base H3 or merge a Turbo LoRA or FastH3 adapter before
creating the FP8 weights. Adapters can reduce denoising steps or improve output
quality for a target task. Q-SPA incorporates the effective weights into the
low-precision, sparse, and parallel path, reducing end-to-end inference cost
for the actual adapter model.

On the matched four-GPU Base H3 workload, TeleFuser completes generation in
52.27 seconds. It is 2.64× faster than LightX2V, 2.19× faster than FastVideo,
and 1.52× faster than SGLang. On Turbo LoRA, TeleFuser is 1.59× faster than
LightX2V and matches SGLang within 0.3%; on FastH3, it is 3.09× faster than
FastVideo.

These implementations and experiments lead to three practical insights for world-model inference:

- FP8 and sparse attention should be designed as one path. Applying either in
  isolation leaves substantial DiT work on the table; matching the quantized
  representation to the sparse kernel allows both optimizations to contribute.
- World-model quality is not guaranteed by a generic quantization wrapper.
  The long-sequence attention layout and activation statistics make a
  hardware-specific, quality-aware implementation necessary.
- A world-model request combines conditioning and reasoning, long-sequence
  video/audio denoising, and decoding; whether its weights fit on one GPU does
  not capture that compute pressure. Ulysses SP, tensor parallelism, and
  communication overlap shorten the full generation path. The quantized path
  preserves near-linear scaling across the tested GPU counts while lowering
  per-GPU memory, leaving room for larger generation requests.

## Further reading

- [TeleFuser](https://github.com/Tele-AI/TeleFuser)
- [MiniMax-H3 model and official pipeline](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [Sol-Attn: on-the-fly attention sparsification](https://nvlabs.github.io/Sana/Sol-Attn/)
- [FastVideo MiniMax-H3 cookbook](https://haoailab.com/FastVideo/cookbook/minimax-h3/)
- [LightX2V MiniMax-H3 examples](https://github.com/ModelTC/LightX2V/tree/main/scripts/minimax_h3)
- [SGLang MiniMax-H3 cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx)
- [TorchAO quantized inference workflows](https://docs.pytorch.org/ao/stable/workflows/inference.html)
