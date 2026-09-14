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

On the matched four-GPU Base H3 workload, TeleFuser is 2.64x faster in
generation and uses 40.3% less representative peak memory than LightX2V while
both run `TP2 x Ulysses SP2`. The adapter evaluations show that the same runtime
also outperforms the working LightX2V Turbo and FastVideo FastH3 baselines. The
accompanying tensor profiles and generated media cover numerical error, final
video, and synchronized audio rather than performance alone.

## Further reading

- [TeleFuser](https://github.com/Tele-AI/TeleFuser)
- [MiniMax-H3 model and official pipeline](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [Sol-Attn: on-the-fly attention sparsification](https://nvlabs.github.io/Sana/Sol-Attn/)
- [FastVideo MiniMax-H3 cookbook](https://haoailab.com/FastVideo/cookbook/minimax-h3/)
- [LightX2V MiniMax-H3 examples](https://github.com/ModelTC/LightX2V/tree/main/scripts/minimax_h3)
- [TorchAO quantized inference workflows](https://docs.pytorch.org/ao/stable/workflows/inference.html)
