<!--
SECTION-CONTRACT
id: 05-lessons
incoming_premise: TeleFuser's combined path improves performance while retaining measurable output quality.
outgoing_question: none
evidence: final benchmark and quality records
do_not_claim: Portability beyond the tested MiniMax-H3 and H100 contract.
-->

# A Unified Efficient-Inference Path in TeleFuser

This work expands TeleFuser from supporting MiniMax-H3 execution to optimizing
the complete DiT path on H100. The framework now combines:

- FP8 Linear and FP8 Sol sparse attention;
- quality-aware FP8 attention;
- configurable dense regions for quality-sensitive computation;
- base, Turbo LoRA, and FastH3-style adapters;
- Ulysses sequence parallelism, tensor parallelism, and communication overlap.

The key outcome is composition. Distilled adapters reduce how many DiT
evaluations are needed; FP8 reduces the cost of dense transformer work; Sol-Attn
reduces attention work; smoothing protects the resulting trajectory; and
Ulysses carries the same path to multiple GPUs.

On the matched four-H100 Base H3 workload, TeleFuser is 2.64x faster in
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
