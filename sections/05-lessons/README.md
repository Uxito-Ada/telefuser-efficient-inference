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

On the matched FastH3 workload, TeleFuser delivers 25.7% lower denoising time,
34.6% higher DiT throughput, and 14.3% lower peak GPU memory than the FastVideo
BF16 + FA4 baseline. The accompanying tensor profiles and generated media show
how the quality controls affect both numerical error and visible output.

## Further reading

- [TeleFuser](https://github.com/Tele-AI/TeleFuser)
- [MiniMax-H3 model and official pipeline](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [Sol-Attn: on-the-fly attention sparsification](https://nvlabs.github.io/Sana/Sol-Attn/)
- [FastVideo MiniMax-H3 cookbook](https://haoailab.com/FastVideo/cookbook/minimax-h3/)
- [TorchAO quantized inference workflows](https://docs.pytorch.org/ao/stable/workflows/inference.html)
