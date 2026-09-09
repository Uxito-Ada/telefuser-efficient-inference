<!--
SECTION-CONTRACT
id: 05-lessons
incoming_premise: One matched experiment measures the complete system.
outgoing_question: none
evidence: final benchmark and quality records
do_not_claim: Portability beyond the tested MiniMax-H3 and H100 contract.
-->

# What We Learned

Efficient world-model inference is less about collecting optimization flags
than about preserving contracts between them. FP8 helps the transformer-heavy
parts of MiniMax-H3; sparse attention helps the long-sequence part. The useful
speedup appears only when both operate on the same tensor layout and avoid
paying for duplicate quantization, conversion, and communication.

Quality techniques belong in that co-design. Attention smoothing, dense
islands, and an explicit sparsity threshold are small pieces of math, but they
decide whether a faster denoising loop still produces stable motion, detail,
and synchronized sound. Multi-GPU execution and adapters then have to preserve
those same statistics and effective weights.

This work currently targets MiniMax-H3 on NVIDIA H100. Its custom FP8 Sol-Attn
kernel is SM90-specific, and the chosen TP2 x Ulysses SP2 topology is a measured
deployment choice rather than a universal replacement for Ring or other
parallel layouts. The broader lesson is portable: optimize the end-to-end
execution graph, validate the generated artifact, and treat quality as a hard
constraint rather than a screenshot selected after benchmarking.

The final measured conclusion is **TBD--new experiment required**.

## Further reading

- [MiniMax-H3 model and official pipeline](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [Sol-Attn: on-the-fly attention sparsification](https://nvlabs.github.io/Sana/Sol-Attn/)
- [FastVideo MiniMax-H3 cookbook](https://haoailab.com/FastVideo/cookbook/minimax-h3/)
- [TorchAO quantized inference workflows](https://docs.pytorch.org/ao/stable/workflows/inference.html)
- [TeleFuser](https://github.com/Tele-AI/TeleFuser)
