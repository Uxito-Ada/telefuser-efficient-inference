<!--
SECTION-CONTRACT
id: 05-lessons
incoming_premise: Matched external and diagnostic experiments expose both performance and quality.
outgoing_question: none
evidence: final benchmark and quality records
do_not_claim: Portability beyond the tested MiniMax-H3 and H100 contract.
-->

# What Carries Beyond This Benchmark

The most reusable lesson is that efficient inference features are not vertical
options. Quantization changes the representation consumed by attention.
Sparsity changes the set of values accumulated by that attention. Sequence
parallelism changes the tensor over which scales and means are defined.
Adapters change the weights that should be quantized. Treating any one of
these as an isolated toggle leaves the interfaces to chance.

Three design rules emerged from the MiniMax-H3 work.

**Own the boundary where representations change.** The FP8 Sol-Attn path works
because quantization, scale metadata, sparse routing, and the H100 tile layout
have one owner. General framework support is still valuable, but it cannot
substitute for the missing kernel at a hardware-specific boundary.

**Use mathematical invariants as quality tools.** Centering K and V is useful
because attention provides identities that preserve the intended
higher-precision operation. This gives us a reason to expect quality
improvement and a precise place to measure it. It is stronger than adding an
arbitrary correction after degraded videos appear.

**Benchmark the artifact, not only the kernel.** A world-model request includes
weight preparation, denoising, decoding, audio, and media output. Kernel
throughput explains where time moved; end-to-end latency, peak per-GPU memory,
and playable synchronized output determine whether the system improved.
Warm-up and compilation must be separated, and failed output must be excluded
rather than assigned an impressive speed.

The scope is intentionally narrow. This implementation targets MiniMax-H3 on
NVIDIA H100. The FP8 Sol kernel is SM90-specific, the selected sparse policy is
validated for the tested profiles, and the best parallel topology can change
with interconnect, duration, or concurrency. Full-reference video metrics also
measure trajectory agreement, not human preference; they are evidence for
regressions, not a replacement for viewing and listening.

Within that scope, the work turns FP8 Linear, FP8 sparse attention, attention
smoothing, adapters, and Ulysses from separate demonstrations into one
deployable execution path. The broader pattern is the real result: reduce work
aggressively, preserve the model's numerical contracts deliberately, and
measure both speed and meaning at the end.

## Further reading

- [MiniMax-H3 model and official pipeline](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [Sol-Attn: on-the-fly attention sparsification](https://nvlabs.github.io/Sana/Sol-Attn/)
- [FastVideo MiniMax-H3 cookbook](https://haoailab.com/FastVideo/cookbook/minimax-h3/)
- [TorchAO quantized inference workflows](https://docs.pytorch.org/ao/stable/workflows/inference.html)
- [TeleFuser](https://github.com/Tele-AI/TeleFuser)
