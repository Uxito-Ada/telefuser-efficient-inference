<!--
SECTION-CONTRACT
id: 01-why-co-design
incoming_premise: TeleFuser needs to accelerate a high-quality multimodal DiT without treating each feature independently.
outgoing_question: What does TeleFuser add at the model, attention, and distributed levels?
evidence: MiniMax-H3 execution profile and hardware-specific kernel support
do_not_claim: General FP8 support implies an FP8 sparse-attention path.
-->

# Why FP8 and Sparse Attention Need One Design

**FP8 and sparse attention address different compute bottlenecks.** FP8 reduces
the compute and bandwidth cost of Linear/MLP layers, while Sol-Attn skips
unimportant blocks in long-sequence attention. Accelerating the full DiT
requires both in one execution path.

**The core conflict is a fixed quantized layout meeting per-forward dynamic
reordering.** Offline, online, or lazy quantization creates low-precision values
and scales per tensor, row, channel, group, or block, after which the kernel
assumes a stable layout. Sol-Attn, Top-K, and Top-P select, evict, and compact
blocks from the current Q/K/V, changing group boundaries and tile indices so
quantized values, scales, and sparse indices no longer retain their original
mapping.

**Existing quantization and sparsity implementations cannot be composed as two
independent switches.** Per-row, per-channel, and per-group scales directly lose
their mapping; a per-tensor scale remains numerically valid, but the kernel's
packing and addressing contract still breaks. Dequantizing selected blocks to
BF16 restores compatibility at the cost of the low-precision speedup.

**Low-precision kernels must also match the deployed GPU.** MXFP8 and NVFP4
implementations for newer hardware do not automatically cover existing
platforms such as SM90, so TeleFuser supplies hardware-matched low-precision
paths for those GPUs.

**Q-SPA gives reordered data, scales, sparse indices, and kernel tiles one
mapping.** Dominant Linear work and selected attention blocks stay on the
hardware-matched FP8 path, while sensitive normalization and positional
transforms remain in higher precision.
