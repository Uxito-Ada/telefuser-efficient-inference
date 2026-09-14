<!--
SECTION-CONTRACT
id: 01-why-co-design
incoming_premise: TeleFuser needs to accelerate a high-quality multimodal DiT without treating each feature independently.
outgoing_question: What does TeleFuser add at the model, attention, and distributed levels?
evidence: MiniMax-H3 execution profile and hardware-specific kernel support
do_not_claim: General FP8 support implies an FP8 sparse-attention path.
-->

# Why FP8 and Sparse Attention Need One Design

World models pay for quality with compute. In MiniMax-H3, large projections and
MLPs make the DiT compute-heavy, while visual, audio, and conditioning tokens
make attention expensive. No single optimization addresses both costs.

| Technique | Primary benefit | Remaining cost |
|---|---|---|
| FP8 Linear | cheaper projections and MLPs | long-sequence attention |
| FP8 attention | lower QK/PV precision and bandwidth | dense token pairs |
| Sol-Attn | fewer attention blocks | dense transformer layers |
| Ulysses SP | lower per-GPU sequence state | communication between GPUs |

The difficulty is not simply that separate libraries expose separate APIs.
Quantization scales are defined over a fixed partition of the tensor: per
tensor, row, group, or block. Sol-Attn, Top-K, and Top-P sparsity select and
evict attention blocks at runtime. The surviving K/V blocks are gathered,
compacted, or re-indexed, so their physical tiles no longer line up with the
scale groups assumed by a dense quantized kernel. Dequantizing the selected
blocks back to BF16 restores compatibility but gives up much of the intended
bandwidth and compute benefit.

Sequence parallelism adds another layout transformation. Tokens and attention
statistics are split across ranks, while sparse selection still needs a
consistent global meaning. Quantization metadata, sparse block indices, and
the per-rank tensor layout therefore have to be designed together.

Hardware support sharpens the issue. Low-precision formats and kernels do not
have uniform coverage across GPU generations: a path optimized for newer
hardware does not automatically provide an equivalent implementation on SM90.
A framework-level "FP8 enabled" switch therefore says little about whether
dense DiT compute and sparse attention can remain in low precision together.

Q-SPA makes sparse routing, quantization metadata, and distributed attention
share one layout contract. Sensitive normalization and positional transforms
remain in higher precision, while the dominant Linear work and selected
attention blocks stay on the hardware-matched FP8 path.
