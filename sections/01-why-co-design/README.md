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
make attention expensive. FP8 reduces the cost of projections and MLPs;
Sol-Attn skips unimportant attention blocks. Further acceleration requires both
to operate in one execution path.

The difficulty is not simply that separate libraries expose separate APIs.
Quantization scales are defined over a fixed partition of the tensor: per
tensor, row, group, or block. Sol-Attn, Top-K, and Top-P sparsity select and
evict attention blocks at runtime. The surviving K/V blocks are gathered,
compacted, or re-indexed, so their physical tiles no longer line up with the
scale groups assumed by a dense quantized kernel. Dequantizing the selected
blocks back to BF16 restores compatibility but gives up much of the intended
bandwidth and compute benefit.

Hardware support sharpens the issue. Low-precision formats and kernels do not
cover every GPU generation. For example, MXFP8 and NVFP4 implementations built
for newer hardware do not automatically support widely deployed platforms such
as SM90. New accelerators do not make the installed base disappear. TeleFuser
builds hardware-matched low-precision paths so existing platforms can run the
latest world models efficiently.

Q-SPA gives sparse indices, quantization scales, and kernel tiles a compatible
layout. Sensitive normalization and positional transforms remain in higher
precision, while the dominant Linear work and selected attention blocks stay
on the hardware-matched FP8 path.
