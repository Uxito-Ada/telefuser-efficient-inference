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

The natural answer is to combine them, but existing features do not
automatically compose. General quantization libraries often accelerate Linear
layers without supplying the sparse-attention kernel required by a particular
GPU generation. Sparse implementations may still expect BF16 QKV, erasing part
of the gain through format conversion. Sequence parallelism changes how
attention data is distributed, and adapters change the effective model weights
that low-precision execution must represent.

Hardware support sharpens the issue. Low-precision formats and kernels do not
have uniform coverage across GPU generations: a path optimized for newer
hardware does not automatically provide an equivalent implementation on SM90.
A framework-level "FP8 enabled" switch therefore says little about whether
dense DiT compute and sparse attention can remain in low precision together.

TeleFuser addresses the combination as one system feature. Sensitive
normalization and positional transforms remain in higher precision, while the
dominant Linear work and Sol attention run through a hardware-matched FP8 path.
Sparse routing, QKV precision, quality correction, adapter loading, and
distributed execution share the same model contract. That common contract is
what lets the individual optimizations add up instead of interfering with one
another.
