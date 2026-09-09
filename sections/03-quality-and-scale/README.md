<!--
SECTION-CONTRACT
id: 03-quality-and-scale
incoming_premise: One FP8 sparse path removes redundant work but compounds two approximations.
outgoing_question: Does the complete system beat a maintained external runtime?
evidence: quality-suite metrics and four-GPU distributed profile
do_not_claim: One prompt or pixel metric proves perceptual equivalence.
-->

# Keeping Quality While Scaling Out

An optimization is not useful if it accelerates the wrong trajectory. Video
diffusion is especially unforgiving: small errors are fed into later denoising
updates, where they can appear as texture flicker, broken motion, or unstable
audio. FP8 rounding and sparse routing perturb that same trajectory, so quality
control has to be part of the execution path rather than a post-processing
step.

We use two lightweight controls.

The first is **attention smoothing**. K can be centered without changing
softmax attention because subtracting the same key mean shifts every logit in a
row by the same constant. V is centered before quantization and its mean is
added back to the output. Both operations reduce the range that FP8 must
represent while preserving the corresponding high-precision attention
identity. This is closer to numerical conditioning inside attention than to
offline SmoothQuant.

The second is **selective dense computation**. Early denoising updates and a
small set of sensitive layers can stay dense, while Sol-Attn handles the rest.
The sparsity threshold remains an explicit quality/performance control rather
than a hidden constant. In the final profile we use one dense opening update,
then the validated Sol threshold for the remaining transformer evaluations.

The same numerical contract must survive distribution. TeleFuser communicates
Q, K, and V first and performs attention preparation after Ulysses has given
each rank complete sequence context for its local heads. This avoids deriving
scales and means from the wrong shard. The four-GPU deployment combines TP2
with Ulysses SP2: tensor parallelism covers wide Linear/MLP work, while sequence
parallelism covers long attention. TeleFuser also supports overlapping Ulysses
communication with attention computation.

Finally, optimized weights remain adapter-aware. Turbo and FastH3 adapters are
merged before quantization, with higher-precision accumulation for the weight
update. FP8 caches are then created inside the worker that owns each GPU. These
choices make the same optimized path usable for the base model and released
LoRA variants without keeping stale weights or parent-process CUDA state.

The result is one deployable configuration rather than separate FP8, sparse,
parallel, and adapter demos. The remaining question is end-to-end: does it
outperform an external H3 runtime while its generated video and audio remain
useful?
