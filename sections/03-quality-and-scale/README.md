<!--
SECTION-CONTRACT
id: 03-quality-and-scale
incoming_premise: The shared FP8 sparse path is faster but perturbs a recurrent denoising trajectory.
outgoing_question: What performance and quality does the complete path deliver?
evidence: PR40 quality suite
do_not_claim: One prompt or one full-reference metric proves perceptual equivalence.
-->

# Quality-aware FP8

Diffusion models feed each prediction into the next denoising update, so local
FP8 error can accumulate across the trajectory. TeleFuser therefore includes
attention smoothing in its FP8 path rather than treating quality as a separate
post-processing step.

Profiling MiniMax-H3 layers showed that some K/V tensors have clearly non-zero
mean distributions. TeleFuser centralizes K/V before FP8 attention and restores
the equivalent shift in the output. This attention-specific asymmetric
quantization keeps more of the useful FP8 range without changing the model's
interface.

Centralization and output correction are fused into FP8 Sol-Attn and enabled by
default. The end-to-end quality and overhead measurements appear after the main
performance evaluation.
