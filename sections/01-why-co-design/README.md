<!--
SECTION-CONTRACT
id: 01-why-co-design
incoming_premise: MiniMax-H3 quality is the product contract, not a variable to optimize away.
outgoing_question: What execution path lets FP8 and sparse attention share work?
evidence: operator profile from the new four-GPU experiment
do_not_claim: General FP8 support implies a compatible sparse-attention kernel.
-->

# Why FP8 and Sparsity Have to Be Co-designed

MiniMax-H3 packs text conditioning, video latents, and audio latents into the
same denoising process. Richer outputs therefore increase two different kinds
of work.

First, projections, MLPs, and conditioning layers move large weight matrices
through every transformer block. FP8 is a natural fit: H100 tensor cores can
execute lower-precision matrix multiplication at much higher throughput while
storing the dominant weights in half the bytes of BF16.

Second, attention operates over a long multimodal sequence. Reducing precision
does not change how many token pairs dense attention evaluates. Sol-Attn attacks
that orthogonal dimension by selecting important blocks online and correcting
for the skipped tail. The result is less attention work without retraining the
model.

Either technique alone leaves a major bottleneck untouched. Using both should
be complementary, but a framework cannot get there by independently enabling
an FP8 Linear wrapper and a sparse-attention backend. A typical path quantizes
Q, K, and V for the three projections, converts them back to BF16 for
normalization and rotary embedding, and then quantizes them again for
low-precision attention. Under sequence parallelism, doing this before
all-to-all also gives each rank scales for a tensor that is about to be
redistributed.

Hardware support adds another boundary. General FP8 Linear inference is
available on H100, but an optimized format or kernel does not automatically
cover sparse QK/PV attention, its routing metadata, or its scale layout. Some
newer MXFP8 and NVFP4 routes target Blackwell and cannot simply be projected
onto SM90. We therefore needed a native H100 path whose owner understands both
quantization and sparse attention.

That path is the center of this work: quantize once at the point where the
attention tensor has its final layout, then keep routing and matrix
multiplication inside one Sol-Attn implementation.
