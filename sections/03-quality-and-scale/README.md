<!--
SECTION-CONTRACT
id: 03-quality-and-scale
incoming_premise: The shared FP8 sparse path is efficient but perturbs one recurrent denoising trajectory.
outgoing_question: What does the complete path buy, and which measurements are genuinely comparable?
evidence: PR40 quality suite, PR37 overlap work, and distributed benchmark
do_not_claim: One prompt or one full-reference metric proves perceptual equivalence.
-->

# Stabilizing the Approximation, Then Scaling It

Diffusion inference repeatedly feeds model predictions into the next update.
An error that looks small at one attention boundary can reappear as texture
flicker, unstable motion, or audio drift several updates later. FP8 rounding
and sparse routing perturb the same recurrent trajectory, so quality has to be
controlled before distribution makes the path harder to inspect.

## Attention smoothing is numerical conditioning

Profiles of real MiniMax-H3 QKV captures showed that the key and value
distributions were not always centered around zero. Symmetric FP8 quantization
has no integer-style zero point to subtract at inference; it represents zero
exactly and uses a scale around zero. A non-zero tensor mean therefore spends
part of the available dynamic range representing an offset rather than useful
variation.

We exploit two attention identities:

~~~python
mu_k = K.mean(dim=sequence)
K_centered = K - mu_k

# Every logit in a query row moves by the same constant.
# Softmax cancels that constant exactly in real arithmetic.
softmax(Q @ K_centered.T) == softmax(Q @ K.T)

mu_v = V.mean(dim=sequence)
V_centered = V - mu_v

# Attention probabilities sum to one along the key sequence.
attention(Q, K, V) == attention(Q, K, V_centered) + mu_v

V_fp8, v_scale = quantize_fp8(V_centered)
rounding_mean = effective_sequence_mean(V_fp8, v_scale)
output_correction = mu_v - rounding_mean
~~~

K and V are centered before FP8 conversion. K needs no restoration because
softmax is shift-invariant. V's sequence mean is added to the attention output,
where the identity says it belongs. FP8 rounding can introduce a much smaller
mean into the centered V, so the correction subtracts that effective
per-head/channel mean as well.

The runtime kernel consumes FP8 V and its scale directly; it does not
materialize a separate BF16 "reconstructed V." The effective mean is reduced
from the FP8 representation and scale. Dequantized-V statistics are therefore
only an analytical proxy, while output error at the real attention boundary is
the stronger measurement.

This is attention-specific smoothing, not offline SmoothQuant. It does not
move activation scale into weights, calibrate a dataset, or change the trained
model. It changes the representation seen by the quantizer while preserving
the corresponding high-precision attention operation.

The first implementation exposed the cost of doing this as separate tensor
passes. On one H100, an unfused version added about 11.7% denoising time over
unsmoothed FP8 Sol-Attn. Folding centering and V restoration into the QKV
preparation/output path reduced that penalty to 2.2%. In the refined path,
KV smoothing and V correction are enabled by default, so the standard
MiniMax-H3 example does not need a quality-only collection of command-line
flags.

### A real MiniMax-H3 layer, not a synthetic distribution

To check that the argument matched the model, we captured post-QK-norm,
post-RoPE Q/K/V from the first active Sol layer. The tensor shape was
`(1, 32,626, 56, 128)`; the measurement covered four heads over the complete
32,626-token context and used 64 evenly spaced queries for FP32 dense-attention
reference math.

Centering reduced K quantization MSE from `9.380e-4` to `7.349e-4`
(-21.65%). More importantly, dense attention-output MSE fell from
`7.034e-4` to `6.459e-4` (-8.18%), while output SQNR rose from 29.23 to
29.60 dB. Q was unchanged. These are boundary diagnostics, not perceptual
scores, but they verify the mechanism on tensors the model actually produced.

The end-to-end ablation used one H100, 50 denoising steps, 1344 x 768 output,
107 frames, and the same prompt and seed. Smoothed FP8 Sol retained a 39.4%
denoising-throughput advantage over BF16 Linear + FlashAttention 4 and reduced
peak allocated memory by 42.6%. Against unsmoothed FP8 Sol, the final fused
smoothing path cost 2.1% throughput with no change in peak allocated memory.

![Performance cost and retained speedup of attention smoothing](assets/smoothing-performance.png)

Smoothing improved every reported video trajectory metric for this sample:
frame cosine moved from 0.87488 to 0.87729, PSNR from 14.695 to 14.800 dB, and
mean SSIM from 0.5464 to 0.5659. The audio metrics were mixed: the unsmoothed
trajectory was slightly closer to BF16 under the measured waveform and
spectral distances. We therefore claim a video improvement for this case, not
a universal audio improvement. Same-seed metrics are regression signals; they
do not say that one plausible diffusion sample is aesthetically superior.

The original synchronized outputs are embedded below so the numerical evidence
can be checked against motion, appearance, and sound.

<div class="video-grid video-grid-three" data-sync-group="smoothing">
  <figure>
    <figcaption>BF16 Linear + FlashAttention 4</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="bf16-quality"></video>
  </figure>
  <figure>
    <figcaption>FP8 Linear + FP8 Sol, unsmoothed</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="fp8-unsmoothed"></video>
  </figure>
  <figure>
    <figcaption>FP8 Linear + FP8 Sol, KV smoothing + V correction</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="fp8-smoothed"></video>
  </figure>
</div>

## Sparse attention still needs dense islands

Smoothing protects precision; it does not decide which attention blocks may be
skipped. Early denoising updates establish global structure, and a small set of
layers can be more sensitive to missing long-range interactions. TeleFuser
keeps an explicit dense prefix and allows selected layers to stay dense before
Sol-Attn handles the remaining work.

This is not a claim that the first update is universally special. It is a
controlled policy surface: dense steps, dense layers, threshold type, and
`tau` are recorded with each result. For the FastH3 adapter profile evaluated
below, two opening updates are dense and later updates use the validated Sol
threshold. The output pass combines the dense-prefix replacement with the
sparse correction rather than launching separate full-tensor repair work.

## Statistics must be computed after redistribution

Once smoothing is correct on one GPU, Ulysses introduces a simple but critical
ordering rule. Before all-to-all, each rank sees a sequence shard containing
all heads. After all-to-all, it sees its local heads over the complete
sequence. A key/value mean computed before redistribution is therefore not the
mean required by attention on that rank.

TeleFuser communicates QKV first, then performs the joint attention
preparation. That preserves the same centering and scaling semantics as the
single-GPU path. Ulysses reduces the local long-sequence attention problem,
while tensor parallelism can split the wide Linear/MLP work on a four-GPU
deployment. The topology is selected to match the workload: two GPUs use
`TP1 x Ulysses SP2`; four GPUs use `TP2 x Ulysses SP2`.

Communication can otherwise replace the compute that FP8 and sparsity removed.
The distributed attention path therefore supports overlapping Ulysses
communication with attention computation. This capability was refined in
subsequent TeleFuser work; at the community-blog level, the important point is
that scaling is part of the operator schedule rather than a wrapper placed
around a single-GPU kernel.

The result is one semantic path across precision, sparsity, adapters, and
parallel execution. We can now ask the only useful performance question:
against a maintained external MiniMax-H3 implementation, what does the matched
denoising region cost, what changes under multi-GPU execution, and is the
generated artifact still valid?
