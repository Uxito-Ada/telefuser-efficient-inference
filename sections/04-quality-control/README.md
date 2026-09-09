<!--
SECTION-CONTRACT
id: 04-quality-control
incoming_premise: The unified path introduces FP8 rounding and sparse approximation into one denoising trajectory.
outgoing_question: Can the same numerical contract remain valid after Ulysses redistributes tokens and heads?
evidence: Real H3 tensor error, new prompt-suite media, and fused-boundary microbenchmarks.
do_not_claim: Trajectory similarity metrics are not absolute perceptual scores.
-->

# 4. Protecting Quality Under Compound Approximation

The unified kernel makes quantization and sparsity fast enough to matter
together. It also removes the comforting fiction that their errors can be
validated independently. A rounded Q or K value changes both the exact
attention logits and the proxy used to choose exact blocks. A sparse routing
decision changes which rounded values receive full computation. The resulting
output becomes the input to the next DiT layer and, later, the next denoising
update.

This feedback explains a common failure mode in generative optimization: every
kernel returns finite tensors, a few sampled operator errors look small, but
the final video loses a face, changes an object, or develops unstable motion.
Numerical validity is necessary and insufficient.

We control this compound error at three levels: where approximation is allowed,
how tensors are represented before rounding, and how the added protection is
executed.

## Dense islands allocate the error budget

Sensitivity is not uniform across the graph. Early denoising updates establish
global structure from noisy latents. Some transformer layers have a larger
effect on conditioning or the final representation. MiniMax-H3 also carries
conditioning tokens whose exact contribution should not be mixed with an
aggressive route chosen for the much larger visual sequence.

The final policy therefore keeps explicit dense islands:

- an initial set of denoising updates uses dense attention;
- selected early or sensitive layers remain dense;
- the conditioning prefix is recomputed through the exact path;
- Sol's \(\tau\) controls the route threshold for the remaining region.

These controls are related, not four arbitrary tuning knobs. Dense steps
protect trajectory formation, dense layers protect structurally sensitive
transformations, prefix replacement protects conditioning, and \(\tau\)
sets the per-input exact-attention budget elsewhere. The quality sweep will
select the least expensive policy that passes the same prompt-suite gate as the
external BF16 reference. Its final values remain
**TBD--new experiment required**.

Consider the running fox example. The first updates establish the animal,
camera path, and snow field, so they receive dense computation. Later updates
mostly refine texture and local motion, where dynamic routing can spend exact
blocks around the moving subject while summarizing less influential context.
For a dialogue example, the dense prefix protects text and audio conditioning
even if the visual route changes with the speaker.

## Centering K without changing exact attention

Dense islands limit where error enters; they do not improve the FP8
representation inside the allowed region. Real post-normalization,
post-RoPE H3 tensors can contain offsets and outliers that consume E4M3 dynamic
range. We reshape that distribution using identities of the attention
function.

For one head, let \(\mu_K\) be the sequence mean of K. Then

\[
\operatorname{softmax}(Q(K-\mu_K)^T)
=
\operatorname{softmax}(QK^T-Q\mu_K^T)
=
\operatorname{softmax}(QK^T).
\]

For each query, \(Q\mu_K^T\) is the same scalar shift across all key logits.
Softmax removes that shift. K can therefore be centered in FP32 before E4M3
rounding without changing full-precision attention. The transform is useful
because it spends the finite FP8 grid on variation around the mean rather than
on a removable offset.

This is attention smoothing, not SmoothQuant. No activation scale is migrated
into a Linear weight, and the equivalence follows from softmax invariance.

## Centering V requires an explicit correction

Let \(P=\operatorname{softmax}(QK^T)\) and \(\mu_V\) be the sequence mean of
V. Since each row of P sums to one,

\[
P(V-\mu_V)+\mu_V = PV.
\]

We center V before quantization and restore its mean after attention. E4M3
rounding can leave a small nonzero mean in the reconstructed centered tensor,
so the implementation measures that residual per head and channel and corrects
it at the BF16 output boundary.

K centering reduces logit error; V centering and correction reduce output bias.
The two transformations address different failure mechanisms and should be
measured separately on captured H3 tensors. The draft reserves the following
new-data table rather than importing an earlier result:

| Boundary | FP8 Sol | FP8 Sol + smoothing | Change |
|---|---:|---:|---:|
| K reconstruction MSE | TBD | TBD | TBD |
| V reconstruction MSE | TBD | TBD | TBD |
| V residual mean bias | TBD | TBD | TBD |
| Dense attention-output MSE | TBD | TBD | TBD |
| Attention-output cosine | TBD | TBD | TBD |

## Exact math still needs efficient execution

An algebraically neutral transform can still make the system slower. Exact K/V
means require reductions over the live sequence, and a naive implementation
adds separate centering, quantization, correction, and prefix-merge kernels.
On a 32,626-token tensor, those launches and full-tensor memory passes are not
free.

The implementation computes exact statistics in FP32, fuses centering into FP8
preparation, and combines output correction with dense-prefix replacement.
Tests compare the fused path against an unfused reference before performance is
measured. The new boundary latency and its contribution to end-to-end time are
**TBD--new experiment required**.

We now have a quality-aware contract for one local attention problem: which
regions stay exact, how the approximate region is represented, and how its
bias is corrected. Sequence parallelism changes that local problem by moving
tokens and heads between GPUs. Preserving the contract requires placing every
operation relative to that redistribution.
