<!--
SECTION-CONTRACT
id: 01-quality-cost
incoming_premise: The article claims that faithful world-model inference is compute-bound.
outgoing_question: If both bit width and executed work must fall, why not enable FP8 and sparsity independently?
evidence: Model architecture, representative live tensor shapes, and the new profiler trace.
do_not_claim: Do not imply that quality necessarily improves with model size or step count.
-->

# 1. Quality Has a Computational Cost

Consider a five-second scene: a fox runs through fresh snow at dawn, powder
moves under its paws, the camera tracks the animal, and footsteps remain
synchronized with the motion. A useful world model must do more than render a
sharp first frame. Identity must survive motion; occluded geometry must
reappear plausibly; illumination and camera motion must agree; and the audio
track must follow the visible event. These constraints are a form of implicit
world reasoning expressed through pixels and sound.

[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) generates video and
stereo audio in one diffusion system. Its DiT repeatedly updates a long latent
sequence conditioned on text and optional multimodal inputs. At the 768p
configuration used in this study, a representative self-attention boundary has
32,626 tokens, 56 heads, and head dimension 128. One denoising request invokes
the same large transformer multiple times before video and audio can be
decoded.

That repetition concentrates cost in two matrix-multiplication families:

\[
\begin{aligned}
XW &\quad &&\text{for QKV, output, and feed-forward projections},\\
\operatorname{softmax}(QK^T/\sqrt{d})V
   &&&\text{for attention}.
\end{aligned}
\]

The first scales with model width and parameter count. Dense attention scales
quadratically with sequence length. Increasing temporal duration adds tokens
and also increases the area of the attention matrix; increasing spatial
resolution does the same in two dimensions. The DiT is therefore
compute-bound precisely where richer motion and longer context demand more
work.

## Why the obvious reductions change the problem

There are easy ways to lower latency if the output contract is negotiable:
generate fewer frames, lower the resolution, shorten the clip, use fewer
denoising updates, or replace the base model with a smaller one. Those choices
can be valuable products, but they do not answer the question in this article.
They purchase speed by changing what must be generated.

Our contract fixes the model family, task, output shape, duration, frame rate,
and sampling work before comparing systems. Under that constraint, efficiency
has to come from executing the same request more economically.

The two strongest levers are complementary:

- **Low precision** reduces bytes moved and increases tensor-core throughput
  for work that still executes.
- **Sparse attention** reduces the number of QK and PV blocks that need exact
  evaluation.

Neither lever covers the full DiT. Weight/activation quantization accelerates
the many Linear layers but does not automatically change attention precision.
Sparse attention removes attention work but leaves projection and feed-forward
GEMMs dense. Long-sequence world-model inference needs both fewer bits and
fewer operations.

## The quality constraint is active, not decorative

Both levers also perturb the computation. FP8 rounds values to a finite grid
whose error depends on scale selection and outliers. Dynamic sparse attention
uses a data-dependent proxy to decide which blocks receive exact computation.
The error is therefore not constant across prompts, layers, or denoising
timesteps.

This matters more in a diffusion trajectory than in a single feed-forward
classification pass. A perturbation at an early update changes the latent
consumed by every later update. It may leave every tensor finite while altering
a face, motion direction, object count, or audio event. Efficiency and quality
cannot be evaluated as separate post-processing steps; quality defines the
numerical budget within which acceleration is allowed.

The resulting target is more demanding than either technique alone:

> execute MiniMax-H3 with low-precision dense operators and dynamically sparse
> attention, while keeping sensitive computation exact and preserving the
> distributed semantics of the reference model.

That target appears straightforward at the algorithm level. At the framework
boundary, however, the output contract of an FP8 Linear layer does not match
the input contract of a sparse attention kernel. The next section explains why
two individually useful optimizations fail to compose.
