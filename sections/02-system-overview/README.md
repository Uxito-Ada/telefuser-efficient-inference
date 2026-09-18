<!--
SECTION-CONTRACT
id: 02-system-overview
incoming_premise: FP8 and sparsity need one hardware-aware model contract.
outgoing_question: How does TeleFuser keep the combined approximation stable?
evidence: TeleFuser PRs 16, 25, 30, 35, and 44
do_not_claim: TeleFuser invented Sol-Attn or makes all H3 operators FP8.
-->

# Q-SPA in TeleFuser {#q-spa-system}

The MiniMax-H3 work extends TeleFuser at three levels: dense DiT compute,
long-sequence attention, and model-scale execution. Users select one supported
inference profile that brings these capabilities together.

## Layout-aware FP8 sparse attention

TeleFuser applies FP8 to the DiT projections and MLPs, then carries the same
precision into Sol-Attn. Sparse block indices, quantization scales, and the
tiles consumed by the attention kernel are kept in the same layout contract.
Selected QKV blocks can therefore remain in FP8 instead of returning to a BF16
attention backend.

**TeleFuser also optimizes Sol-Attn itself.** QK and PV GEMMs execute in FP8,
dequantization is fused into attention, and Two-way KV splitting raises SM
utilization for sparse shapes. The combined FP8 and sparse path also restores
the true route length after tile-aligned Tail padding, so padded tokens never
enter valid outputs. Dense windows, dense layers, threshold modes, and sparsity
strength remain configurable; the MiniMax-H3 defaults are tuned for immediate
use.

## Distilled and quality adapters

MiniMax-H3 is used both as a base model and with acceleration or style adapters.
TeleFuser supports the official Turbo LoRA path as well as FastH3-style LoRA
and dense adapters. Adapter changes are incorporated into the effective model
before its reusable FP8 representation is created, ensuring that low-precision
execution represents the requested model rather than the base checkpoint.

Distilled adapters such as Turbo LoRA can reduce denoising steps, while other
adapters target output quality for a specific task. Q-SPA incorporates the
adapter's effective weights into the FP8, sparse, and parallel path, reducing
end-to-end inference cost for the requested model rather than optimizing only
the Base checkpoint.

## Distributed execution for long sequences

TeleFuser does more than enable Ulysses SP for MiniMax-H3. It adapts the
multi-GPU path to FP8, Sol-Attn, and the model's video-token layout. FP8
quantization and Sol routing run on the rank-local attention view established
by Ulysses All-to-All, keeping quantization scales and sparse blocks aligned
with the actual compute layout. Three-dimensional video tokens are reordered
before sequence partitioning; a scalar diffusion timestep remains replicated,
while per-token timesteps follow the token shards. These rules preserve the
model semantics across single- and multi-GPU execution.

TeleFuser also adds sequence-parallel kernels and overlaps Ulysses communication
with attention compute. The resulting two- and four-GPU MiniMax-H3 path,
including `TP2 × Ulysses SP2`, carries FP8 and sparse-attention gains into
end-to-end distributed inference.
