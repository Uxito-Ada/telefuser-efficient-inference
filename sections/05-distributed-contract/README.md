<!--
SECTION-CONTRACT
id: 05-distributed-contract
incoming_premise: Quality controls are defined over the complete sequence consumed by one attention problem.
outgoing_question: How can mutable adapter weights and FP8 caches remain correct across spawned workers?
evidence: Distributed layout tests and the new 4xH100 communication/compute trace.
do_not_claim: Four-GPU speedup is unknown until the new experiment is complete.
-->

# 5. Preserving the Numerical Contract Across GPUs

The quality controls in the previous section use sequence statistics. K/V means
are reduced over tokens; Q/K scales correspond to 64-token blocks; Sol routing
compares blocks across the attention context. Those definitions are
unambiguous on one GPU. Sequence parallelism changes the tensor that is local
to each rank, so it also changes where those definitions are valid.

MiniMax-H3 needs multiple GPUs for more than aggregate capacity. A 32,626-token
attention problem exposes enough computation to benefit from distributing the
sequence, but only if communication does not introduce redundant transforms or
change sparse semantics. We use Ulysses because its post-collective layout
matches the local problem expected by the current Sol kernel.

## Ulysses changes ownership before attention

Let the logical Q/K/V shape be \([B,S,H,D]\) and let \(p\) be the sequence-
parallel degree. Around the all-to-all, ownership changes as follows:

| Phase | Tokens on one rank | Heads on one rank | What the rank can compute |
|---|---:|---:|---|
| Before all-to-all | \(S/p\) | \(H\) | projections and token-local transforms |
| After all-to-all | \(S\) | \(H/p\) | complete attention for local heads |

Quantizing Q/K/V before the collective would attach scales to a partial
sequence across all heads. The all-to-all then rearranges those values into a
complete sequence for fewer heads. Block boundaries, sequence means, and the
head set no longer match the domain that produced the scales.

TeleFuser therefore orders the path as:

```text
local-token QKV projection
  -> QK norm and RoPE
  -> Ulysses all-to-all
  -> exact K/V statistics over the complete local-head sequence
  -> smoothing and FP8 preparation
  -> local FP8 Sol attention
  -> inverse all-to-all
```

The ordering follows the final-consumer rule established in Section 2:
communication determines ownership first; quantization and approximation are
defined second. FP8 preparation remains device-local, so quantized tensors do
not need a new distributed scale-exchange protocol.

## Why Ulysses, rather than Ring, is the first composition

Ring attention keeps sequence shards local and circulates K/V blocks. Dense
online attention can merge partial maximum, exponential sum, and output state
as blocks arrive. Native Sol adds another distributed state: dynamic route
decisions and summary corrections must remain consistent across ranks.

The current FP8 Sol kernel expects the complete sequence for its local heads.
Ulysses produces exactly that contract after one all-to-all, whereas a Ring
implementation would require a new distributed routing and log-sum-exp merge.
Choosing Ulysses is therefore not a claim that it is universally faster. It is
the parallel layout that preserves the already validated sparse numerical
problem without inventing an untested distributed approximation.

TeleFuser also supports overlapping Ulysses communication with attention
computation to reduce exposed distributed overhead.

## Not every sequence-shaped input is sharded

Parallel wrappers often fail at inputs smaller than Q/K/V. MiniMax-H3 carries
timestep and conditioning information in more than one shape:

- a scalar or batch-level timestep controls every token and is replicated;
- a token-shaped timestep or condition has a real sequence axis and follows
  the sequence partition;
- prefix metadata describes logical positions and must account for padding.

Blindly slicing a scalar timestep either produces an empty tensor or changes
the denoising state across ranks. Conversely, refusing to slice a token-shaped
condition misaligns it with the local latent. TeleFuser switches on tensor
semantics and shape rather than applying one "sequence parallel" operation to
every input.

The same care applies when \(S\) is not divisible by \(p\). Communication may
pad the physical tensor, but scale reductions, K/V means, routing statistics,
and dense-prefix replacement must use the valid logical sequence. Otherwise
padding zeros become part of an allegedly exact statistic.

## The four-GPU topology is part of the final method

The flagship configuration uses a two-dimensional four-GPU topology:
TP2 x Ulysses SP2. Tensor parallelism partitions the wide projection and
feed-forward work, while each Ulysses group redistributes sequence tokens into
complete attention contexts for its local heads. The topology increases
effective bandwidth for both dominant DiT operator families instead of forcing
all four devices into a single parallel dimension.

The external FastVideo baseline uses its maintained SP4 configuration. Both
systems receive the same four H100 GPUs; each framework retains its intended
distributed layout. The new experiment will separate:

- all-to-all time;
- QKV preparation and smoothing time;
- FP8 Sol attention time;
- total denoising time; and
- full encode-to-MP4 latency.

The resulting four-GPU throughput and communication fraction are
**TBD--new experiment required**. No earlier TP/SP measurement is substituted.

With this ordering, precision, sparsity, and distribution describe the same
tensor. The remaining assumption is that model weights are static. Fast
MiniMax-H3 deployments violate that assumption: Turbo and FastH3 are delivered
as adapters that change the effective Linear weights. A cached FP8 model must
therefore treat weight mutation as part of its distributed lifecycle.
