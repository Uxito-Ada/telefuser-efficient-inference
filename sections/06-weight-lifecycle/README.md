<!--
SECTION-CONTRACT
id: 06-weight-lifecycle
incoming_premise: The distributed FP8 path is correct only while effective weights remain unchanged.
outgoing_question: Does the complete adapter-aware four-GPU system beat an external implementation at comparable quality?
evidence: Adapter mapping tests, cache lifecycle tests, and process-spawn tests.
do_not_claim: Dense FastH3 support implies support for learned VSA replacement gates.
-->

# 6. Treating Optimized Weights as Mutable State

The distributed path now has a precise tensor contract, but its weight cache
introduces a new source of state. Online FP8 converts a BF16 checkpoint into a
device-efficient representation. An adapter then changes the effective model
from \(W\) to \(W+\Delta W\). If the adapter is applied after FP8
materialization, the optimized GEMM continues to read a stale copy of \(W\)
even though the Python model appears to contain the adapter.

This failure is especially dangerous for generation. Shapes remain valid,
inference completes, and the output can look like a plausible sample from the
base model. No exception proves that the intended distilled model was never
executed.

## Merge semantics precede precision

The only reliable source of truth is the fully adapted high-precision weight.
TeleFuser therefore uses the following lifecycle:

```text
load BF16 base weight on CPU
  -> identify adapter format and scaling convention
  -> compute adapter update in FP32
  -> merge into the BF16 source
  -> move or shard the final weight for its worker
  -> materialize cached E4M3 weight and scale
  -> release the superseded BF16 source
```

The ordering avoids an update to already rounded values and guarantees that
every FP8 cache represents the requested model:

\[
W_{\mathrm{FP8}} =
Q_{\mathrm{E4M3}}\left(
  W_{\mathrm{BF16}} + \lambda\Delta W_{\mathrm{adapter}}
\right).
\]

The adapter computation uses FP32 accumulation before writing the BF16 source.
Quantizing each low-rank term independently and then summing in FP8 would add a
second, unnecessary approximation before denoising begins.

## Adapter files encode model semantics

MiniMax-H3 Turbo and FastH3 illustrate why a loader cannot infer behavior from
a filename ending in `.safetensors`. A conventional LoRA contains low-rank A/B
pairs plus an alpha/rank convention. The FastH3 dense adapter combines
low-rank updates with exact residual tensors and uses its published scaling
contract. Missing the residuals or applying the wrong implicit scale produces
a different model.

Some FastH3 VSA releases additionally contain learned compression-gate
replacements. Those gates define a trained sparse model; they are not
equivalent to the training-free Sol route. Until the runtime implements that
gate contract, it must reject the adapter rather than apply only the familiar
low-rank tensors. This distinction also matters in evaluation: a VSA result is
a related sparse system, not a strictly matched weight comparison.

## Cache ownership must follow process ownership

Four-GPU execution uses spawned worker processes. Two implementation details
follow from that process model.

First, an `FP8Linear` instance cannot retain a Python extension module object.
Such modules are not pickleable, so the parent model cannot be serialized for a
spawned worker. The wrapper stores only serializable configuration and resolves
the kernel implementation at runtime inside the owning process.

Second, the parent cannot build CUDA FP8 caches and expect child workers to
inherit them safely. Workers begin from CPU weights, establish their CUDA
device, and construct device-local caches lazily. This avoids inheriting
parent-process CUDA tensors and ensures each cache belongs to the allocator and
device that execute it.

These choices also make memory accounting meaningful. The temporary BF16
source must be released after cache construction; otherwise a reported FP8
peak includes both copies and says little about the steady model. Whole-process
peak may still occur in the text encoder or decoder because end-to-end memory
is the maximum over pipeline phases, not simply the byte ratio of DiT weights.

The system is now complete in the sense required by the opening problem:
low-precision dense GEMMs, sparse low-precision attention, quality controls,
four-GPU tensor ownership, and adapter-correct weights participate in one
execution graph. We can therefore ask a single experimental question: does
this complete path outperform a maintained external MiniMax-H3 implementation
without violating the same output contract?
