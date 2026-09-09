<!--
SECTION-CONTRACT
id: 00-introduction
incoming_premise: none
outgoing_question: Why do low precision and sparse attention need one design?
evidence: experiments/h100-4gpu-e2e/raw/summary.json
do_not_claim: Do not publish a speedup before the matched experiment is complete.
-->

# Fast and Faithful MiniMax-H3 Inference on H100

*Co-designing FP8, sparse attention, and multi-GPU execution in TeleFuser*

MiniMax-H3 raises the bar for open video generation: one large diffusion
transformer jointly produces high-resolution video and synchronized audio.
That unified model is also expensive to serve. Long visual and audio sequences
make attention costly, while the transformer's projections and feed-forward
layers keep the tensor cores busy throughout denoising.

The usual optimization menu looks straightforward: quantize the dense layers,
make attention sparse, and split the sequence across GPUs. In practice, those
features often work only as isolated switches. They disagree about tensor
layouts, quantization scales, and where communication happens. Combining them
can erase the speedup or, worse, produce a fast but visibly degraded result.

We built a MiniMax-H3 path in TeleFuser that treats these choices as one
execution design:

- FP8 Linear layers and an FP8 Sol-Attn kernel for NVIDIA H100;
- attention smoothing and selective dense computation to protect quality;
- tensor and Ulysses sequence parallelism for multi-GPU inference;
- adapter-aware weight preparation for Turbo and FastH3-style adapters.

This post explains the design at a systems level, then compares the complete
path with FastVideo's maintained FastH3 recipe on the same four H100 GPUs. The
headline performance and quality results are **TBD--new experiment required**.

MiniMax-H3 is the only model in this study. That constraint is deliberate: the
goal is not to collect unrelated kernel wins, but to show that low precision,
sparsity, distribution, and quality control can survive one real end-to-end
generation workload.
