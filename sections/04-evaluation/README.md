<!--
SECTION-CONTRACT
id: 04-evaluation
incoming_premise: The final path combines precision, sparsity, quality controls, adapters, and four-GPU execution.
outgoing_question: What can the result teach beyond this one benchmark?
evidence: only new matched records under experiments/
do_not_claim: Do not use invalid media, different GPU counts, or old PR measurements.
-->

# End-to-End Evaluation

We compare against **FastVideo's maintained FastH3 Dense/Data-Free recipe**,
not another TeleFuser mode. The baseline uses its intended BF16 Linear and FA4
path with SP4. TeleFuser uses FP8 Linear, FP8 Sol-Attn with smoothing, and
TP2 x Ulysses SP2. Each framework keeps its native distributed strategy, but
both receive the same four H100 80GB GPUs and the same generation request.

The matched workload uses the MiniMax-H3 base checkpoint, the released FastH3
Dense/Data-Free adapter at strength 1.0, one fixed prompt and seed, 1344 x 768
output, 124 frames at 24 FPS, and five scheduler sigma points (four actual DiT
forwards). We run one full warm-up followed by five measured generations.
End-to-end latency starts before prompt processing and ends after the MP4,
including stereo audio, is closed.

![Four-H100 end-to-end comparison](assets/end-to-end.svg)

The final result is **TBD--new experiment required**. Once the four GPUs pass
the clean-device gate, the chart will report median end-to-end latency,
completed videos per hour, denoising throughput, and peak memory from the same
runs. Raw samples, environment revisions, adapter hashes, and MP4 validity
checks live in
[the experiment directory](../../experiments/h100-4gpu-e2e/README.md).

## Quality is a gate, not decoration

Every timed output must first pass mechanical checks: exact resolution and
frame count, decodable stereo audio, non-empty media, and no corrupt or black
video. We then evaluate a fixed prompt suite covering fast motion, camera
motion, multiple subjects, low-light detail, faces and hands, speech, and
event-aligned sound.

Tensor MSE, cosine similarity, and SQNR isolate error at the quantized attention
boundary. Frame cosine, PSNR, SSIM, waveform cosine, and spectral distance
describe end-to-end trajectory divergence. These metrics are diagnostic rather
than a substitute for viewing the samples: two diffusion trajectories can
diverge sample by sample while remaining perceptually valid.

The HTML version of this post presents the original outputs as synchronized
players:

<div class="video-pair" data-sync-group="primary">
  <figure>
    <figcaption>FastVideo FastH3 baseline</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="fastvideo-primary"></video>
  </figure>
  <figure>
    <figcaption>TeleFuser FP8 Sol-Attn</figcaption>
    <video controls playsinline preload="metadata" data-result-slot="telefuser-primary"></video>
  </figure>
</div>

Both video and audio conclusions remain **TBD--new experiment required** until
the matched outputs are generated and scored.
