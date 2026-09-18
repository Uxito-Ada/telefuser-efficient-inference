# MiniMax-H3 H100 evidence

Status: **complete for the claims published in the article**.

The article keeps the TeleFuser scaling run, the four-H100 framework
comparison, the four-H100 adapter comparison, and the communication-overlap
regression as separate protocols.

## TeleFuser one-, two-, and four-GPU scaling

The scaling comparison uses the same MiniMax-H3 Base prompt, seed, output
shape, 50-point schedule, and FP8 Linear + FP8 Sol profile at all three points.
KV smoothing and feature cache are disabled to isolate parallel execution. The
topologies are local, `TP2`, and `TP2 x Ulysses SP2`. Raw measurements are
stored in `raw/telefuser-base-h3-{1,2}gpu.json`; the normalized TeleFuser series
is `raw/telefuser-base-h3-scaling.json`. The separate FastVideo Base H3 control
is recorded in `raw/fastvideo-base-h3.json` and uses SP4.

```bash
python experiments/h100-4gpu-e2e/scripts/plot_scaling.py \
  --input experiments/h100-4gpu-e2e/raw/telefuser-base-h3-scaling.json \
  --figure sections/04-evaluation/assets/base-scaling.svg
```

## Primary four-H100 framework comparison

SGLang, LightX2V, FastVideo, and TeleFuser use:

- MiniMax-H3 Base T2AV on four NVIDIA H100 80GB GPUs;
- `TP2 x Ulysses SP2` for SGLang, LightX2V, and TeleFuser; FastVideo uses SP4;
- the same prompt, seed 0, 1344 x 768 output, 124 frames, 24 FPS, and 50
  configured sampling points;
- feature cache disabled; and
- one complete warm-up followed by one measured request.

The working LightX2V baseline is BF16 + SageAttention2. TeleFuser uses
tf-kernel W8A8 FP8 Linear + FP8 Sol-Attn with exact routing. The invalid
LightX2V Base H3 Sol output is not used as a performance result.

The SGLang point is the matched 79.37-second, 67.8-GiB H100 run retained in
TeleFuser's MiniMax-H3 documentation. Its `TP2 x Ulysses SP2` topology is also
listed as verified by the official SGLang MiniMax-H3 cookbook.

FastVideo's matched four-GPU Base H3 rerun uses the same ramen prompt, seed 0,
1344 x 768 output, 124 frames, and 50-point schedule. Its measured median is
114.698 seconds E2E and 100.747 seconds denoising; the retained output is
`sections/04-evaluation/assets/fastvideo-base-h3.mp4`. The run uses the
official dense FA4 path with SP4 and no DiT offload.

Normalized records:

- `raw/lightx2v-base-h3-comparison.json`
- `raw/fastvideo-base-h3.json`
- `raw/lightx2v-base-h3-summary.json`

Generate the chart with:

```bash
python experiments/h100-4gpu-e2e/scripts/plot_lightx2v_results.py \
  --input experiments/h100-4gpu-e2e/raw/lightx2v-base-h3-comparison.json \
  --figure sections/04-evaluation/assets/lightx2v-base-h3.svg \
  --summary experiments/h100-4gpu-e2e/raw/lightx2v-base-h3-summary.json
```

The retained LightX2V, FastVideo, and TeleFuser MP4s pass the media gate: 124 decodable 1344 x 768 H.264 frames,
finite 32kHz stereo AAC audio, and matching duration. Device memory was sampled
at 100 ms intervals; the normalized record identifies the ranks used for each
reported statistic.

## FastH3 adapter comparison

The current four-GPU adapter measurements are maintained in
[`experiments/adapter-suite/README.md`](../adapter-suite/README.md). They use the
MiniMax-H3 base with the FastH3 Dense/Data-Free adapter at strength 1.0, and
compare the official FastVideo distributed path with TeleFuser's FP8 Sol path.
The adapter video matrix uses the same ramen prompt for the four-GPU FastVideo
and TeleFuser FastH3 renders.

- FastVideo: BF16 Linear + FlashAttention 4.
- TeleFuser: FP8 Linear + FP8 Sol-Attn, `tau=1.0`, exact routing, quality-aware
  FP8 attention, and selective dense computation.

The unified four-GPU chart uses the normalized records in
`experiments/adapter-suite/raw/` and reports end-to-end throughput and whole-
process peak memory for each supported workload.

## Communication-overlap regression

[TeleFuser PR 37](https://github.com/Tele-AI/TeleFuser/pull/37) reports five
Base H3 prompt/seed cases with resident `TP2 x Ulysses SP2`. Overlap reduced
mean request wall time from 78.546 to 75.766 seconds, or 3.539%, while all five
MP4 outputs remained byte-identical.

This validates the scheduling optimization independently. It is not folded
into the LightX2V speedup because it is a different run set.

Failed or invalid profiles remain documented in
[`evidence/excluded-results.md`](../../evidence/excluded-results.md); their
partial timings do not appear in any chart or claim.
