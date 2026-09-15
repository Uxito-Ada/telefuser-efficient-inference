# MiniMax-H3 H100 evidence

Status: **complete for the claims published in the article**.

The article keeps the TeleFuser scaling run, the four-H100 framework
comparison, the one-H100 adapter runs, and the communication-overlap regression
as separate protocols.

## TeleFuser one-, two-, and four-GPU scaling

The scaling comparison uses the same MiniMax-H3 Base prompt, seed, output
shape, 50-point schedule, and FP8 Linear + FP8 Sol profile at all three points.
KV smoothing and feature cache are disabled to isolate parallel execution. The
topologies are local, `TP2`, and `TP2 x Ulysses SP2`. Raw measurements are
stored in `raw/telefuser-base-h3-{1,2}gpu.json`; the four-GPU source is retained
in `raw/lightx2v-base-h3-comparison.json`, and the normalized series is
`raw/telefuser-base-h3-scaling.json`.

```bash
python experiments/h100-4gpu-e2e/scripts/plot_scaling.py \
  --input experiments/h100-4gpu-e2e/raw/telefuser-base-h3-scaling.json \
  --figure sections/04-evaluation/assets/base-scaling.svg
```

## Primary four-H100 framework comparison

SGLang, LightX2V, and TeleFuser use:

- MiniMax-H3 Base T2AV on four NVIDIA H100 80GB GPUs;
- `TP2 x Ulysses SP2` in both frameworks;
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

Normalized records:

- `raw/lightx2v-base-h3-comparison.json`
- `raw/lightx2v-base-h3-summary.json`

Generate the chart with:

```bash
python experiments/h100-4gpu-e2e/scripts/plot_lightx2v_results.py \
  --input experiments/h100-4gpu-e2e/raw/lightx2v-base-h3-comparison.json \
  --figure sections/04-evaluation/assets/lightx2v-base-h3.svg \
  --summary experiments/h100-4gpu-e2e/raw/lightx2v-base-h3-summary.json
```

Both published MP4s pass the media gate: 124 decodable 1344 x 768 H.264 frames,
finite 32kHz stereo AAC audio, and matching duration. Device memory was sampled
at 100 ms intervals; the normalized record identifies the ranks used for each
reported statistic.

## FastH3 adapter comparison

Both systems use the MiniMax-H3 base, FastH3 Dense/Data-Free adapter at strength
1.0, same prompt and seed, 1344 x 768 output, 124 frames at 24 FPS, five sigma
points, four actual DiT forwards, one H100, one warm-up, and three measured
requests. Neither DiT is CPU-offloaded during denoising.

- FastVideo: BF16 Linear + FlashAttention 4.
- TeleFuser: FP8 Linear + FP8 Sol-Attn, `tau=1.0`, exact routing, quality-aware
  FP8 attention, and selective dense computation.

The chart uses matched denoising time, actual DiT forwards per second, and
whole-process peak GPU memory. It omits an end-to-end speedup because the
recorded frameworks used different prompt-conditioning cache policies.

Normalized records:

- `raw/fastvideo-single-h100.json`
- `raw/telefuser-single-h100.json`
- `raw/telefuser-single-h100-source.json`
- `raw/summary.json`

Generate the chart with:

```bash
python experiments/h100-4gpu-e2e/scripts/plot_results.py \
  --baseline experiments/h100-4gpu-e2e/raw/fastvideo-single-h100.json \
  --telefuser experiments/h100-4gpu-e2e/raw/telefuser-single-h100.json \
  --figure sections/04-evaluation/assets/end-to-end.svg \
  --summary experiments/h100-4gpu-e2e/raw/summary.json
```

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
