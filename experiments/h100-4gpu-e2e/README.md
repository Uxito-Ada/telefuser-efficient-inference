# MiniMax-H3 H100 Evidence

Status: **complete for the claims published in the article**.

The directory name records the original four-GPU experiment plan. The final
article separates two questions whose timing boundaries can be defended:

1. A matched single-H100 comparison against FastVideo isolates the FastH3
   adapter denoising path.
2. A separate four-H100 TeleFuser study validates the resident
   `TP2 x Ulysses SP2` schedule and communication-compute overlap.

Combining those results into one speedup would mix model schedules, framework
cache policies, and timing boundaries, so the blog reports them separately.

## Primary external comparison

Both systems use MiniMax-H3, the FastH3 Dense/Data-Free adapter at strength
1.0, the same prompt and seed, 1344 x 768 output, 124 frames at 24 FPS, five
sigma points, four actual DiT forwards, one H100 80GB, one warm-up, and three
measured requests. Neither DiT is CPU-offloaded during denoising.

- FastVideo: BF16 Linear + FlashAttention 4.
- TeleFuser: FP8 Linear + FP8 Sol-Attn, `tau=1.0`, exact routing, two dense
  opening updates, two dense layers, KV smoothing, and V correction.

The chart uses only matched denoising time, actual DiT forwards per second, and
whole-process peak GPU memory. It intentionally omits an E2E speedup: repeated
FastVideo requests reused prompt conditioning, while the recorded TeleFuser
requests encoded the prompt each time.

Normalized records:

- `raw/fastvideo-single-h100.json`
- `raw/telefuser-single-h100.json`
- `raw/telefuser-single-h100-source.json`
- `raw/summary.json`

Generate the publication chart with:

```bash
python experiments/h100-4gpu-e2e/scripts/plot_results.py \
  --baseline experiments/h100-4gpu-e2e/raw/fastvideo-single-h100.json \
  --telefuser experiments/h100-4gpu-e2e/raw/telefuser-single-h100.json \
  --figure sections/04-evaluation/assets/end-to-end.svg \
  --summary experiments/h100-4gpu-e2e/raw/summary.json
```

The original MP4s are published as `fastvideo-primary.mp4` and
`telefuser-primary.mp4` under the evaluation section assets. Both must pass the
media validity gate before a timing record is admitted: 124 decodable frames,
1344 x 768 H.264 video, finite stereo AAC audio, and matching duration.

## Separate four-H100 validation

[TeleFuser PR 37](https://github.com/Tele-AI/TeleFuser/pull/37) reports five
MiniMax-H3 prompt/seed cases on four H100s with resident
`TP2 x Ulysses SP2`. Communication-compute overlap reduced mean request wall
time from 78.546 to 75.766 seconds, or 3.539%, while the five synchronized MP4
outputs remained byte-identical.

This is evidence that the article's distributed path is exercised and that its
scheduling optimization is lossless. It is not the external FastH3 baseline:
the run uses a 50-point Base H3 protocol rather than the distilled five-point
adapter protocol.

## Reusable distributed harnesses

`scripts/benchmark_fastvideo.py` and `scripts/benchmark_telefuser.py` preserve
the strict four-GPU protocol for a future run when four idle H100s are
available. A result from either harness is not admitted automatically. It must
use the same checkpoint, adapter hash, sampling work, output contract, GPU
count, warm-up policy, and timing scope, and it must produce valid media.

Failed two-GPU attempts made during this refresh are recorded in
[`evidence/excluded-results.md`](../../evidence/excluded-results.md). Their
partial timings do not appear in any chart or claim.
