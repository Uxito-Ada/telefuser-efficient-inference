# Four-H100 End-to-End Experiment

Status: **planned; no result has been admitted**.

## Question

Does TeleFuser's complete FP8 Linear + FP8 Sol + smoothing + Ulysses SP4 path
outperform the maintained FastVideo FastH3 Dense/Data-Free BF16 + FA4 path on
the same MiniMax-H3 request and four H100 GPUs?

## Strict comparison

The following variables must match:

- MiniMax-H3 base checkpoint revision;
- Dense/Data-Free adapter file hash and scale;
- text-to-video-and-audio task;
- prompt bytes and seed;
- 1344 x 768 output, 124 frames, 24 FPS;
- five sigma points and four actual DiT forwards;
- batch size one and request concurrency one;
- four visible H100 GPUs;
- one full warm-up and five formal repeats;
- encode-through-MP4-close timing boundary.

Framework-specific configuration is allowed only to select the documented
attention backend and distributed mode. Source modifications to an external
baseline disqualify it from the strict comparison.

## Profiles

| ID | Framework | Precision and attention | Distributed mode | Role |
|---|---|---|---|---|
| `fastvideo-dense-fa4` | FastVideo | BF16 Linear + FA4 | maintained 4-GPU recipe | primary external baseline |
| `telefuser-fp8-sol` | TeleFuser | FP8 Linear + FP8 Sol + smoothing | Ulysses SP4 | proposed system |
| `fastvideo-vsa` | FastVideo | official VSA route | maintained 4-GPU route | related sparse context, if valid on H100 |

## Timing

Use host `perf_counter` around the complete synchronous generation and file
write. Synchronize CUDA at framework boundaries if generation returns before
GPU work completes. Store each sample, not only an aggregate. Report median,
minimum, maximum, and median absolute deviation.

The warm-up must traverse text encoding, denoising, video/audio decode, mux, and
file close. Cold load, adapter merge, quantization, and compilation are recorded
separately.

## Memory

Sample NVML every 100 ms from the beginning of each formal request until MP4
close. Record every physical GPU. Derive:

- maximum used memory on any one GPU;
- sum across GPUs at every timestamp, then its maximum;
- the phase in which each peak occurs.

Do not infer FP8 memory from model parameter bytes.

## Validity gate

Before a timing sample is admitted, its output must contain:

- 124 decodable video frames at 1344 x 768;
- expected duration and 24 FPS metadata;
- a finite, non-empty stereo audio stream;
- no black/corrupt-frame failure;
- the requested adapter confirmed in logs.

Failed profiles are documented in `evidence/excluded-results.md` without a
performance number.

## Required artifacts

```text
config.yaml
raw/
  environment.json
  fastvideo-dense-fa4.json
  telefuser-fp8-sol.json
  fastvideo-vsa.json              # only if admitted
  summary.json
telemetry/
logs/
videos/
figures/
  end-to-end.svg
```
