# Asset Manifest

| File | Origin | Role |
|---|---|---|
| `performance.png` | PR #25 attachment | Matched quantization performance |
| `comparison.png` | Local MiniMax-H3 quality extraction | Frame comparison |
| `bf16.mp4`, `torchao-fp8.mp4`, `bnb-nf4.mp4`, `tf-kernel-fp8.mp4` | PR #25 attachments | Generated video/audio evidence |
| `minimax_h3_*_50step.metrics.json` | Local validation | BF16, TorchAO FP8, and NF4 metrics |
| `minimax_h3_fl2va_tf_kernel_fp8_h100_50steps.json` | Local validation | tf-kernel FP8 metrics |

The later validation JSONs do not all share the software revision used by the
original PR chart. The section keeps those two evidence groups separate.
