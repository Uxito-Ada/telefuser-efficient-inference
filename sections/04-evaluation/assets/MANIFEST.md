# Assets
- `all-workloads-performance.svg`: combined Base H3, FastH3, and Turbo performance chart; Base H3 points use four GPUs.
- `smoothing-performance-lines.svg`: FP8 attention smoothing throughput comparison.

- `lightx2v-base-h3.mp4`: LightX2V BF16 + SageAttention2 Base H3 output.
- `telefuser-base-h3.mp4`: TeleFuser FP8 + Sol Base H3 output.
- `fastvideo-base-h3.mp4`: FastVideo BF16 + dense FA4 Base H3 output, 4-GPU SP4 run.
- `sglang-base-h3.mp4`: SGLang Base H3 output for the media comparison.
- `turbo-lightx2v.mp4`: LightX2V resident BF16 Turbo LoRA output.
- `turbo-sglang.mp4`: SGLang TP2 x Ulysses SP2 MiniMax-H3 Turbo LoRA output.
- `turbo-telefuser.mp4`: Regenerated TeleFuser FP8 Turbo LoRA T2AV output using the same ramen prompt, seed, and four-GPU setup as the LightX2V and SGLang comparison renders; this clip uses the pre-smoothing FP8 Sol setting (smoothing=none).
- `fastvideo-primary.mp4`: FastVideo BF16 Linear + FA4 FastH3 dense-adapter output, four-GPU SP4 run.
- `telefuser-primary.mp4`: TeleFuser FP8 Linear + FP8 Sol FastH3 dense-adapter output, four-GPU TP2 x Ulysses SP2 run.
