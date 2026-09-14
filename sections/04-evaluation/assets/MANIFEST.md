# Assets

- `lightx2v-base-h3.svg`: matched four-H100 Base H3 performance chart.
- `lightx2v-base-h3.mp4`: LightX2V BF16 + SageAttention2 Base H3 output.
- `telefuser-base-h3.mp4`: TeleFuser FP8 + Sol Base H3 output using a camera-stable presentation prompt.
- `turbo-performance.svg`: matched one-H100 MiniMax-H3 Turbo LoRA chart.
- `turbo-lightx2v.mp4`: LightX2V resident BF16 Turbo LoRA output.
- `turbo-telefuser.mp4`: TeleFuser FP8 Turbo LoRA output using a camera-stable presentation prompt.
- `end-to-end.svg`: generated from the normalized single-H100 records.
- `fastvideo-primary.mp4`: FastVideo BF16 Linear + FA4 output.
- `telefuser-primary.mp4`: TeleFuser FP8 Linear + smoothed FP8 Sol output.

Videos are grouped only within the same Base, Turbo, or FastH3 workload. The
performance records use matched prompts and seeds. The refreshed TeleFuser Base
and Turbo presentation samples use camera-stable prompts, as disclosed in the
evaluation text, and are not presented as paired quality measurements.
