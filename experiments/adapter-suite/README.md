# Adapter benchmark records

This directory preserves the normalized records used by the adapter sections of
the article. Results are not mixed across adapter families.

## MiniMax-H3 Turbo LoRA

The four-GPU comparison uses the same MiniMax-H3 base, Turbo 8-step v1.0 768p
adapter, prompt, seed, 1344 x 768 output, 124 frames, and eight DiT updates.
LightX2V and SGLang use TP2 x Ulysses SP2 with the DiT resident on GPU;
TeleFuser uses TP2 x Ulysses SP2, FP8 Linear, and FP8 Sol-Attn at `tau=1.0`.
CPU offload is disabled for every plotted point. The records report the
end-to-end request time and the peak GPU memory observed during the formal run.

The four-GPU records are one completed request after framework initialization;
the Base H3 and TeleFuser adapter records retain the longer repeated benchmark
series used for the main comparison.

The official LightX2V CPU block-offload run is retained outside this article as
a diagnostic. It is deliberately excluded from the chart because the plotted
comparison keeps the DiT resident in both frameworks.

Raw records:

- `raw/turbo-lightx2v-4gpu.json`
- `raw/turbo-sglang-4gpu.json`
- TeleFuser: `/data/heyang/telefuser-lora-adapter-pr-artifacts/turbo_telefuser_fp8_sol_4gpu.json`

Regenerate the figure with:

```bash
python experiments/adapter-suite/plot_turbo.py \
  --baseline experiments/adapter-suite/raw/turbo-lightx2v-bf16-resident.json \
  --candidate experiments/adapter-suite/raw/turbo-telefuser-fp8.json \
  --figure sections/04-evaluation/assets/turbo-performance.svg \
  --summary experiments/adapter-suite/raw/turbo-summary.json
```

## FastH3 dense adapter

The four-GPU comparison uses the same dense adapter, prompt, seed, output
shape, and four actual DiT forwards. FastVideo uses BF16 Linear + FlashAttention
4 with SP4 and FSDP resharding on H100; TeleFuser uses FP8 Linear + FP8
Sol-Attn with `tau=1.0` and TP2 x Ulysses SP2. Neither path uses CPU offload.

Raw records:

- `raw/fasth3-fastvideo-4gpu.json`
- `raw/fasth3-telefuser-4gpu-ramen.json` (same-prompt matrix render)
- TeleFuser: `/data/heyang/telefuser-lora-adapter-pr-artifacts/fasth3_telefuser_fp8_sol_4gpu.json`
