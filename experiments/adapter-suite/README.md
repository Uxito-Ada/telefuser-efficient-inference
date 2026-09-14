# Adapter benchmark records

This directory preserves the normalized records used by the adapter sections of
the article. Results are not mixed across adapter families.

## MiniMax-H3 Turbo LoRA

- One NVIDIA H100 80GB.
- Same MiniMax-H3 base, Turbo 8-step v1.0 768p adapter, input image, prompt,
  seed, 1344 x 768 output, 124 frames, and eight DiT updates.
- LightX2V: BF16 + Sol-Attn with the DiT resident on GPU. Text-encoder and VAE
  lifecycle follows its example configuration.
- TeleFuser: FP8 Linear + FP8 Sol-Attn, `tau=1.0`, one dense opening update.
- One warm-up followed by three measured generations; denoise time is the
  median. Throughput is eight updates divided by median denoise time.
- Memory is each framework's whole-process formal-run peak after warm-up. This
  is an end-to-end framework metric, not a pure DiT weight-size comparison.

The official LightX2V CPU block-offload run is retained outside this article as
a diagnostic. It is deliberately excluded from the chart because the plotted
comparison keeps the DiT resident in both frameworks.

Raw records:

- `raw/turbo-lightx2v-bf16-resident.json`
- `raw/turbo-telefuser-fp8.json`

Regenerate the figure with:

```bash
python experiments/adapter-suite/plot_turbo.py \
  --baseline experiments/adapter-suite/raw/turbo-lightx2v-bf16-resident.json \
  --candidate experiments/adapter-suite/raw/turbo-telefuser-fp8.json \
  --figure sections/04-evaluation/assets/turbo-performance.svg \
  --summary experiments/adapter-suite/raw/turbo-summary.json
```

## FastH3 dense adapter

The matched FastVideo and TeleFuser records remain under
`experiments/h100-4gpu-e2e/raw/` because the original end-to-end harness
produced them. They use one H100, the same dense adapter, prompt, seed, output
shape, and four actual DiT forwards. See
`experiments/h100-4gpu-e2e/README.md` for the exact timing boundary.
