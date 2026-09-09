# TeleFuser Efficient Inference

## From Quantization to Quality: Building an Efficient Video-DiT Inference Stack

This repository is a modular, paper-style technical blog about six TeleFuser contributions:
[PR #16](https://github.com/Tele-AI/TeleFuser/pull/16),
[PR #25](https://github.com/Tele-AI/TeleFuser/pull/25),
[PR #30](https://github.com/Tele-AI/TeleFuser/pull/30),
[PR #35](https://github.com/Tele-AI/TeleFuser/pull/35),
[PR #40](https://github.com/Tele-AI/TeleFuser/pull/40), and
[PR #44](https://github.com/Tele-AI/TeleFuser/pull/44).

The work follows one system arc: make low precision deployable, move FP8 through
the attention boundary, combine sparse attention with sequence parallelism,
recover quantization quality with algebraically equivalent smoothing, and keep
the optimized path usable after LoRA adapters change the weights.

> **Headline result.** On matched MiniMax-H3 workloads, the stack reaches up to
> 65.0% higher denoising throughput than TeleFuser BF16 Dense on one H100,
> 53.2% higher throughput than the matched four-H100 TeleFuser BF16 + FA4
> baseline, and 2.62x the denoising speed of the deployed LightX2V baseline.
> Attention smoothing then reduces dense attention-output MSE by 8.18% for a
> 2.11% throughput cost, while adapter-aware FP8 Sol retains 36.5%-57.2%
> throughput advantages over the matched external baselines.

These percentages come from different case-specific workloads. They are not
combined into one global speedup.

## Read

- [Complete assembled article](BLOG.md)
- [Authoring and section-editing guide](AUTHORING.md)
- [Reference work and style study](references/README.md)
- [PR-to-section evidence index](references/PR_INDEX.md)
- [Reproducibility contract](sections/10-evaluation/README.md)

## Modular paper map

| Section | Purpose | Primary evidence |
|---|---|---|
| [Abstract](sections/00-abstract/README.md) | Contributions and headline findings | PRs #16, #25, #30, #35, #40, #44 |
| [Background](sections/01-background/README.md) | Video-DiT cost model and low-precision attention | Sol-Attn, SageAttention2, xDiT, FA4 |
| [Motivation](sections/02-motivation/README.md) | Why isolated optimizations are insufficient | Cross-PR failure cases |
| [System overview](sections/03-system-overview/README.md) | Optimization stack and ownership boundaries | TeleFuser implementation |
| [Qwen-Image online quantization](sections/04-online-quantization-qwen/README.md) | General FP8/NF4 infrastructure | PR #16 |
| [MiniMax-H3 online quantization](sections/05-online-quantization-h3/README.md) | Large multimodal DiT deployment | PR #25 |
| [FP8 Sol-Attn](sections/06-fp8-sol-attention/README.md) | Fused quantized sparse attention | PR #30 |
| [Sequence parallel FP8 Sol](sections/07-sequence-parallel/README.md) | TP2 x Ulysses SP2 and tuning | PR #35 |
| [Attention smoothing](sections/08-attention-smoothing/README.md) | Accuracy recovery and fused correction | PR #40 |
| [Adapter-aware deployment](sections/09-adapters/README.md) | Turbo LoRA and FastH3 adapters | PR #44 |
| [Evaluation](sections/10-evaluation/README.md) | Protocols, metrics, and evidence table | Raw JSON and media |
| [Discussion](sections/11-discussion/README.md) | Lessons, limitations, open problems | Cross-case analysis |
| [Conclusion](sections/12-conclusion/README.md) | Efficient-AI takeaways | All cases |

## Repository contract

Each section is self-contained:

- `README.md` is the publishable section and begins with a hidden
  `SECTION-CONTRACT`.
- `section.yaml` declares ownership, source PRs, dependencies, and claims.
- `assets/` contains figures, videos, raw metrics, and a provenance manifest.
- `BLOG.md` is generated. Edit a section and run `python scripts/build_blog.py`.

The repository intentionally preserves negative results: online FP8 can be
slower than BF16, sparse attention can add memory, distributed scaling can be
communication-bound, and lower tensor MSE does not guarantee better audio.
Those observations are part of the system result, not noise to hide.
