# TeleFuser PR Evidence Index

This index keeps the article's narrative layer traceable to the original
engineering work. A section may interpret a result, but the linked PR and its
local asset manifest remain the source of record.

| PR | Contribution | Article section | Local evidence |
|---|---|---|---|
| [#16](https://github.com/Tele-AI/TeleFuser/pull/16) | Online FP8 and NF4 weight quantization | [Qwen-Image online quantization](../sections/04-online-quantization-qwen/README.md) | [Manifest](../sections/04-online-quantization-qwen/assets/MANIFEST.md) |
| [#25](https://github.com/Tele-AI/TeleFuser/pull/25) | Online FP8/NF4 for MiniMax-H3 | [MiniMax-H3 online quantization](../sections/05-online-quantization-h3/README.md) | [Manifest](../sections/05-online-quantization-h3/assets/MANIFEST.md) |
| [#30](https://github.com/Tele-AI/TeleFuser/pull/30) | FP8 QKV and Sol-Attn for Wan and MiniMax-H3 | [FP8 Sol-Attn](../sections/06-fp8-sol-attention/README.md) | [Manifest](../sections/06-fp8-sol-attention/assets/MANIFEST.md) |
| [#35](https://github.com/Tele-AI/TeleFuser/pull/35) | Sequence-parallel FP8 Sol-Attn | [Sequence parallelism](../sections/07-sequence-parallel/README.md) | [Manifest](../sections/07-sequence-parallel/assets/MANIFEST.md) |
| [#40](https://github.com/Tele-AI/TeleFuser/pull/40) | Attention smoothing and quality evaluation | [Attention smoothing](../sections/08-attention-smoothing/README.md) | [Manifest](../sections/08-attention-smoothing/assets/MANIFEST.md) |
| [#44](https://github.com/Tele-AI/TeleFuser/pull/44) | LoRA, FP8 adapters, MiniMax-H3 Turbo, and FastH3 | [Adapter-aware deployment](../sections/09-adapters/README.md) | [Manifest](../sections/09-adapters/assets/MANIFEST.md) |

## Revision boundary

The article describes the state represented by these six PRs and their
recorded benchmarks. PR #40 was open when this snapshot was assembled; the
other five PRs were merged. Future implementation changes should update the
relevant section contract, raw metric artifact, and this index together.
