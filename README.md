# TeleFuser Efficient Inference

**A community technical blog about TeleFuser's quality-aware FP8, sparse
attention, adapter, and multi-GPU optimization stack.**

TeleFuser is an open-source streaming inference and serving framework for
real-time world models and multimodal generation. This article uses
MiniMax-H3 to show how its H100 optimization stack combines FP8 Linear, FP8
Sol-Attn, attention smoothing, adapters, and distributed execution.

- [Read the assembled article](BLOG.md)
- [Open the HTML edition](site/index.html)
- [Inspect and reproduce the benchmark](experiments/h100-4gpu-e2e/README.md)
- [Edit an individual section](AUTHORING.md)

## Article map

| Section | Focus |
|---|---|
| [Introduction](sections/00-introduction/README.md) | TeleFuser and the MiniMax-H3 optimization result |
| [Why co-design](sections/01-why-co-design/README.md) | Why FP8 and sparse attention need one design |
| [Optimization stack](sections/02-system-overview/README.md) | FP8 Sol-Attn, adapters, and distributed execution |
| [Quality and scale](sections/03-quality-and-scale/README.md) | Attention smoothing, quality-aware sparsity, and Ulysses |
| [Results](sections/04-evaluation/README.md) | External performance baseline and generated output |
| [Conclusion](sections/05-lessons/README.md) | The unified TeleFuser inference path |

Each directory owns publishable prose, section metadata, and its media. The
single article is generated with:

    python scripts/build_blog.py
    python scripts/validate_repo.py

The published article is generated from the section directories so each part
can be reviewed and refined without editing one monolithic document.
