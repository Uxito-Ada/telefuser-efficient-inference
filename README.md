# Fast and Faithful MiniMax-H3 Inference

**A community technical blog about co-designing FP8, sparse attention, and
multi-GPU execution in TeleFuser.**

MiniMax-H3 generates high-resolution video and synchronized audio with one
large diffusion transformer. This project explains how TeleFuser combines an
H100-native FP8 Sol-Attn path, attention smoothing, sequence parallelism, and
adapter-aware weight preparation without treating output quality as an
afterthought.

- [Read the assembled article](BLOG.md)
- [Open the HTML edition](site/index.html)
- [Reproduce the four-H100 benchmark](experiments/h100-4gpu-e2e/README.md)
- [Edit an individual section](AUTHORING.md)

## Article map

| Section | Focus |
|---|---|
| [Introduction](sections/00-introduction/README.md) | The MiniMax-H3 efficiency and quality problem |
| [Why co-design](sections/01-why-co-design/README.md) | Why FP8 and sparse attention are complementary but not automatically composable |
| [System overview](sections/02-system-overview/README.md) | The H100 execution path |
| [Quality and scale](sections/03-quality-and-scale/README.md) | Smoothing, dense islands, multi-GPU execution, and adapters |
| [Evaluation](sections/04-evaluation/README.md) | One matched external comparison with generated video and audio |
| [Lessons](sections/05-lessons/README.md) | Scope, limitations, and takeaways |

Each directory owns publishable prose, section metadata, and its media. The
single article is generated with:

    python scripts/build_blog.py
    python scripts/validate_repo.py

Headline values remain marked TBD until the new matched experiment completes.
