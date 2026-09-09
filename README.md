# Fast and Faithful World-Model Inference

**Co-designing FP8, sparse attention, and sequence parallelism for MiniMax-H3**

This repository contains a modular draft of an efficient-AI technical article.
It studies one question: how can a compute-bound world model become faster on
NVIDIA H100 GPUs without giving up the visual, temporal, and audio quality that
made the model useful in the first place?

The article follows a single causal argument. Quality requirements make the
MiniMax-H3 DiT expensive; quantization and sparsity are therefore both needed,
but their numerical and systems contracts conflict. Resolving that conflict
requires a hardware-aware FP8 sparse-attention path, explicit quality controls,
and a distributed weight lifecycle that remains correct after adapters mutate
the model.

> **Draft status:** the method narrative is being written first. Every headline
> performance or quality value is intentionally marked `TBD` until a new,
> matched 4xH100 experiment is complete. Historical PR measurements are not
> reused as final evidence.

## Read the draft

- [Assembled article](BLOG.md)
- [Section editing guide](AUTHORING.md)
- [Flagship experiment contract](experiments/h100-sp4-e2e/README.md)
- [Claim and evidence policy](evidence/README.md)
- [Related work and writing references](references/README.md)

## Article map

| Section | Question it answers | Why the next section is necessary |
|---|---|---|
| [Abstract](sections/00-abstract/README.md) | What system problem and contribution does the article study? | The result needs a precise model-level problem. |
| [Quality has a computational cost](sections/01-quality-cost/README.md) | Why is high-quality world-model inference expensive? | One optimization cannot remove both Linear and attention cost. |
| [The composition gap](sections/02-composition-gap/README.md) | Why do FP8 and sparsity fail to compose through existing APIs? | A shared H100 execution path is missing. |
| [The SM90 FP8 sparse path](sections/03-sm90-fp8-sol/README.md) | How are quantization, layout, routing, and compute fused? | Two approximations now perturb one denoising trajectory. |
| [Quality under compound approximation](sections/04-quality-control/README.md) | How is error controlled without reverting the whole graph to BF16? | The numerical contract must survive tensor redistribution. |
| [Distributed numerical contract](sections/05-distributed-contract/README.md) | Where do communication, quantization, and routing belong under Ulysses? | Production checkpoints are mutable through adapters. |
| [Mutable weight lifecycle](sections/06-weight-lifecycle/README.md) | How do adapters, FP8 caches, and spawn workers remain consistent? | The complete system can finally be evaluated. |
| [End-to-end evaluation](sections/07-evaluation/README.md) | Does the complete SP4 system beat external baselines at comparable quality? | Results need interpretation and boundaries. |
| [Discussion and conclusion](sections/08-discussion/README.md) | What generalizes, and what remains architecture-specific? | - |

## Repository contract

Each section is intentionally self-contained for editing but not independent in
argument. Its `README.md` contains publishable prose; `section.yaml` records the
incoming premise, outgoing bridge, evidence requirements, and forbidden claims;
`assets/` owns only media used by that section. `BLOG.md` is generated in order.

```bash
python scripts/build_blog.py
python scripts/validate_repo.py
```
