# Authoring Guide

This repository is organized like a collection of narrowly scoped skills. A
section can be edited independently as long as its local contract is respected.

## Editing one section

1. Open `sections/<id>/README.md`.
2. Read its hidden `SECTION-CONTRACT` before changing prose.
3. Check `section.yaml` for upstream dependencies and canonical claims.
4. Check `assets/MANIFEST.md` before replacing a figure or metric file.
5. Edit only that section unless a changed claim affects the abstract,
   evaluation synthesis, or conclusion.
6. Run `python scripts/build_blog.py --check` to see whether `BLOG.md` is stale.
7. Run `python scripts/build_blog.py`, then `python scripts/validate_repo.py`.

## Source priority

When sources disagree, use this order:

1. The final public PR description at the cited revision.
2. Raw JSON from the final measured profile.
3. The benchmark README associated with the PR.
4. Intermediate local reports.
5. Conversation notes.

Do not silently mix warm and cold measurements, denoise and end-to-end time,
allocated and NVML-sampled memory, or Base H3 and distilled FastH3 workloads.

## Claim rules

- State the model, resolution, frame count, denoising work, GPU count, and
  warm-up policy close to every primary performance result.
- Use `step/s` only when the numerator is the configured scheduler points.
  Use `DiT updates/s` when reporting actual transformer evaluations.
- Call a comparison "matched" only when prompt, output shape, denoising work,
  and measurement boundary match. Explain remaining framework differences.
- Treat same-seed PSNR, SSIM, cosine, and audio metrics as trajectory
  similarity, not absolute perceptual quality.
- Never infer a global speedup by multiplying results from separate cases.
- Keep failures and rejected alternatives when they explain the final design.

## Media rules

The checked-in MP4s are evidence snapshots. The section text also retains the
original GitHub `user-attachments` URLs because standalone URLs render as
players in pull-request Markdown. In this repository, use a thumbnail or
contact sheet followed by a normal link to the local MP4.

No source model checkpoints, captured tensors, credentials, or machine-local
absolute paths belong in the repository.

## Rebuilding

```bash
python scripts/build_blog.py
python scripts/validate_repo.py
git diff --check
```
