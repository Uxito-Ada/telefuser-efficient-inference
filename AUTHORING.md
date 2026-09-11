# Authoring Guide

The repository is modular like a collection of skills, but the article must
read as one argument. A locally correct section is not acceptable if it breaks
the causal hand-off to the section before or after it.

## Editing one section

1. Read the hidden `SECTION-CONTRACT` at the top of `README.md`.
2. Read `section.yaml`, especially `incoming_premise` and
   `outgoing_question`.
3. Preserve the four-part reasoning unit: observation, explanation, design,
   evidence.
4. Put concrete MiniMax-H3 examples inside that reasoning; do not create a
   chapter merely because an implementation once had its own pull request.
5. Update the abstract, evaluation, and conclusion only when new evidence
   changes the article-level claim.
6. Rebuild and validate the assembled article.

## Claim policy

- Final numbers must come from normalized benchmark records under
  `experiments/h100-4gpu-e2e/`; the directory name is retained from the
  original experiment plan.
- Do not copy a historical PR number into a final chart.
- A speedup claim requires the same checkpoint, adapter, task, output shape,
  sampling work, GPU count, warm-up policy, and timing boundary. If framework
  cache policies differ, compare only the matched phase.
- The primary performance baseline is an external framework, not another
  TeleFuser mode.
- Use actual DiT forwards when describing denoising throughput. Scheduler
  points and transformer evaluations are not interchangeable.
- Whole-process memory must be measured after warm-up. Multi-GPU studies must
  identify both per-GPU and aggregate maxima.
- Paired PSNR, SSIM, LPIPS, cosine, or spectral error measure trajectory
  similarity to BF16. They do not by themselves prove perceptual equivalence.
- A framework that OOMs, crashes, or produces invalid media is an excluded
  result, not a performance datapoint.

## Claim provenance

Do not estimate a value from an old run or from another hardware generation.
Every headline value must remain linked to its raw report in the claim ledger.

## Media policy

The final article is built as HTML so video evidence is displayed with native
`<video controls>` players. Do not replace dynamic evidence with a contact
sheet. Each paired comparison must expose the original MP4 files and use the
same prompt, seed, resolution, frame count, and FPS.

Do not commit checkpoints, captured full-size tensors, credentials, or
machine-local absolute paths.

## Validation

```bash
python scripts/build_blog.py
python scripts/validate_repo.py
git diff --check
```
