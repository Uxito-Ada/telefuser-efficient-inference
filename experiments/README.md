# Experiments

This directory contains the only datasets allowed to fill final article
claims. Historical PR artifacts are intentionally excluded.

- `h100-4gpu-e2e/`: the matched single-H100 external comparison, separate
  four-H100 scheduling evidence, and reusable distributed harnesses. The
  directory name is retained for continuity with the original four-GPU plan.
- `quality-suite/`: predefined prompts, tensor diagnostics, and paired media.

An experiment moves from `planned` to `complete` only after raw samples,
environment metadata, commands, output validity, and derived summaries are all
present.
