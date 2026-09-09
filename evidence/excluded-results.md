# Excluded and Negative Results

This file records failed or incomparable profiles without turning them into
performance claims.

## Admission rules

A profile is excluded when it:

- changes checkpoint, adapter, output, sampling work, or GPU count in the
  headline comparison;
- requires source development in an external framework rather than documented
  configuration;
- fails before a complete warm-up;
- runs out of memory;
- produces corrupt, black, meaningless, or missing-audio media; or
- cannot expose the common end-to-end timing boundary.

## Current pre-experiment observations

- A preliminary FastVideo checkout encountered a compile/FSDP interaction and,
  with compile disabled, exceeded available memory on a two-H100 attempt. These
  are setup observations, not four-H100 performance results.
- An earlier LightX2V Sol environment produced invalid MiniMax-H3 media. It must
  be reproduced from the current official example before it can enter the new
  comparison.
- Published B200 FastH3 numbers are not projected onto H100.

These entries will be replaced or expanded with exact revisions and logs during
the experiment phase.
