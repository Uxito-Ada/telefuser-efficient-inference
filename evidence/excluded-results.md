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

## Multi-H100 setup results

- FastVideo commit `a943220`, Dense/Data-Free adapter, two H100 80GB GPUs,
  replicated DiT: excluded. The clean maintained profile reached about
  79.05 GiB per GPU and OOMed in full MiniMax-H3 VAE decode. Disabling VAE
  compile did not make the resident replicated profile fit.
- Reducing that request to 107 frames was rejected by FastVideo's official
  MiniMax-H3 duration guard. Its supported FastH3 range starts at five seconds,
  so a smaller off-protocol output was not used to manufacture a fit.
- FastVideo's official lazy-module route with LoRA and FSDP: excluded. During
  distributed materialization, rank 1 attempted to build a second transformer
  copy and OOMed before a complete warm-up.
- The same FastVideo commit with FSDP-sharded DiT and regional compile:
  excluded. Warm-up failed at the first DiT block because the regional Dynamo
  graph attempted to trace an FSDP hook marked `torch._dynamo.disable`.
- Early environment probes that mixed the clean FastVideo package with an
  `examples` module from an editable checkout are excluded without results.
  The final harness pins both imports to the declared clean source root and
  records their resolved files.
- TeleFuser's resident two-GPU stage topology: excluded. The text and VAE
  workers were already resident when the denoising worker onloaded the DiT,
  leaving insufficient HBM for a complete request. This is a topology capacity
  result, not a denoising performance result.
- An earlier LightX2V Sol environment produced invalid MiniMax-H3 media. It must
  be reproduced from the current official example before it can enter the new
  comparison.
- Published B200 FastH3 numbers are not projected onto H100.

No partial denoising or warm-up time from these runs is used as performance
data.
