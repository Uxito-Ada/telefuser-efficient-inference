# MiniMax-H3 Quality Suite

Status: **planned; prompts are frozen before generation**.

The suite tests failure modes exposed by low-precision sparse denoising. It does
not select prompts after seeing which outputs look favorable.

Every admitted profile uses the same prompt text, seed, output specification,
and adapter as its paired reference. The final HTML article embeds original
MP4s with synchronized controls.

## Evidence layers

1. Operator diagnostics on captured post-norm/post-RoPE H3 Q/K/V.
2. Automated video, temporal, semantic, audio, and stream-validity metrics.
3. Direct paired video and audio inspection.

Paired pixel and spectral metrics diagnose deviation from BF16; they are not
treated as human preference scores.
