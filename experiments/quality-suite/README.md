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

## Matched media comparison

After the four-GPU runs complete, compare their admitted MP4 outputs with:

    /data/zuoxin/workspace/TeleFuser/.venv/bin/python \
      experiments/quality-suite/scripts/compare_media.py \
      --reference sections/04-evaluation/assets/fastvideo-primary.mp4 \
      --candidate sections/04-evaluation/assets/telefuser-primary.mp4 \
      --output experiments/quality-suite/raw/media-metrics.json

The script streams decoded frames rather than materializing the complete RGB
video in memory. It reports all-frame cosine and PSNR, sampled SSIM, waveform
cosine and MSE, spectral convergence, and log-spectral distance.
