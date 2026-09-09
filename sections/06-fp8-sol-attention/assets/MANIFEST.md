# Asset Manifest

## Figures

| File | Origin |
|---|---|
| `h3-performance.svg`, `wan-performance.svg` | PR #30 performance attachments |
| `h3-quality.png`, `wan-quality.png` | PR #30 frame-comparison attachments |

## Video evidence

The eight MP4 files reproduce the BF16/FP8 x Dense/Sol matrix for MiniMax-H3
and Wan. They were downloaded from the immutable PR #30 attachment URLs. The
original URLs remain in the [PR description](https://github.com/Tele-AI/TeleFuser/pull/30).

## Raw data

- `minimax_h3_{bf16_dense,bf16_sol,fp8_dense,fp8_sol}.metrics.json`
- `minimax_h3_sol_parameter_sweep.json`
- `results.json` for the independent Wan clean benchmark

The Wan `results.json` is an independent clean rerun with a different prompt
and step count. It supports mechanism analysis but does not replace the final
PR chart values used in the text.
