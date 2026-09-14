#!/usr/bin/env python3
"""Plot the matched MiniMax-H3 Turbo adapter benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def reduction(candidate: float, baseline: float) -> float:
    return (1.0 - candidate / baseline) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    baseline: dict[str, Any] = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate: dict[str, Any] = json.loads(args.candidate.read_text(encoding="utf-8"))
    baseline_time = baseline["denoising_seconds_median"]
    candidate_time = candidate["denoising_seconds_median"]
    updates = baseline["effective_dit_updates"]
    baseline_throughput = updates / baseline_time
    candidate_throughput = updates / candidate_time
    baseline_memory = baseline["formal_run_peak_memory_mib"] / 1024.0
    candidate_memory = candidate["whole_process_peak_memory_mib"] / 1024.0

    names = ["LightX2V\nBF16 + Sol", "TeleFuser\nFP8 + Sol"]
    colors = ["#3874a5", "#23806f"]
    panels = (
        ("Denoise time", "seconds / video", [baseline_time, candidate_time]),
        (
            "Denoise throughput",
            "steps / second",
            [baseline_throughput, candidate_throughput],
        ),
        ("Peak memory", "GiB / GPU", [baseline_memory, candidate_memory]),
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.edgecolor": "#cfd6db",
            "axes.labelcolor": "#5b6670",
            "xtick.color": "#46515b",
            "ytick.color": "#69747d",
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(11.4, 6.8))
    for axis, (label, unit, values) in zip(axes, panels, strict=True):
        bars = axis.bar(names, values, width=0.28, color=colors)
        axis.set_title(label, fontsize=14, fontweight=700, color="#17212b", pad=14)
        axis.set_ylabel(unit, fontsize=10)
        axis.grid(axis="y", color="#e7ebee", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#dce2e6")
        axis.spines["bottom"].set_color("#dce2e6")
        axis.tick_params(axis="x", length=0, pad=8)
        axis.margins(x=0.5)
        upper = max(values) * 1.22
        axis.set_ylim(0, upper)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + upper * 0.025,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight=700,
                color="#27323b",
            )

    figure.text(
        0.5,
        0.035,
        "MiniMax-H3 Turbo LoRA | 1344 x 768 | 124 frames | 8 DiT updates | "
        "1 x H100 80GB\n"
        "LightX2V: resident DiT, BF16 + Sol | "
        "TeleFuser: FP8 Linear + FP8 Sol, tau=1.0",
        ha="center",
        color="#69747d",
        fontsize=9.5,
    )
    figure.subplots_adjust(left=0.075, right=0.99, top=0.88, bottom=0.22, wspace=0.35)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, format=args.figure.suffix.removeprefix("."), dpi=180)
    plt.close(figure)

    summary = {
        "source": [str(args.baseline), str(args.candidate)],
        "denoise_time_reduction_percent": reduction(candidate_time, baseline_time),
        "denoise_throughput_increase_percent": (
            candidate_throughput / baseline_throughput - 1.0
        )
        * 100.0,
        "peak_memory_reduction_percent": reduction(candidate_memory, baseline_memory),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
