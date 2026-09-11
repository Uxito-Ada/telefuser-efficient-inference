#!/usr/bin/env python3
"""Create the publication chart and derived summary from benchmark reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(report: dict[str, Any], *keys: str) -> float:
    value: Any = report
    for key in keys:
        value = value[key]
    return float(value)


def percent_change(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--telefuser", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    baseline = load(args.baseline)
    telefuser = load(args.telefuser)
    reports = [baseline, telefuser]
    names = ["FastVideo\nBF16 + FA4", "TeleFuser\nFP8 + Sol"]
    colors = ["#3874a5", "#23806f"]
    panels = [
        (
            "Denoise time",
            "seconds / video",
            [
                metric(report, "measurement", "denoising", "median")
                for report in reports
            ],
        ),
        (
            "DiT throughput",
            "forwards / second",
            [
                metric(report, "measurement", "actual_dit_forwards_per_second")
                for report in reports
            ],
        ),
        (
            "Peak memory",
            "GiB / GPU",
            [
                metric(report, "measurement", "max_per_gpu_peak_mib") / 1024.0
                for report in reports
            ],
        ),
    ]

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
    figure, axes = plt.subplots(1, 3, figsize=(12, 6.4))
    figure.patch.set_facecolor("#ffffff")
    for axis, (label, unit, values) in zip(axes, panels, strict=True):
        bars = axis.bar(names, values, width=0.32, color=colors)
        axis.set_title(label, fontsize=14, fontweight=700, color="#17212b", pad=14)
        axis.set_ylabel(unit, fontsize=10)
        axis.grid(axis="y", color="#e7ebee", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#dce2e6")
        axis.spines["bottom"].set_color("#dce2e6")
        axis.tick_params(axis="x", length=0, pad=8)
        axis.margins(x=0.45)
        upper = max(values) * 1.22
        axis.set_ylim(0, upper if upper > 0 else 1)
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

    workload = telefuser["workload"]
    figure.text(
        0.5,
        0.025,
        f"MiniMax-H3 + FastH3 Dense/Data-Free | "
        f"{workload['resolution'][0]} x {workload['resolution'][1]} | "
        f"{workload['frames']} frames | {telefuser['hardware']} | no DiT CPU offload\n"
        "FastVideo: BF16 Linear + FA4 | TeleFuser: FP8 Linear + FP8 Sol, tau=1.0, KV smoothing",
        ha="center",
        color="#69747d",
        fontsize=9.5,
    )
    figure.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.20, wspace=0.34)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, format=args.figure.suffix.removeprefix("."), dpi=180)
    plt.close(figure)

    summary = {
        "baseline": str(args.baseline),
        "telefuser": str(args.telefuser),
        "denoise_time_reduction_percent": -percent_change(
            panels[0][2][1], panels[0][2][0]
        ),
        "dit_throughput_increase_percent": percent_change(
            panels[1][2][1], panels[1][2][0]
        ),
        "peak_memory_change_percent": percent_change(panels[2][2][1], panels[2][2][0]),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
