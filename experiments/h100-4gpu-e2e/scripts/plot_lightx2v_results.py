#!/usr/bin/env python3
"""Plot the matched four-H100 SGLang, LightX2V, and TeleFuser comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def percent_reduction(candidate: float, baseline: float) -> float:
    return (1.0 - candidate / baseline) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    report: dict[str, Any] = json.loads(args.input.read_text(encoding="utf-8"))
    sglang = report["results"]["sglang_matched_h100"]
    baseline = report["results"]["lightx2v_bf16_sageattention2"]
    candidate = report["results"]["telefuser_fp8_sol_exact"]
    names = ["SGLang", "LightX2V", "TeleFuser"]
    colors = ["#8C72A8", "#3874A5", "#23806F"]
    panels = (
        (
            "Generation time",
            "seconds / video",
            [
                sglang["generation_seconds"],
                baseline["generation_seconds"],
                candidate["generation_seconds"],
            ],
        ),
        (
            "Peak memory",
            "GiB / GPU",
            [
                sglang["representative_peak_memory_mib"] / 1024.0,
                baseline["representative_peak_memory_mib"] / 1024.0,
                candidate["representative_peak_memory_mib"] / 1024.0,
            ],
        ),
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
            "svg.fonttype": "none",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 6.8))
    figure.patch.set_facecolor("#ffffff")
    for axis, (label, unit, values) in zip(axes, panels, strict=True):
        bars = axis.bar(names, values, width=0.34, color=colors)
        axis.set_title(label, fontsize=14, fontweight=700, color="#17212b", pad=14)
        axis.set_ylabel(unit, fontsize=10)
        axis.grid(axis="y", color="#e7ebee", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#dce2e6")
        axis.spines["bottom"].set_color("#dce2e6")
        axis.tick_params(axis="x", length=0, pad=8)
        axis.margins(x=0.30)
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
        0.03,
        "MiniMax-H3 Base | 1344 x 768 | 124 frames | 50 steps | "
        "4 x H100 80GB | TP2 x Ulysses SP2",
        ha="center",
        color="#69747d",
        fontsize=9.5,
    )
    figure.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.21, wspace=0.30)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, format=args.figure.suffix.removeprefix("."), dpi=180)
    plt.close(figure)
    svg = args.figure.read_text(encoding="utf-8")
    args.figure.write_text(
        "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
        encoding="utf-8",
    )

    summary = {
        "source": str(args.input),
        "generation_speedup": baseline["generation_seconds"]
        / candidate["generation_seconds"],
        "generation_time_reduction_percent": percent_reduction(
            candidate["generation_seconds"], baseline["generation_seconds"]
        ),
        "denoise_speedup": baseline["denoise_seconds"] / candidate["denoise_seconds"],
        "denoise_time_reduction_percent": percent_reduction(
            candidate["denoise_seconds"], baseline["denoise_seconds"]
        ),
        "denoise_throughput_increase_percent": (
            candidate["config_points_per_second"] / baseline["config_points_per_second"]
            - 1.0
        )
        * 100.0,
        "peak_memory_reduction_percent": percent_reduction(
            candidate["representative_peak_memory_mib"],
            baseline["representative_peak_memory_mib"],
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
