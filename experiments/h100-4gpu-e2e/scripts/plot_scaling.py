#!/usr/bin/env python3
"""Plot matched TeleFuser MiniMax-H3 Base one-, two-, and four-GPU results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.input.read_text(encoding="utf-8"))
    keys = ("1_gpu", "2_gpu", "4_gpu")
    names = ("1 GPU", "2 GPUs", "4 GPUs")
    results = [report["results"][key] for key in keys]
    x = np.arange(len(results), dtype=float)
    denoise = np.array([item["denoise_seconds"] for item in results], dtype=float)
    throughput = np.array([item["config_points_per_second"] for item in results], dtype=float)
    memory = np.array([item["peak_memory_mib"] / 1024.0 for item in results], dtype=float)

    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, memory_axis = plt.subplots(figsize=(13.5, 6.8), facecolor="white")
    memory_axis.set_facecolor("white")
    bars = memory_axis.bar(
        x, memory, width=0.40, color=("#9AAFC0", "#D28A4F", "#168A72"),
        edgecolor=("#667F91", "#A66732", "#075E4C"), linewidth=1.2, zorder=2,
    )
    memory_axis.set_ylabel("Peak GPU memory (GiB)", color="#344054", fontsize=11, labelpad=10)
    memory_axis.set_ylim(0, max(memory) * 1.30)
    memory_axis.set_xticks(x, names, fontsize=11)
    memory_axis.tick_params(axis="x", length=0, pad=10)
    memory_axis.tick_params(axis="y", colors="#667085", labelsize=10)
    memory_axis.grid(axis="y", color="#E7EBEF", linewidth=0.8, zorder=0)
    memory_axis.spines[["top", "right"]].set_visible(False)
    memory_axis.spines[["left", "bottom"]].set_color("#D0D5DD")
    for bar, value in zip(bars, memory, strict=True):
        memory_axis.text(
            bar.get_x() + bar.get_width() / 2, value + max(memory) * 0.025,
            f"{value:.1f}", ha="center", va="bottom", fontsize=10.5,
            color="#344054", fontweight=600,
        )

    time_axis = memory_axis.twinx()
    time_axis.plot(
        x, denoise, color="#C93F4B", marker="o", markersize=7, linewidth=2.0,
        markeredgecolor="white", markeredgewidth=1.0, zorder=4,
    )
    time_axis.set_ylabel("Denoise time (s / video)", color="#C93F4B", fontsize=11, labelpad=10)
    time_axis.set_ylim(0, max(denoise) * 1.28)
    time_axis.tick_params(axis="y", colors="#C93F4B", labelsize=10)
    time_axis.spines["top"].set_visible(False)
    time_axis.spines["right"].set_color("#C93F4B")
    for point_x, value in zip(x, denoise, strict=True):
        time_axis.text(point_x, value + max(denoise) * 0.035, f"{value:.1f}", ha="center", va="bottom", fontsize=10, color="#C93F4B")

    throughput_axis = memory_axis.twinx()
    throughput_axis.spines["right"].set_position(("outward", 58))
    throughput_axis.plot(
        x, throughput, color="#1769AA", marker="s", markersize=6.5, linewidth=2.0,
        markeredgecolor="white", markeredgewidth=1.0, zorder=5,
    )
    throughput_axis.set_ylabel("Denoise throughput (steps / s)", color="#1769AA", fontsize=11, labelpad=10)
    throughput_axis.set_ylim(0, max(throughput) * 1.55)
    throughput_axis.tick_params(axis="y", colors="#1769AA", labelsize=10)
    throughput_axis.spines["top"].set_visible(False)
    throughput_axis.spines["right"].set_color("#1769AA")
    for point_x, value in zip(x, throughput, strict=True):
        throughput_axis.text(point_x, value + max(throughput) * 0.06, f"{value:.2f}", ha="center", va="bottom", fontsize=10, color="#1769AA")

    memory_axis.annotate("↓ lower is better", xy=(0.02, 0.98), xycoords="axes fraction", ha="left", va="top", fontsize=10, color="#344054")
    time_axis.annotate("↓ lower is better", xy=(0.98, 0.98), xycoords="axes fraction", ha="right", va="top", fontsize=10, color="#C93F4B")
    throughput_axis.annotate("↑ higher is better", xy=(1.18, 0.98), xycoords="axes fraction", ha="right", va="top", fontsize=10, color="#1769AA")

    handles = [
        Patch(facecolor="#168A72", edgecolor="#075E4C", label="Peak GPU memory"),
        Line2D([0], [0], color="#C93F4B", marker="o", linewidth=2, label="Denoise time"),
        Line2D([0], [0], color="#1769AA", marker="s", linewidth=2, label="Denoise throughput"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.035), fontsize=10.5, columnspacing=2.2)
    fig.subplots_adjust(left=0.09, right=0.82, top=0.94, bottom=0.20)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, format="svg", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
