#!/usr/bin/env python3
"""Render the four-GPU overview and unified speed/memory figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patheffects import SimplePatchShadow, withStroke


ROOT = Path(__file__).resolve().parents[2]
UNIFIED_OUT = ROOT / "sections/04-evaluation/assets/all-workloads-performance.svg"
HERO_OUT = ROOT / "sections/00-introduction/assets/four-gpu-throughput.svg"


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def experiment_data() -> tuple[list[str], list[float], list[float], list[str]]:
    base = load("experiments/h100-4gpu-e2e/raw/lightx2v-base-h3-comparison.json")["results"]
    light_turbo = load("experiments/adapter-suite/raw/turbo-lightx2v-4gpu.json")
    sglang_turbo = load("experiments/adapter-suite/raw/turbo-sglang-4gpu.json")
    fastvideo_fasth3 = load("experiments/adapter-suite/raw/fasth3-fastvideo-4gpu.json")

    labels = [
        "LightX2V Base H3",
        "FastVideo Base H3",
        "SGLang Base H3",
        "TeleFuser Base H3",
        "LightX2V Turbo",
        "SGLang Turbo",
        "TeleFuser Turbo",
        "FastVideo FastH3",
        "TeleFuser FastH3",
    ]
    generation_seconds = [
        base["lightx2v_bf16_sageattention2"]["generation_seconds"],
        base["fastvideo_bf16_fa4"]["generation_seconds"],
        base["sglang_matched_h100"]["generation_seconds"],
        base["telefuser_fp8_sol_exact"]["generation_seconds"],
        light_turbo["measurement"]["pipeline_seconds"],
        sglang_turbo["measurement"]["generation_seconds"],
        13.574335,
        fastvideo_fasth3["measurement"]["generation_seconds"]["median"],
        7.029866,
    ]
    throughput = [3600.0 / value for value in generation_seconds]
    memory = [
        base["lightx2v_bf16_sageattention2"]["representative_peak_memory_mib"] / 1024,
        base["fastvideo_bf16_fa4"]["representative_peak_memory_mib"] / 1024,
        base["sglang_matched_h100"]["representative_peak_memory_mib"] / 1024,
        base["telefuser_fp8_sol_exact"]["representative_peak_memory_mib"] / 1024,
        light_turbo["measurement"]["peak_reserved_gib"],
        sglang_turbo["measurement"]["peak_reserved_mib"] / 1024,
        43334 / 1024,
        fastvideo_fasth3["measurement"]["observed_peak_memory_mib"] / 1024,
        43264 / 1024,
    ]
    frameworks = [label.split(" ", 1)[0] for label in labels]
    return labels, throughput, memory, frameworks


def plot_unified() -> None:
    labels, throughput, memory, frameworks = experiment_data()
    palette = {
        "LightX2V": "#5B7DB1",
        "FastVideo": "#E68A36",
        "SGLang": "#8B6BB7",
        "TeleFuser": "#168A72",
    }
    groups = (("Base H3", [0, 1, 2, 3]), ("Turbo LoRA", [4, 5, 6]), ("FastH3", [7, 8]))
    fig, axes = plt.subplots(1, 3, figsize=(20.5, 7.6), facecolor="white")
    for axis, (title, indices) in zip(axes, groups, strict=True):
        axis.set_facecolor("white")
        x = np.arange(len(indices), dtype=float)
        local_memory = [memory[i] for i in indices]
        local_throughput = [throughput[i] for i in indices]
        local_frameworks = [frameworks[i] for i in indices]
        local_colors = [palette[name] for name in local_frameworks]
        bars = axis.bar(x, local_memory, width=0.56, color=local_colors, alpha=0.9, zorder=2)
        axis.set_title(title, color="#17212b", fontsize=14, fontweight=700, pad=14)
        axis.set_ylabel("Peak GPU memory (GiB)", color="#344054", labelpad=8)
        axis.set_ylim(0, max(local_memory) * 1.32)
        axis.set_xticks(x, local_frameworks, fontsize=9.5)
        axis.tick_params(axis="x", length=0, pad=8)
        axis.tick_params(axis="y", colors="#667085")
        axis.grid(axis="y", color="#E7EBEF", linewidth=0.8, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#D0D5DD")
        for bar, value, framework in zip(bars, local_memory, local_frameworks, strict=True):
            if framework == "TeleFuser":
                bar.set_edgecolor("#075E4C")
                bar.set_linewidth(1.8)
                bar.set_path_effects([SimplePatchShadow(offset=(2, -2), alpha=0.35), withStroke(linewidth=2.5, foreground="#075E4C")])
            axis.text(bar.get_x() + bar.get_width() / 2, value + max(local_memory) * 0.025, f"{value:.1f}", ha="center", va="bottom", fontsize=8.5, color="#344054")

        speed_axis = axis.twinx()
        speed_axis.plot(x, local_throughput, color="#C93F4B", linewidth=1.6, zorder=4)
        for point_x, point_y, framework in zip(x, local_throughput, local_frameworks, strict=True):
            marker = "*" if framework == "TeleFuser" else "o"
            size = 12 if marker == "*" else 5
            speed_axis.plot(point_x, point_y, marker=marker, color="#C93F4B", markersize=size, markeredgecolor="white", markeredgewidth=0.8, zorder=5)
        speed_axis.set_ylim(0, max(local_throughput) * 1.28)
        speed_axis.tick_params(axis="y", colors="#C93F4B")
        speed_axis.spines["top"].set_visible(False)
        speed_axis.spines["right"].set_color("#C93F4B")
        speed_axis.set_ylabel("Throughput (5 s videos/hour)", color="#C93F4B", labelpad=8)
        axis.annotate("↓ lower is better", xy=(0.02, 0.97), xycoords="axes fraction", color="#344054", fontsize=9, ha="left", va="top")
        speed_axis.annotate("↑ higher is better", xy=(0.98, 0.97), xycoords="axes fraction", color="#C93F4B", fontsize=9, ha="right", va="top")

    legend_handles = [Patch(facecolor=color, edgecolor="none", label=name) for name, color in palette.items()]
    legend_handles.extend([Line2D([0], [0], color="#C93F4B", marker="o", linewidth=1.6, markersize=5, label="Other framework throughput"), Line2D([0], [0], color="#C93F4B", marker="*", linewidth=1.6, markersize=10, label="TeleFuser throughput")])
    fig.legend(handles=legend_handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.005), fontsize=9.5, columnspacing=1.5)
    fig.subplots_adjust(left=0.045, right=0.965, top=0.90, bottom=0.18, wspace=0.38)
    UNIFIED_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(UNIFIED_OUT, format="svg", facecolor="white")
    plt.close(fig)


def plot_hero() -> None:
    labels, throughput, _, frameworks = experiment_data()
    selected = [0, 1, 2, 3, 6]
    hero_labels = [labels[index] for index in selected]
    hero_values = [throughput[index] for index in selected]
    colors = ["#B7C3CE" if frameworks[index] != "TeleFuser" else "#168A72" for index in selected]
    x = np.arange(len(selected), dtype=float)

    fig, axis = plt.subplots(figsize=(13.6, 5.8), facecolor="white")
    axis.set_facecolor("white")
    bars = axis.barh(x, hero_values, height=0.56, color=colors, zorder=2)
    axis.set_xlabel("End-to-end throughput (5 s videos/hour)", color="#344054")
    axis.set_yticks(x, hero_labels, fontsize=10)
    axis.tick_params(axis="y", length=0, pad=10)
    axis.tick_params(axis="x", colors="#667085")
    axis.grid(axis="x", color="#E7EBEF", linewidth=0.8, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#D0D5DD")
    axis.set_xlim(0, max(hero_values) * 1.18)
    for bar, value in zip(bars, hero_values, strict=True):
        axis.text(
            value + max(hero_values) * 0.018,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0f}",
            ha="left",
            va="center",
            fontsize=10,
            color="#344054",
        )
    axis.annotate(
        "↑ higher is better",
        xy=(0.99, 0.96),
        xycoords="axes fraction",
        ha="right",
        va="top",
        color="#168A72",
        fontsize=10,
    )
    axis.invert_yaxis()
    fig.subplots_adjust(left=0.23, right=0.98, top=0.94, bottom=0.18)
    HERO_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(HERO_OUT, format="svg", facecolor="white")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    plot_unified()
    plot_hero()


if __name__ == "__main__":
    main()
