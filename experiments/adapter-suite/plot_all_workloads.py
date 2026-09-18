#!/usr/bin/env python3
"""Render the four-GPU overview and unified speed/memory figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
        "LightX2V\nBase H3",
        "FastVideo\nBase H3",
        "SGLang\nBase H3",
        "TeleFuser\nBase H3",
        "LightX2V\nTurbo",
        "SGLang\nTurbo",
        "TeleFuser\nTurbo",
        "FastVideo\nFastH3",
        "TeleFuser\nFastH3",
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
    frameworks = [label.split("\n", 1)[0] for label in labels]
    return labels, throughput, memory, frameworks


def plot_unified() -> None:
    labels, throughput, memory, frameworks = experiment_data()
    palette = {
        "LightX2V": "#5B7DB1",
        "FastVideo": "#E68A36",
        "SGLang": "#8B6BB7",
        "TeleFuser": "#168A72",
    }
    colors = [palette[name] for name in frameworks]
    x = np.arange(len(labels), dtype=float)

    fig, memory_axis = plt.subplots(figsize=(17.2, 8.4), facecolor="white")
    memory_axis.set_facecolor("white")
    bars = memory_axis.bar(x, memory, width=0.42, color=colors, alpha=0.88, zorder=2)
    memory_axis.set_ylabel("Peak GPU memory (GiB)", color="#344054", labelpad=10)
    memory_axis.set_ylim(0, max(memory) * 1.30)
    memory_axis.set_xticks(x, labels, fontsize=9.5)
    memory_axis.tick_params(axis="x", length=0, pad=11)
    memory_axis.tick_params(axis="y", colors="#667085")
    memory_axis.grid(axis="y", color="#E7EBEF", linewidth=0.8, zorder=0)
    memory_axis.spines[["top", "right"]].set_visible(False)
    memory_axis.spines[["left", "bottom"]].set_color("#D0D5DD")
    for bar, value in zip(bars, memory, strict=True):
        memory_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.25,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#344054",
        )

    speed_axis = memory_axis.twinx()
    speed_axis.plot(
        x,
        throughput,
        color="#C93F4B",
        marker="o",
        markersize=5.2,
        linewidth=1.45,
        zorder=4,
    )
    speed_axis.set_ylabel("End-to-end throughput (5 s videos/hour)", color="#C93F4B", labelpad=12)
    speed_axis.set_ylim(0, max(throughput) * 1.25)
    speed_axis.tick_params(axis="y", colors="#C93F4B")
    speed_axis.spines["top"].set_visible(False)
    speed_axis.spines["right"].set_color("#C93F4B")

    memory_axis.annotate(
        "↓ lower is better",
        xy=(0.01, 0.965),
        xycoords="axes fraction",
        color="#344054",
        fontsize=10,
        ha="left",
        va="top",
    )
    speed_axis.annotate(
        "↑ higher is better",
        xy=(0.995, 0.965),
        xycoords="axes fraction",
        color="#C93F4B",
        fontsize=10,
        ha="right",
        va="top",
    )
    for boundary in (3.5, 6.5):
        memory_axis.axvline(boundary, color="#D0D5DD", linewidth=0.8, zorder=1)
    for center, caption in ((1.5, "Base H3"), (5.0, "Turbo LoRA"), (7.5, "FastH3")):
        memory_axis.text(
            center,
            -0.18,
            caption,
            transform=memory_axis.get_xaxis_transform(),
            ha="center",
            color="#667085",
            fontsize=10,
        )

    fig.subplots_adjust(left=0.075, right=0.91, top=0.96, bottom=0.24)
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

    fig, axis = plt.subplots(figsize=(12.8, 6.8), facecolor="white")
    axis.set_facecolor("white")
    bars = axis.bar(x, hero_values, width=0.48, color=colors, zorder=2)
    axis.set_ylabel("End-to-end throughput (5 s videos/hour)", color="#344054")
    axis.set_xticks(x, hero_labels, fontsize=10)
    axis.tick_params(axis="x", length=0, pad=10)
    axis.tick_params(axis="y", colors="#667085")
    axis.grid(axis="y", color="#E7EBEF", linewidth=0.8, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#D0D5DD")
    axis.set_ylim(0, max(hero_values) * 1.20)
    for bar, value in zip(bars, hero_values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(hero_values) * 0.025,
            f"{value:.0f}",
            ha="center",
            va="bottom",
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
    fig.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.20)
    HERO_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(HERO_OUT, format="svg", facecolor="white")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    plot_unified()
    plot_hero()


if __name__ == "__main__":
    main()
