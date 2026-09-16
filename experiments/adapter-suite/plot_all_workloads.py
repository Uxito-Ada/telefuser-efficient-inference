#!/usr/bin/env python3
"""Plot one compact speed/memory comparison across Base and adapters."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "sections/04-evaluation/assets/all-workloads-performance.svg"


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main() -> None:
    base = load("experiments/h100-4gpu-e2e/raw/lightx2v-base-h3-comparison.json")["results"]
    scaling = load("experiments/h100-4gpu-e2e/raw/telefuser-base-h3-scaling.json")["results"]
    turbo_lx = load("experiments/adapter-suite/raw/turbo-lightx2v-bf16-resident.json")
    turbo_tf = load("experiments/adapter-suite/raw/turbo-telefuser-fp8.json")
    fasth3_fv = load("experiments/h100-4gpu-e2e/raw/fastvideo-single-h100.json")["measurement"]
    fasth3_tf = load("experiments/h100-4gpu-e2e/raw/telefuser-single-h100.json")["measurement"]

    labels = [
        "LightX2V\\nBase H3\\n4 GPU",
        "FastVideo\\nBase H3\\n4 GPU",
        "SGLang\\nBase H3\\n4 GPU",
        "TeleFuser\\nBase H3\\n4 GPU",
        "FastVideo\\nFastH3\\n1 GPU",
        "TeleFuser\\nFastH3\\n1 GPU",
        "LightX2V\\nTurbo LoRA\\n1 GPU",
        "TeleFuser\\nTurbo LoRA\\n1 GPU",
    ]
    speed = [
        50 / base["lightx2v_bf16_sageattention2"]["denoise_seconds"],
        50 / base["fastvideo_bf16_fa4"]["denoise_seconds"],
        50 / base["sglang_matched_h100"]["generation_seconds"],
        50 / base["telefuser_fp8_sol_exact"]["denoise_seconds"],
        fasth3_fv["actual_dit_forwards_per_second"],
        fasth3_tf["actual_dit_forwards_per_second"],
        turbo_lx["steps_per_second"],
        turbo_tf["scheduler_points_per_second"],
    ]
    memory = [
        base["lightx2v_bf16_sageattention2"]["representative_peak_memory_mib"] / 1024,
        base["fastvideo_bf16_fa4"]["representative_peak_memory_mib"] / 1024,
        base["sglang_matched_h100"]["representative_peak_memory_mib"] / 1024,
        base["telefuser_fp8_sol_exact"]["representative_peak_memory_mib"] / 1024,
        fasth3_fv["max_per_gpu_peak_mib"] / 1024,
        fasth3_tf["max_per_gpu_peak_mib"] / 1024,
        turbo_lx["formal_run_peak_memory_mib"] / 1024,
        turbo_tf["whole_process_peak_memory_mib"] / 1024,
    ]
    framework_colors = {
        "LightX2V": "#4C78A8",
        "FastVideo": "#F58518",
        "SGLang": "#8E6BBE",
        "TeleFuser": "#159A84",
    }
    colors = [framework_colors[label.split("\\n")[0]] for label in labels]
    x = list(range(len(labels)))
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, memory_axis = plt.subplots(figsize=(15, 7.6), facecolor="white")
    memory_axis.set_facecolor("white")
    bars = memory_axis.bar(x, memory, width=0.52, color=colors, alpha=0.82, zorder=2)
    memory_axis.set_ylabel("Peak GPU memory (GiB)", color="#344054", labelpad=10)
    memory_axis.set_ylim(0, max(memory) * 1.24)
    memory_axis.set_xticks(x, labels, fontsize=9)
    memory_axis.tick_params(axis="x", length=0, pad=10)
    memory_axis.tick_params(axis="y", colors="#667085")
    memory_axis.grid(axis="y", color="#E8ECF0", linewidth=0.8, zorder=0)
    memory_axis.spines[["top", "right"]].set_visible(False)
    memory_axis.spines[["left", "bottom"]].set_color("#D0D5DD")
    for bar, value in zip(bars, memory, strict=True):
        memory_axis.text(bar.get_x() + bar.get_width() / 2, value + max(memory) * 0.018, f"{value:.1f}", ha="center", va="bottom", fontsize=8.5, color="#344054")
    speed_axis = memory_axis.twinx()
    speed_axis.plot(x, speed, color="#D64550", marker="o", markersize=5.5, linewidth=1.7, zorder=4)
    speed_axis.set_ylabel("Denoising throughput (steps/s; higher is better)", color="#D64550", labelpad=12)
    speed_axis.set_ylim(0, max(speed) * 1.34)
    speed_axis.tick_params(axis="y", colors="#D64550")
    speed_axis.spines["top"].set_visible(False)
    speed_axis.spines["right"].set_color("#D64550")
    speed_axis.annotate("↑ higher is better", xy=(1.01, 0.97), xycoords="axes fraction", color="#D64550", fontsize=10, ha="left", va="top")
    memory_axis.annotate("↓ lower is better", xy=(0.01, 0.97), xycoords="axes fraction", color="#344054", fontsize=10, ha="left", va="top")
    for start, end in ((0, 3), (4, 5), (6, 7)):
        memory_axis.axvline(end + 0.5, color="#D0D5DD", linewidth=0.8, zorder=1)
    memory_axis.text(1.5, -0.17, "Base H3", transform=memory_axis.get_xaxis_transform(), ha="center", color="#667085", fontsize=10)
    memory_axis.text(4.5, -0.17, "FastH3", transform=memory_axis.get_xaxis_transform(), ha="center", color="#667085", fontsize=10)
    memory_axis.text(6.5, -0.17, "Turbo LoRA", transform=memory_axis.get_xaxis_transform(), ha="center", color="#667085", fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.89, top=0.95, bottom=0.23)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, format="svg", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
