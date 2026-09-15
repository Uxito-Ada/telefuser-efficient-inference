#!/usr/bin/env python3
"""Plot matched TeleFuser MiniMax-H3 Base one-, two-, and four-GPU results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree as ET


SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)


def element(parent: ET.Element, tag: str, **attributes: object) -> ET.Element:
    return ET.SubElement(parent, f"{{{SVG}}}{tag}", {key: str(value) for key, value in attributes.items()})


def text(parent: ET.Element, value: str, x: float, y: float, **attributes: object) -> None:
    node = element(parent, "text", x=x, y=y, **attributes)
    node.text = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.input.read_text(encoding="utf-8"))
    one = report["results"]["1_gpu"]
    two = report["results"]["2_gpu"]
    four = report["results"]["4_gpu"]
    panels = (
        (
            "Denoise time",
            "seconds / video",
            [one["denoise_seconds"], two["denoise_seconds"], four["denoise_seconds"]],
        ),
        (
            "Denoise throughput",
            "steps / second",
            [
                one["config_points_per_second"],
                two["config_points_per_second"],
                four["config_points_per_second"],
            ],
        ),
        (
            "Peak memory",
            "GiB / GPU",
            [
                one["peak_memory_mib"] / 1024.0,
                two["peak_memory_mib"] / 1024.0,
                four["peak_memory_mib"] / 1024.0,
            ],
        ),
    )

    width, height = 1200, 640
    root = ET.Element(
        f"{{{SVG}}}svg",
        {"viewBox": f"0 0 {width} {height}", "width": str(width), "height": str(height)},
    )
    element(root, "rect", x=0, y=0, width=width, height=height, fill="#FFFFFF")
    style = element(root, "style")
    style.text = "text { font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; }"

    chart_top, chart_bottom = 100, 475
    panel_width = 300
    panel_starts = (60, 450, 840)
    colors = ("#7894A8", "#D48B4E", "#167D6B")
    names = ("1 GPU", "2 GPUs", "4 GPUs")

    for start, (title, unit, values) in zip(panel_starts, panels, strict=True):
        text(root, title, start + panel_width / 2, 42, **{"text-anchor": "middle", "font-size": 20, "font-weight": 700, "fill": "#16232C"})
        text(root, unit, start, 76, **{"font-size": 14, "fill": "#65727B"})
        upper = max(values) * 1.18
        for tick in range(5):
            ratio = tick / 4
            y = chart_bottom - ratio * (chart_bottom - chart_top)
            element(root, "line", x1=start, y1=y, x2=start + panel_width, y2=y, stroke="#E7ECEF", **{"stroke-width": 1})
            text(root, f"{upper * ratio:.1f}", start - 10, y + 5, **{"text-anchor": "end", "font-size": 12, "fill": "#74818A"})

        bar_width = 42
        centers = (start + 62, start + 150, start + 238)
        for center, name, color, value in zip(centers, names, colors, values, strict=True):
            bar_height = value / upper * (chart_bottom - chart_top)
            y = chart_bottom - bar_height
            element(root, "rect", x=center - bar_width / 2, y=y, width=bar_width, height=bar_height, rx=2, fill=color)
            text(root, f"{value:.2f}", center, y - 12, **{"text-anchor": "middle", "font-size": 14, "font-weight": 700, "fill": "#26343D"})
            text(root, name, center, chart_bottom + 31, **{"text-anchor": "middle", "font-size": 14, "font-weight": 600, "fill": "#38464F"})
        element(root, "line", x1=start, y1=chart_bottom, x2=start + panel_width, y2=chart_bottom, stroke="#CBD5DA", **{"stroke-width": 1})

    text(
        root,
        "MiniMax-H3 Base · 1344 × 768 · 124 frames · 50 steps · FP8 Linear + FP8 Sol",
        width / 2,
        555,
        **{"text-anchor": "middle", "font-size": 15, "font-weight": 600, "fill": "#4D5B64"},
    )
    text(
        root,
        "1 GPU: local · 2 GPUs: TP2 · 4 GPUs: TP2 × Ulysses SP2 · feature cache disabled",
        width / 2,
        585,
        **{"text-anchor": "middle", "font-size": 14, "fill": "#69767F"},
    )

    args.figure.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(args.figure, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    main()
