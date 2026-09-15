#!/usr/bin/env python3
"""Render the matched one-GPU MiniMax-H3 smoothing ablation as SVG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree as ET


SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)


def add(parent: ET.Element, tag: str, **attrs: object) -> ET.Element:
    return ET.SubElement(parent, f"{{{SVG}}}{tag}", {key: str(value) for key, value in attrs.items()})


def label(parent: ET.Element, value: str, x: float, y: float, **attrs: object) -> None:
    node = add(parent, "text", x=x, y=y, **attrs)
    node.text = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()

    records = [
        json.loads((args.raw_dir / name).read_text(encoding="utf-8"))
        for name in ("bf16-fa4.json", "fp8-sol-unsmoothed.json", "fp8-sol-smoothed.json")
    ]
    panels = (
        ("Denoise time", "seconds / video", [r["runtime_metrics"]["denoising_seconds"] for r in records]),
        ("Denoise throughput", "steps / second", [r["denoising_steps_per_second"] for r in records]),
        ("Peak allocated memory", "GiB", [r["runtime_metrics"]["peak_allocated_bytes"] / 2**30 for r in records]),
    )
    names = ("BF16 + FA4", "FP8 raw", "FP8 smoothing")
    colors = ("#5379A5", "#D57C57", "#23806F")
    width, height = 1200, 650
    root = ET.Element(f"{{{SVG}}}svg", {"viewBox": f"0 0 {width} {height}", "width": str(width), "height": str(height)})
    add(root, "rect", x=0, y=0, width=width, height=height, fill="#FFFFFF")
    style = add(root, "style")
    style.text = "text { font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; }"
    top, bottom = 105, 470
    for start, (title, unit, values) in zip((60, 450, 840), panels, strict=True):
        label(root, title, start + 150, 42, **{"text-anchor": "middle", "font-size": 20, "font-weight": 700, "fill": "#17212B"})
        label(root, unit, start, 76, **{"font-size": 14, "fill": "#65727B"})
        upper = max(values) * 1.18
        for tick in range(5):
            ratio = tick / 4
            y = bottom - ratio * (bottom - top)
            add(root, "line", x1=start, y1=y, x2=start + 300, y2=y, stroke="#E7ECEF", **{"stroke-width": 1})
        for center, name, color, value in zip((122, 210, 298), names, colors, values, strict=True):
            bar_height = value / upper * (bottom - top)
            y = bottom - bar_height
            add(root, "rect", x=start + center - 81, y=y, width=42, height=bar_height, rx=2, fill=color)
            label(root, f"{value:.2f}", start + center - 60, y - 12, **{"text-anchor": "middle", "font-size": 13, "font-weight": 700, "fill": "#26343D"})
            label(root, name, start + center - 60, bottom + 30, **{"text-anchor": "middle", "font-size": 12, "font-weight": 600, "fill": "#38464F"})
        add(root, "line", x1=start, y1=bottom, x2=start + 300, y2=bottom, stroke="#CBD5DA", **{"stroke-width": 1})
    label(root, "MiniMax-H3 Base · 1344 × 768 · 107 frames · 4 seconds · 50 steps · 1 × H100", width / 2, 557, **{"text-anchor": "middle", "font-size": 15, "font-weight": 600, "fill": "#4D5B64"})
    label(root, "seed 17 · FP8 Sol: tau 1.0, exact routing, 10 dense steps, 2 dense layers · no warm-up", width / 2, 588, **{"text-anchor": "middle", "font-size": 14, "fill": "#69767F"})
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(args.figure, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    main()
