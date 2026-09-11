#!/usr/bin/env python3
"""Render the English and Chinese Markdown articles as static pages."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import markdown


ROOT = Path(__file__).resolve().parents[1]
EDITIONS: tuple[dict[str, Any], ...] = (
    {
        "lang": "en",
        "source": ROOT / "BLOG.md",
        "output": ROOT / "site" / "index.html",
        "description": "TeleFuser's quality-aware FP8, sparse attention, adapter, and multi-GPU optimization stack for world models.",
        "title": "TeleFuser: Fast and Faithful World Model Inference",
        "system_label": "System",
        "system_href": "#telefusers-optimization-stack",
        "results_label": "Results",
        "results_href": "#results-on-h100",
        "code_label": "Code",
        "language_label": "中文",
        "language_href": "index.zh.html",
        "footer": "TeleFuser Engineering | MiniMax-H3 on NVIDIA H100",
        "reproduce": "Reproduce the benchmark",
    },
    {
        "lang": "zh-CN",
        "source": ROOT / "BLOG.zh.md",
        "output": ROOT / "site" / "index.zh.html",
        "description": "TeleFuser 面向世界模型的高质量 FP8、稀疏注意力、Adapter 与多 GPU 优化。",
        "title": "TeleFuser：高性能、高质量的世界模型推理",
        "system_label": "优化方案",
        "system_href": "#telefuser-optimization-stack",
        "results_label": "测试结果",
        "results_href": "#h100-results",
        "code_label": "代码",
        "language_label": "English",
        "language_href": "index.html",
        "footer": "TeleFuser Engineering | MiniMax-H3 on NVIDIA H100",
        "reproduce": "查看测试方法",
    },
)

MEDIA = {
    "fastvideo-primary": "../sections/04-evaluation/assets/fastvideo-primary.mp4",
    "telefuser-primary": "../sections/04-evaluation/assets/telefuser-primary.mp4",
    "bf16-quality": "../sections/03-quality-and-scale/assets/bf16-fa4.mp4",
    "fp8-unsmoothed": "../sections/03-quality-and-scale/assets/fp8-sol-unsmoothed.mp4",
    "fp8-smoothed": "../sections/03-quality-and-scale/assets/fp8-sol-smoothed.mp4",
}


def render(edition: dict[str, Any]) -> str:
    source = edition["source"].read_text(encoding="utf-8")
    source = re.sub(r"<!--.*?-->", "", source, flags=re.DOTALL)
    source = source.replace("](sections/", "](../sections/")
    source = source.replace("](experiments/", "](../experiments/")
    source = source.replace("](README.md)", "](../README.md)")
    body = markdown.markdown(
        source,
        extensions=["extra", "sane_lists", "smarty", "toc"],
        output_format="html5",
    )
    for slot, path in MEDIA.items():
        body = body.replace(
            f'data-result-slot="{slot}"',
            f'src="{path}" data-result-slot="{slot}"',
        )

    return f"""<!doctype html>
<html lang="{edition["lang"]}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="{edition["description"]}">
  <title>{edition["title"]}</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header class="topbar">
    <a class="brand" href="../README.md">TeleFuser Engineering</a>
    <nav aria-label="Article links">
      <a href="{edition["system_href"]}">{edition["system_label"]}</a>
      <a href="{edition["results_href"]}">{edition["results_label"]}</a>
      <a href="https://github.com/Tele-AI/TeleFuser">{edition["code_label"]}</a>
      <a href="{edition["language_href"]}">{edition["language_label"]}</a>
    </nav>
  </header>
  <main>
    <article>{body}</article>
  </main>
  <footer>
    <span>{edition["footer"]}</span>
    <a href="../experiments/h100-4gpu-e2e/README.md">{edition["reproduce"]}</a>
  </footer>
  <script src="video-sync.js"></script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    stale: list[str] = []
    for edition in EDITIONS:
        generated = render(edition)
        output = edition["output"]
        if args.check:
            current = output.read_text(encoding="utf-8") if output.exists() else ""
            if current != generated:
                stale.append(output.name)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(generated, encoding="utf-8")

    if stale:
        raise SystemExit(f"{', '.join(stale)} stale; run python scripts/build_site.py")


if __name__ == "__main__":
    main()
