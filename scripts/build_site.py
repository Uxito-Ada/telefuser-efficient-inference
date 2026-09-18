#!/usr/bin/env python3
"""Render the English and Chinese Markdown articles as static pages."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Any

import markdown


ROOT = Path(__file__).resolve().parents[1]
EDITIONS: tuple[dict[str, Any], ...] = (
    {
        "lang": "en",
        "source": ROOT / "BLOG.md",
        "output": ROOT / "site" / "index.html",
        "description": "Q-SPA co-designs quality-aware FP8, dynamic block sparsity, and distributed attention for world models with TeleFuser.",
        "title": "Q-SPA: Quantized Sparse-Parallel Attention for World Models with TeleFuser",
        "system_label": "System",
        "system_href": "#q-spa-system",
        "results_label": "Results",
        "results_href": "#results",
        "code_label": "Code",
        "language_label": "中文",
        "language_href": "index.zh.html",
        "footer": "Q-SPA | Quantized Sparse-Parallel Attention with TeleFuser",
        "reproduce": "Reproduce the benchmark",
    },
    {
        "lang": "zh-CN",
        "source": ROOT / "BLOG.zh.md",
        "output": ROOT / "site" / "index.zh.html",
        "description": "Q-SPA 在 TeleFuser 中协同设计面向世界模型的 FP8、动态块稀疏与分布式 attention。",
        "title": "Q-SPA：用 TeleFuser 实现量化、稀疏与并行协同的世界模型推理",
        "system_label": "优化方案",
        "system_href": "#q-spa-system",
        "results_label": "测试结果",
        "results_href": "#results",
        "code_label": "代码",
        "language_label": "English",
        "language_href": "index.html",
        "footer": "Q-SPA | Quantized Sparse-Parallel Attention with TeleFuser",
        "reproduce": "查看测试方法",
    },
)

MEDIA = {
    "lightx2v-base-h3": "assets/lightx2v-base-h3.mp4",
    "telefuser-base-h3": "assets/telefuser-base-h3.mp4",
    "fastvideo-base-h3": "assets/fastvideo-base-h3.mp4",
    "sglang-base-h3": "assets/sglang-base-h3.mp4",
    "turbo-lightx2v": "assets/turbo-lightx2v.mp4",
    "turbo-sglang": "assets/turbo-sglang.mp4",
    "turbo-telefuser": "assets/turbo-telefuser.mp4",
    "fastvideo-primary": "assets/fastvideo-primary.mp4",
    "telefuser-primary": "assets/telefuser-primary.mp4",
    "bf16-quality": "assets/bf16-fa4.mp4",
    "fp8-unsmoothed": "assets/fp8-sol-unsmoothed.mp4",
    "fp8-smoothed": "assets/fp8-sol-smoothed.mp4",
}

SITE_ASSET_SOURCES = tuple(sorted((ROOT / "sections").glob("*/assets/*")))


def copy_site_assets() -> None:
    destination = ROOT / "site" / "assets"
    destination.mkdir(parents=True, exist_ok=True)
    for source in SITE_ASSET_SOURCES:
        if source.is_file() and source.suffix.lower() in {".mp4", ".svg", ".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            shutil.copy2(source, destination / source.name)


def render(edition: dict[str, Any]) -> str:
    source = edition["source"].read_text(encoding="utf-8")
    source = re.sub(r"<!--.*?-->", "", source, flags=re.DOTALL)
    source = re.sub(r"\]\(sections/[^)]*/assets/([^)]*)\)", r"](assets/\1)", source)
    source = re.sub(r"src=\"sections/[^/\"]*/assets/([^\"]+)\"", r"src=\"assets/\1\"", source)
    source = source.replace("](sections/", "](../sections/")
    source = source.replace('src="sections/', 'src="../sections/')
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
  <meta name="author" content="Heyang Sun">
  <title>{edition["title"]}</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header class="topbar">
    <a class="brand" href="../README.md">TeleFuser</a>
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

    if not args.check:
        copy_site_assets()

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
