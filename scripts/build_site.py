#!/usr/bin/env python3
"""Render the assembled Markdown article as a self-contained static page."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import markdown


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "BLOG.md"
OUTPUT = ROOT / "site" / "index.html"


def render() -> str:
    source = SOURCE.read_text(encoding="utf-8")
    source = re.sub(r"<!--.*?-->", "", source, flags=re.DOTALL)
    source = source.replace("](sections/", "](../sections/")
    source = source.replace("](experiments/", "](../experiments/")
    source = source.replace("](README.md)", "](../README.md)")
    body = markdown.markdown(
        source,
        extensions=["extra", "sane_lists", "smarty"],
        output_format="html5",
    )
    body = body.replace(
        'data-result-slot="fastvideo-primary"',
        'src="../sections/04-evaluation/assets/fastvideo-primary.mp4" '
        'data-result-slot="fastvideo-primary"',
    )
    body = body.replace(
        'data-result-slot="telefuser-primary"',
        'src="../sections/04-evaluation/assets/telefuser-primary.mp4" '
        'data-result-slot="telefuser-primary"',
    )
    body = body.replace(
        'data-result-slot="bf16-quality"',
        'src="../sections/03-quality-and-scale/assets/bf16-fa4.mp4" '
        'data-result-slot="bf16-quality"',
    )
    body = body.replace(
        'data-result-slot="fp8-unsmoothed"',
        'src="../sections/03-quality-and-scale/assets/fp8-sol-unsmoothed.mp4" '
        'data-result-slot="fp8-unsmoothed"',
    )
    body = body.replace(
        'data-result-slot="fp8-smoothed"',
        'src="../sections/03-quality-and-scale/assets/fp8-sol-smoothed.mp4" '
        'data-result-slot="fp8-smoothed"',
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Co-designing FP8, sparse attention, and multi-GPU execution for MiniMax-H3 in TeleFuser.">
  <title>Fast and Faithful MiniMax-H3 Inference on H100</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header class="topbar">
    <a class="brand" href="../README.md">TeleFuser Engineering</a>
    <nav aria-label="Article links">
      <a href="#building-the-h100-execution-path">System</a>
      <a href="#what-the-complete-path-actually-buys">Results</a>
      <a href="https://github.com/Tele-AI/TeleFuser">Code</a>
    </nav>
  </header>
  <main>
    <article>{body}</article>
  </main>
  <footer>
    <span>TeleFuser Engineering | MiniMax-H3 on NVIDIA H100</span>
    <a href="../experiments/h100-4gpu-e2e/README.md">Reproduce the benchmark</a>
  </footer>
  <script src="video-sync.js"></script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generated = render()
    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != generated:
            raise SystemExit(
                "site/index.html is stale; run python scripts/build_site.py"
            )
        return
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(generated, encoding="utf-8")


if __name__ == "__main__":
    main()
