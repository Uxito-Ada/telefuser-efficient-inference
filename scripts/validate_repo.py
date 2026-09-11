#!/usr/bin/env python3
"""Validate section metadata, local Markdown links, and asset size limits."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_ASSET_BYTES = 100 * 1024 * 1024


def main() -> None:
    errors: list[str] = []
    sections = sorted((ROOT / "sections").glob("*"))
    for section in sections:
        if not section.is_dir():
            continue
        for required in (
            "README.md",
            "README.zh.md",
            "section.yaml",
            "assets/MANIFEST.md",
        ):
            if not (section / required).is_file():
                errors.append(f"missing {section.relative_to(ROOT) / required}")
        readme = section / "README.md"
        if readme.is_file() and "SECTION-CONTRACT" not in readme.read_text():
            errors.append(f"missing contract in {readme.relative_to(ROOT)}")

    markdown_files = list(ROOT.rglob("*.md"))
    link_pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
    for markdown in markdown_files:
        for target in link_pattern.findall(markdown.read_text(encoding="utf-8")):
            target = target.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (markdown.parent / target).resolve()
            if not resolved.exists():
                errors.append(f"broken link in {markdown.relative_to(ROOT)}: {target}")

    for asset in (ROOT / "sections").rglob("*"):
        if asset.is_file() and asset.stat().st_size >= MAX_ASSET_BYTES:
            errors.append(
                f"asset exceeds GitHub 100 MiB limit: {asset.relative_to(ROOT)}"
            )

    if errors:
        raise SystemExit("\n".join(errors))
    print(
        f"validated {len(sections)} sections and {len(markdown_files)} Markdown files"
    )


if __name__ == "__main__":
    main()
