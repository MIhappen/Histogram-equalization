"""Build a Markdown copy that Zhihu can import without broken image URLs."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "README.md"
TARGET = ROOT / "zhihu_import.md"
REPO = "https://github.com/MIhappen/Histogram-equalization/blob/main/"
EXPECTED = {"comparison.png", "shadow_detail.png", "intensity_regions.png"}


def main() -> None:
    article = SOURCE.read_text(encoding="utf-8")
    seen: list[str] = []

    def replace_image(match: re.Match[str]) -> str:
        alt, filename = match.groups()
        seen.append(filename)
        return (
            f"【插图位置：{alt}。请在知乎编辑器中上传 {filename}，"
            "上传后删除这行提示。】"
        )

    article = re.sub(r"!\[([^\]]+)\]\(assets/([^)]+)\)", replace_image, article)
    if set(seen) != EXPECTED or len(seen) != len(EXPECTED):
        raise ValueError(f"Unexpected image references: {seen}")

    for filename in ("enhance.py", "generate_figures.py"):
        article = article.replace(f"]({filename})", f"]({REPO}{filename})")
    if re.search(r"!\[[^\]]*\]\(", article):
        raise ValueError("An image reference was left in the Zhihu copy")
    TARGET.write_text(article, encoding="utf-8")
    print(f"Wrote {TARGET.name} with {len(seen)} image insertion markers")


if __name__ == "__main__":
    main()
