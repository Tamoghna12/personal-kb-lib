from __future__ import annotations

from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag

_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})


@dataclass
class LayoutBlock:
    type: str        # heading | paragraph | list | table | figure | code | unknown
    content: str
    level: int | None  # heading level 1-6; None for non-headings
    index: int         # 0-based document order


def _classify(tag: Tag, index: int) -> LayoutBlock | None:
    name = tag.name.lower() if tag.name else ""

    if name in _HEADING_TAGS:
        return LayoutBlock(
            type="heading",
            content=tag.get_text(strip=True),
            level=int(name[1]),
            index=index,
        )
    if name == "p":
        text = tag.get_text(strip=True)
        return LayoutBlock(type="paragraph", content=text, level=None, index=index) if text else None
    if name in ("ul", "ol"):
        items = [li.get_text(strip=True) for li in tag.find_all("li") if li.get_text(strip=True)]
        content = "\n".join(f"- {item}" for item in items)
        return LayoutBlock(type="list", content=content, level=None, index=index)
    if name == "table":
        return LayoutBlock(type="table", content=str(tag), level=None, index=index)
    if name in ("figure", "img"):
        return LayoutBlock(type="figure", content=str(tag), level=None, index=index)
    if name in ("pre", "code"):
        return LayoutBlock(type="code", content=tag.get_text(), level=None, index=index)
    # Generic block (div, blockquote, section, …)
    text = tag.get_text(strip=True)
    return LayoutBlock(type="unknown", content=text, level=None, index=index) if text else None


def parse_layout(html: str) -> list[LayoutBlock]:
    """Extract an ordered list of semantic blocks from an HTML fragment.

    Only processes direct children of the document root (or <body> if present)
    so that nested tags are not double-counted.
    """
    if not html.strip():
        return []

    soup = BeautifulSoup(html, "html.parser")
    root = soup.find("body") or soup

    blocks: list[LayoutBlock] = []
    for i, child in enumerate(root.children):
        if not isinstance(child, Tag):
            continue
        block = _classify(child, i)
        if block is not None:
            blocks.append(block)

    return blocks
