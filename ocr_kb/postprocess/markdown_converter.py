from __future__ import annotations

import re

from markdownify import markdownify as _md

from ocr_kb.postprocess.html_parser import clean_html


def html_to_markdown(html: str) -> str:
    """Strip scripts/styles, convert HTML fragment to Markdown, then clean it up."""
    md = _md(clean_html(html), heading_style="ATX", bullets="-")
    return clean_markdown(md)


def plain_to_markdown(text: str) -> str:
    """Minimal cleanup for plain-text OCR output (not HTML)."""
    return clean_markdown(text)


def clean_markdown(text: str) -> str:
    """Collapse excess blank lines and strip leading/trailing whitespace."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def convert_math_blocks(text: str) -> str:
    r"""Convert LaTeX delimiters to standard Markdown math notation.

    \(...\)  → $...$
    \[...\]  → $$...$$
    """
    text = re.sub(r"\\\((.+?)\\\)", r"$\1$", text, flags=re.DOTALL)
    text = re.sub(r"\\\[(.+?)\\\]", r"$$\1$$", text, flags=re.DOTALL)
    return text
