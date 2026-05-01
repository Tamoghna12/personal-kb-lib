from __future__ import annotations

import re

from bs4 import BeautifulSoup

_BLANK_THRESHOLD = 10  # characters of real text below which a page is blank


def strip_to_fragment(html: str) -> str:
    """Remove <html>/<head>/<body> wrappers if present; return the inner fragment."""
    soup = BeautifulSoup(html, "html.parser")
    body = soup.find("body")
    if body:
        return "".join(str(c) for c in body.children)
    return html


def clean_html(raw: str) -> str:
    """Remove scripts, styles, and metadata; normalise whitespace in text nodes."""
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "meta", "link", "head", "noscript"]):
        tag.decompose()
    # Collapse runs of whitespace inside text nodes (not inside <pre>)
    for text_node in soup.find_all(string=True):
        if text_node.parent and text_node.parent.name == "pre":
            continue
        cleaned = re.sub(r"[ \t]{2,}", " ", str(text_node))
        text_node.replace_with(cleaned)
    return str(soup)


def is_blank(content: str) -> bool:
    """True when the page contains fewer than _BLANK_THRESHOLD real characters.

    Works on both plain text and HTML.
    """
    soup = BeautifulSoup(content, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return len(text) < _BLANK_THRESHOLD


def extract_text(html: str) -> str:
    """Return all visible text from an HTML fragment, whitespace-normalised."""
    soup = BeautifulSoup(html, "html.parser")
    return " ".join(soup.get_text(separator=" ").split())
