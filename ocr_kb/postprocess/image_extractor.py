from __future__ import annotations

import base64
import re
from pathlib import Path

from bs4 import BeautifulSoup
from PIL import Image


def save_page_image(
    image: Image.Image,
    source_path: Path,
    page_number: int,
    output_dir: Path,
) -> Path:
    """Save a rendered page image to *output_dir* and return the path.

    Filenames follow the pattern: <stem>_p<NNN>.jpg
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{source_path.stem}_p{page_number:03d}.jpg"
    image.save(out, format="JPEG", quality=85)
    return out


def extract_images_from_html(
    html: str,
    output_dir: Path,
    prefix: str = "img",
) -> list[Path]:
    """Decode any base64 data-URI images embedded in *html* and write them to disk.

    Returns the list of paths written.  Non-data-URI src values are skipped.
    """
    soup = BeautifulSoup(html, "html.parser")
    saved: list[Path] = []

    for i, tag in enumerate(soup.find_all("img")):
        src: str = tag.get("src", "")
        if not src.startswith("data:image/"):
            continue
        try:
            header, data = src.split(",", 1)
            # header looks like "data:image/jpeg;base64"
            ext_match = re.search(r"image/(\w+)", header)
            ext = ext_match.group(1) if ext_match else "jpg"
            image_bytes = base64.b64decode(data)
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"{prefix}_{i:03d}.{ext}"
            path.write_bytes(image_bytes)
            saved.append(path)
        except Exception:
            continue

    return saved
