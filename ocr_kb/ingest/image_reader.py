from pathlib import Path

from PIL import Image


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def upscale_if_needed(image: Image.Image, min_dim: int) -> Image.Image:
    """Scale up so the shorter side reaches min_dim; no-op if already large enough."""
    w, h = image.size
    short = min(w, h)
    if short >= min_dim:
        return image
    scale = min_dim / short
    return image.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
