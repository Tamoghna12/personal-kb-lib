from dataclasses import dataclass

from PIL import Image


@dataclass
class OcrRequest:
    """Input to the OCR client — a PIL image and the prompt to send alongside it."""
    image: Image.Image
    prompt: str


@dataclass
class OcrResponse:
    """Raw output from the OCR client."""
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
