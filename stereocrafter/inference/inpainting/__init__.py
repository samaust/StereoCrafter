"""Extensible StereoCrafter inpainting backends."""

from .base import (
    InpaintingBackend,
    available_inpainting_backends,
    create_inpainter,
    register_inpainting_backend,
)

__all__ = [
    "InpaintingBackend",
    "available_inpainting_backends",
    "create_inpainter",
    "register_inpainting_backend",
]
