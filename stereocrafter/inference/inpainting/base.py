from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class InpaintingBackend(Protocol):
    """Common contract implemented by StereoCrafter inpainting backends."""

    name: str

    def inpaint(
        self,
        frames_warped: torch.Tensor,
        frames_mask: torch.Tensor,
        **options: Any,
    ) -> torch.Tensor:
        """Return RGB right-eye frames as float tensors in [T, H, W, C] layout."""


BackendFactory = Callable[..., InpaintingBackend]

_BUILTIN_BACKENDS = {
    "wan_vace": (
        "stereocrafter.inference.inpainting.wan_vace",
        "WanVaceInpainter",
    ),
    "svd": (
        "stereocrafter.inference.inpainting.svd",
        "SvdInpainter",
    ),
}
_REGISTERED_BACKENDS: dict[str, BackendFactory] = {}


def _validate_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError("Backend name must be a non-empty string")
    if not name.replace("_", "").isalnum() or name.lower() != name:
        raise ValueError(
            "Backend names must contain only lowercase letters, numbers, and underscores"
        )
    return name


def available_inpainting_backends() -> tuple[str, ...]:
    """Return built-in and application-registered backend names."""

    return tuple(dict.fromkeys((*_BUILTIN_BACKENDS, *_REGISTERED_BACKENDS)))


def register_inpainting_backend(
    name: str,
    factory: BackendFactory,
    *,
    overwrite: bool = False,
) -> None:
    """Register an application-provided backend factory."""

    name = _validate_name(name)
    if not callable(factory):
        raise TypeError("Backend factory must be callable")
    if not overwrite and name in available_inpainting_backends():
        raise ValueError(f"Inpainting backend {name!r} is already registered")
    _REGISTERED_BACKENDS[name] = factory


def create_inpainter(
    backend: str = "wan_vace",
    **model_options: Any,
) -> InpaintingBackend:
    """Create an inpainting backend without importing unused model stacks."""

    backend = _validate_name(backend)
    factory = _REGISTERED_BACKENDS.get(backend)
    if factory is None and backend in _BUILTIN_BACKENDS:
        module_name, class_name = _BUILTIN_BACKENDS[backend]
        factory = getattr(import_module(module_name), class_name)
    if factory is None:
        choices = ", ".join(available_inpainting_backends())
        raise ValueError(
            f"Unknown inpainting backend {backend!r}; choose one of: {choices}"
        )

    inpainter = factory(**model_options)
    if not isinstance(inpainter, InpaintingBackend):
        raise TypeError(
            f"Backend factory {backend!r} returned an object that does not implement "
            "InpaintingBackend"
        )
    return inpainter
