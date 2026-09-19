from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import torch
from torch import nn

__all__ = ["Model"]

_TENSOR_ANNOTATIONS = (torch.Tensor, "torch.Tensor", "Tensor")

class Model(nn.Module, ABC):
    """An `nn.Module` whose subclasses must implement `forward() -> torch.Tensor`."""

    __call__: Callable[..., torch.Tensor]

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        forward = cls.__dict__.get("forward")
        if forward is None or getattr(forward, "__isabstractmethod__", False):
            return
        annotation = getattr(forward, "__annotations__", {}).get("return")
        if annotation not in _TENSOR_ANNOTATIONS:
            raise TypeError(
                f"{cls.__name__}.forward must be annotated to return torch.Tensor, "
                f"got {annotation!r}"
            )
