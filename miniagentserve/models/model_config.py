from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import torch

__all__ = ["ModelConfig"]


# Precision names as they appear in a checkpoint's `config.json`.
_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "half": torch.float16,
    "float32": torch.float32,
    "float": torch.float32,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}


def parse_dtype(value: Any) -> torch.dtype:
    """Resolve a `torch.dtype` or a checkpoint dtype name such as `"bfloat16"`."""
    if isinstance(value, torch.dtype):
        return value
    try:
        return _DTYPES[str(value).removeprefix("torch.")]
    except KeyError:
        raise ValueError(f"unsupported dtype {value!r}, expected one of {sorted(_DTYPES)}") from None


@dataclass(frozen=True, kw_only=True)
class ModelConfig:
    # Decoder shape
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    max_position_embeddings: int
    tie_word_embeddings: bool = False

    # Special tokens
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int

    # RoPE
    rope_theta: float | None = None

    # Precision: `dtype` is the weight/compute precision the model runs in;
    # `kv_cache_dtype` defaults to it (see `kv_dtype`).
    dtype: torch.dtype = torch.bfloat16
    kv_cache_dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        # Accept dtype names (e.g. from `config.json`) as well as `torch.dtype`.
        object.__setattr__(self, "dtype", parse_dtype(self.dtype))
        if self.kv_cache_dtype is not None:
            object.__setattr__(self, "kv_cache_dtype", parse_dtype(self.kv_cache_dtype))

    @property
    def kv_dtype(self) -> torch.dtype:
        """Precision of the KV cache, which follows `dtype` unless overridden."""
        return self.kv_cache_dtype or self.dtype

    @property
    def kv_dtype_bytes(self) -> int:
        """Bytes per KV cache element, for sizing the block manager."""
        return self.kv_dtype.itemsize

    @classmethod
    def from_json(cls, path: str) -> ModelConfig:
        with open(path) as f:
            top = json.load(f)
        text = top.get("text_config", top)
        return cls(**cls.base_kwargs(top, text), **cls.extra_kwargs(top, text))

    @classmethod
    def base_kwargs(cls, top: dict[str, Any], text: dict[str, Any]) -> dict[str, Any]:
        rope = text.get("rope_parameters") or text.get("rope_scaling") or {}
        dtype = text.get("dtype") or text.get("torch_dtype") or top.get("dtype") or top.get("torch_dtype")
        return dict(
            vocab_size=text["vocab_size"],
            hidden_size=text["hidden_size"],
            intermediate_size=text["intermediate_size"],
            num_hidden_layers=text["num_hidden_layers"],
            num_attention_heads=text["num_attention_heads"],
            num_key_value_heads=text.get("num_key_value_heads") or text["num_attention_heads"],
            head_dim=text.get("head_dim") or text["hidden_size"] // text["num_attention_heads"],
            rms_norm_eps=text["rms_norm_eps"],
            max_position_embeddings=text["max_position_embeddings"],
            tie_word_embeddings=text.get("tie_word_embeddings", top.get("tie_word_embeddings", False)),
            rope_theta=rope.get("rope_theta", text.get("rope_theta")),
            **({"dtype": parse_dtype(dtype)} if dtype else {}),
        )

    @classmethod
    def extra_kwargs(cls, top: dict[str, Any], text: dict[str, Any]) -> dict[str, Any]:
        return {}
