from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from miniagentserve.models.model_config import ModelConfig

__all__ = ["Devstral2Config"]


MISTRAL_TOKEN_IDS = {"bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 11}


@dataclass(frozen=True, kw_only=True)
class Devstral2Config(ModelConfig):
    # YaRN rope
    rope_factor: float
    rope_original_max_position_embeddings: int
    rope_beta_fast: float
    rope_beta_slow: float
    rope_mscale: float
    rope_mscale_all_dim: float
    rope_truncate: bool

    llama_4_scaling_beta: float

    @classmethod
    def base_kwargs(cls, top: dict[str, Any], text: dict[str, Any]) -> dict[str, Any]:
        return {**super().base_kwargs(top, text), **MISTRAL_TOKEN_IDS}

    @classmethod
    def extra_kwargs(cls, top: dict[str, Any], text: dict[str, Any]) -> dict[str, Any]:
        rope = text.get("rope_parameters") or text.get("rope_scaling") or {}
        return dict(
            rope_factor=rope["factor"],
            rope_original_max_position_embeddings=rope["original_max_position_embeddings"],
            rope_beta_fast=rope.get("beta_fast", 32.0),
            rope_beta_slow=rope.get("beta_slow", 1.0),
            rope_mscale=rope.get("mscale", 0.0),
            rope_mscale_all_dim=rope.get("mscale_all_dim", 0.0),
            rope_truncate=rope.get("truncate", True),
            llama_4_scaling_beta=rope.get("llama_4_scaling_beta", 0.0),
        )
