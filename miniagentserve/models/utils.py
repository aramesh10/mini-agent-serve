"""
utils.py
Model-agnostic checkpoint helpers.
"""

from __future__ import annotations

import os


def _resolve_checkpoint_dir(path: str) -> str:
    if os.path.isdir(path):
        return path
    from huggingface_hub import snapshot_download

    return snapshot_download(
        path,
        allow_patterns=["config.json", "generation_config.json", "tokenizer.json", "model*.safetensors*"],
    )
