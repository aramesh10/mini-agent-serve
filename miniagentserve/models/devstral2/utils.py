"""
utils.py
Checkpoint loading helpers for the Devstral 2 model.
"""

from __future__ import annotations

import glob
import json
import os


def _shard_files(path: str) -> list[str]:
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
        return sorted({os.path.join(path, name) for name in weight_map.values()})
    shards = sorted(glob.glob(os.path.join(path, "model*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no safetensors weights found in {path}")
    return shards


def _convert_key(key: str) -> str | None:
    """Map a checkpoint key onto this module tree, or `None` to skip it."""
    if key.startswith("language_model."):
        key = key[len("language_model.") :]
    if key.startswith(("model.vision_tower", "model.multi_modal_projector", "vision_tower", "multi_modal_projector")):
        return None
    return key
