"""
sequence.py
"""
from dataclasses import dataclass
from typing import Callable


@dataclass
class SamplingParams:
    temperature: float = 0.0
    max_tokens: int = 64

class Sequence:

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams = SamplingParams(), on_finish: Callable[[str], None] | None = None):
        self.token_ids = token_ids
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0          # tokens whose K/V are already in the cache
        self.block_table: list[int] = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.on_finish = on_finish

    def __len__(self):
        return len(self.token_ids)

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]
