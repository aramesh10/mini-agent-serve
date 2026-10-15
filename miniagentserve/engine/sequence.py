"""
sequence.py
"""
import time
from dataclasses import dataclass
from typing import Callable

from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream


@dataclass
class SamplingParams:
    temperature: float = 0.0
    max_tokens: int = 64
    ignore_eos: bool = False
    skip_special_tokens: bool = True   # False streams control tokens too, e.g. a model's tool-call markers

class Sequence:

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams = SamplingParams(), on_token: Callable[[str, bool], None] | None = None):
        self.token_ids = token_ids
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0          # tokens whose K/V are already in the cache
        self.num_new_tokens = 0             # tokens scheduled into the current step
        self.row = 0                        # its row in the current step's inputs and samples, set by `prepare`
        self.block_table: list[int] = []
        self.block_hashes: list[int] = []   # chained hashes of the leading full blocks registered in the prefix cache
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.on_token = on_token            # called each step with (new text, finished)
        self.decoder = DecodeStream(skip_special_tokens=sampling_params.skip_special_tokens)
        self.finished = False
        self.arrival_time = self.enqueue_time = time.perf_counter()
        self.first_token_time: float | None = None

    def __len__(self):
        return len(self.token_ids)

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    def emit(self, tokenizer: Tokenizer, token_id: int):
        if self.on_token is not None:
            self.on_token(self.decoder.step(tokenizer, token_id) or "", self.finished)
