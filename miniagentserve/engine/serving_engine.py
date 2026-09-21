"""
serving_engine.py
"""
import itertools
import os
import time
from typing import Callable
from collections import deque

from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream

from miniagentserve.engine.model_runner import ModelRunner
from miniagentserve.engine.block_manager import KVBlockManager
from miniagentserve.engine.sequence import SamplingParams, Sequence
from miniagentserve.models.utils import _resolve_checkpoint_dir

class ServingEngine:

    def __init__(self, path: str, num_blocks: int = 4096, block_size: int = 16, max_num_seqs: int = 64):
        self.model_runner = ModelRunner(path, num_blocks, block_size)
        self.tokenizer = Tokenizer.from_file(os.path.join(_resolve_checkpoint_dir(path), "tokenizer.json"))
        self.max_num_seqs = max_num_seqs
        self.block_manager = KVBlockManager(num_blocks, block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.num_generated_tokens = 0
        self.request_metrics: deque[dict] = deque(maxlen=100)  # most recent finished requests
        self.request_ids = itertools.count()                   # in finish order, so readers can tell which are new
        self.warmup()

    def warmup(self, n: int = 10):
        for i in range(n):  # distinct prompt lengths so the dynamic-shape prefill gets compiled too
            self.add_request("warmup " * (i + 1), SamplingParams(max_tokens=4))
            while not self.has_no_work():
                self.step()
        self.num_generated_tokens = 0
        self.request_metrics.clear()
        self.request_ids = itertools.count()

    def run(self):
        while True:
            if self.has_no_work():
                time.sleep(0.001)
                continue
            for seq in self.step():
                seq.on_token(seq.decoder.step(self.tokenizer, seq.token_ids[-1]) or "", seq.finished)

    def add_request(self, prompt: str, sampling_params: SamplingParams = SamplingParams(), on_token: Callable[[str, bool], None] | None = None):
        bos_token_id = self.model_runner.model.config.bos_token_id
        prompt = [bos_token_id] + self.tokenizer.encode(prompt).ids
        seq = Sequence(prompt, sampling_params, on_token)
        seq.decoder = DecodeStream(skip_special_tokens=True)
        self.waiting.append(seq)
        return seq

    def preempt(self, seq: Sequence):
        self.block_manager.deallocate(seq)
        seq.enqueue_time = time.perf_counter()
        self.waiting.appendleft(seq)

    def schedule(self) -> list[Sequence]:
        if self.waiting and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0]
            if self.block_manager.can_allocate(seq):
                self.waiting.popleft()
                self.block_manager.allocate(seq)
                self.running.append(seq)
                return [seq]

        scheduled = []
        while self.running:
            seq = self.running.popleft()
            while not self.block_manager.can_allocate(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                self.block_manager.allocate(seq)
                scheduled.append(seq)
        self.running.extend(scheduled)
        return scheduled

    def step(self) -> list[Sequence]:
        """Runs one forward step and returns the sequences that advanced in it."""
        seqs = self.schedule()
        token_ids = self.model_runner.run(seqs)
        eos_token_id = self.model_runner.model.config.eos_token_id
        now = time.perf_counter()
        self.num_generated_tokens += len(seqs)
        for seq, token_id in zip(seqs, token_ids):
            seq.num_cached_tokens = len(seq)
            seq.token_ids.append(token_id)
            seq.first_token_time = seq.first_token_time or now
            seq.finished = token_id == eos_token_id or len(seq.completion_token_ids) == seq.max_tokens
            if seq.finished:
                self.running.remove(seq)
                self.block_manager.deallocate(seq)
                self.request_metrics.append(self.request_stats(seq, now))
        return seqs

    def request_stats(self, seq: Sequence, now: float) -> dict:
        n = len(seq.completion_token_ids)
        return {
            "id": next(self.request_ids),
            "ttft_ms": round((seq.first_token_time - seq.arrival_time) * 1e3, 2),
            "tpot_ms": round((now - seq.first_token_time) * 1e3 / (n - 1), 2) if n > 1 else None,
            "e2e_ms": round((now - seq.arrival_time) * 1e3, 2),
            "output_tokens": n,
            "prompt": self.tokenizer.decode(seq.token_ids[:seq.num_prompt_tokens])[:50],
            "response": self.tokenizer.decode(seq.completion_token_ids)[:50],
        }

    def has_no_work(self):
        return not self.waiting and not self.running

