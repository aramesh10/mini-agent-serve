"""
serving_engine.py
"""
import itertools
import logging
import os
import threading
import time
from typing import Callable
from collections import deque

from tokenizers import Tokenizer

from miniagentserve.engine.model_runner import ModelRunner
from miniagentserve.engine.block_manager import KVBlockManager
from miniagentserve.engine.sequence import SamplingParams, Sequence
from miniagentserve.models.utils import _resolve_checkpoint_dir

logger = logging.getLogger(__name__)

class ServingEngine:

    def __init__(self, path: str, 
                 num_blocks: int = 4096, 
                 block_size: int = 16, 
                 max_num_seqs: int = 64,
                 max_num_batched_tokens: int = 512):
        assert max_num_batched_tokens >= 2 * max_num_seqs, "every running sequence must fit its decode token, plus graph padding"
        self.model_runner = ModelRunner(path, num_blocks, block_size, max_num_seqs, max_num_batched_tokens)
        self.tokenizer = Tokenizer.from_file(os.path.join(_resolve_checkpoint_dir(path), "tokenizer.json"))
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens   # per step: decodes plus prompt chunks
        self.block_manager = KVBlockManager(num_blocks, block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.new_request = threading.Event()                   # wakes an idle `run`
        self.num_generated_tokens = 0
        self.request_metrics: deque[dict] = deque(maxlen=100)  # most recent finished requests
        self.request_ids = itertools.count()                   # in finish order, so readers can tell which are new
        self.warmup()

    def warmup(self, n: int = 10):
        for i in range(n):  # distinct prompt lengths so the dynamic-shape prefill gets compiled too
            self.add_request("warmup " * (i + 1), SamplingParams(max_tokens=4))
            while not self.has_no_work():
                self.step()
        for i in range(n):  # together, and longer than a step's budget: chunked prompts mixed with decodes
            self.add_request("warmup " * (i * self.max_num_batched_tokens // 4 + 1), SamplingParams(max_tokens=4))
        while not self.has_no_work():
            self.step()
        self.num_generated_tokens = 0
        self.request_metrics.clear()
        self.request_ids = itertools.count()

    def run(self):
        """Engine loop: runs until the process exits, and outlives any one request's failures."""
        while True:
            if self.has_no_work():
                self.new_request.wait()
                self.new_request.clear()        # safe: the loop re-checks for work before waiting again
                continue
            for seq in self.step():
                try:
                    seq.emit(self.tokenizer)
                except Exception:      # a disconnected client must not take the engine down with it
                    logger.exception("on_token callback failed; dropping the stream for this sequence")
                    seq.on_token = None

    def add_request(self, prompt: str, sampling_params: SamplingParams = SamplingParams(), on_token: Callable[[str, bool], None] | None = None):
        bos_token_id = self.model_runner.model.config.bos_token_id
        prompt = [bos_token_id] + self.tokenizer.encode(prompt).ids
        seq = Sequence(prompt, sampling_params, on_token)
        self.waiting.append(seq)
        self.new_request.set()
        return seq

    def preempt(self, seq: Sequence):
        self.block_manager.deallocate(seq)
        seq.enqueue_time = time.perf_counter()
        self.waiting.appendleft(seq)

    def schedule(self) -> list[Sequence]:
        # graphs pad decodes up to a captured row count: reserve that padding, so a full step still fits the budget
        num_decodes = sum(len(seq) - seq.num_cached_tokens == 1 for seq in self.running)
        budget = self.max_num_batched_tokens - (self.model_runner.decode_rows(num_decodes) or num_decodes) + num_decodes
        running, self.running = self.running, deque()
        while running:   
            seq = running.popleft()     # oldest first
            while not self.block_manager.can_allocate(seq) and running:
                self.preempt(running.pop())
            if self.block_manager.can_allocate(seq):
                budget -= self.admit(seq, budget)
            else:
                self.preempt(seq)
        while (self.waiting and budget 
               and len(self.running) < self.max_num_seqs
               and self.block_manager.can_allocate(self.waiting[0])):
            budget -= self.admit(self.waiting.popleft(), budget)
        return list(self.running)

    def admit(self, seq: Sequence, budget: int) -> int:
        self.block_manager.allocate(seq)
        seq.num_new_tokens = min(len(seq) - seq.num_cached_tokens, budget)
        self.running.append(seq)
        return seq.num_new_tokens

    def step(self) -> list[Sequence]:
        """Runs one forward step and returns the sequences that advanced in it."""
        seqs = self.schedule()
        if not seqs:
            return []
        token_ids = self.model_runner.run(seqs)
        now = time.perf_counter()
        
        eos_token_id = self.model_runner.model.config.eos_token_id
        advanced = []
        for seq, token_id in zip(seqs, token_ids):
            seq.num_cached_tokens += seq.num_new_tokens
            if seq.num_cached_tokens < len(seq):   # a prompt chunk short of the end: its sample is discarded
                continue
            advanced.append(seq)
            seq.token_ids.append(token_id)
            seq.first_token_time = seq.first_token_time or now
            seq.finished = token_id == eos_token_id or len(seq.completion_token_ids) == seq.max_tokens
            if seq.finished:
                self.running.remove(seq)
                self.block_manager.deallocate(seq)
                self.request_metrics.append(self.request_stats(seq, now))
        self.num_generated_tokens += len(advanced)
        return advanced

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

