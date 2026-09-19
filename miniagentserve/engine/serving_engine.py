"""
serving_engine.py
"""
import os
import time
from typing import Callable
from collections import deque

from tokenizers import Tokenizer

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

    def run(self):
        while True:
            if self.has_no_work():
                time.sleep(0.001)
                continue
            next_tokens = self.step()
            for seq in next_tokens:
                seq.on_finish(self.tokenizer.decode(seq.completion_token_ids))

    def add_request(self, prompt: str, sampling_params: SamplingParams = SamplingParams(), on_finish: Callable[[str], None] | None = None):
        bos_token_id = self.model_runner.model.config.bos_token_id
        prompt = [bos_token_id] + self.tokenizer.encode(prompt).ids
        seq = Sequence(prompt, sampling_params, on_finish)
        self.waiting.append(seq)
        return seq

    def preempt(self, seq: Sequence):
        self.block_manager.deallocate(seq)
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
        """Runs one forward step and returns the sequences that finished in it."""
        seqs = self.schedule()
        token_ids = self.model_runner.run(seqs)
        eos_token_id = self.model_runner.model.config.eos_token_id
        finished = []
        for seq, token_id in zip(seqs, token_ids):
            seq.num_cached_tokens = len(seq)
            seq.token_ids.append(token_id)
            if token_id == eos_token_id or len(seq.completion_token_ids) == seq.max_tokens:
                self.running.remove(seq)
                self.block_manager.deallocate(seq)
                finished.append(seq)
        return finished

    def has_no_work(self):
        return not self.waiting and not self.running

