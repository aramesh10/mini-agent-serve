"""
serving_engine.py
"""
import itertools
import os
import threading
import time
from typing import Callable
from collections import deque

import torch
from tokenizers import Tokenizer

from miniagentserve.engine.model_runner import ModelRunner
from miniagentserve.engine.block_manager import KVBlockManager
from miniagentserve.engine.sequence import SamplingParams, Sequence
from miniagentserve.models.utils import _resolve_checkpoint_dir


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
        self.new_request = threading.Event()
        self.num_generated_tokens = 0
        self.request_metrics: deque[dict] = deque(maxlen=100)  
        self.request_ids = itertools.count()
    
    def has_no_work(self):
        return not self.waiting and not self.running

    def run(self):
        """Engine loop: runs until the process exits."""
        while True:
            # Wait for requests
            self.new_request.wait()
            self.new_request.clear()
            if self.has_no_work(): continue
            
            # Run steps, finishing each one while the GPU runs the next
            prev_step = self.launch_step()
            while not self.has_no_work():
                step = self.launch_step()
                if prev_step:
                    self.emit(self.finish_step(*prev_step))
                prev_step = step
            if prev_step:
                self.emit(self.finish_step(*prev_step))

    def add_request(self, prompt: str, sampling_params: SamplingParams = SamplingParams(), on_token: Callable[[str, bool], None] | None = None):
        bos_token_id = self.model_runner.model.config.bos_token_id
        prompt = [bos_token_id] + self.tokenizer.encode(prompt).ids
        seq = Sequence(prompt, sampling_params, on_token)
        self.waiting.append(seq)
        self.new_request.set()
        return seq

    def emit(self, seqs: list[tuple[Sequence, int]]):
        for seq, token_id in seqs:
            seq.emit(self.tokenizer, token_id)
            
    def launch_step(self) -> tuple[torch.Tensor, torch.cuda.Event, list[tuple[Sequence, int, int]]] | None:
        # Get scheduled sequences
        seqs = self.schedule()
        if not seqs: return None

        # Run scheduled sequences
        sampled, done_event = self.model_runner.launch(seqs)

        # Handle sequences while GPU runs
        pending = []
        for seq in seqs:
            seq.num_cached_tokens += seq.num_new_tokens
            if seq.num_cached_tokens < len(seq): # chunked prefill
                continue
            seq.token_ids.append(-1 - seq.row)
            pending.append((seq, len(seq) - 1, seq.row))
            if len(seq.completion_token_ids) == seq.max_tokens:
                self.running.remove(seq)
        return sampled, done_event, pending

    def finish_step(self, sampled: torch.Tensor, done_event: torch.cuda.Event, pending: list[tuple[Sequence, int, int]]) -> list[tuple[Sequence, int]]:
        # wait for tensor to reach CPU
        done_event.synchronize()
        sampled = sampled.tolist()
        now = time.perf_counter()

        # forward pass complete, handle EOS and max token limit
        eos_token_id = self.model_runner.model.config.eos_token_id
        advanced = []
        for seq, i, row in pending:
            # already finished, ignore result
            if seq.finished: continue   
            
            # add sequence to list of sequences that advanced
            sampled_token = sampled[row]
            advanced.append((seq, sampled_token))
        
            # update sequence 
            seq.token_ids[i] = sampled_token
            seq.first_token_time = seq.first_token_time or now
            seq.finished = (sampled_token == eos_token_id  or i + 1 - seq.num_prompt_tokens == seq.max_tokens)
            
            # release if finished
            if seq.finished: self.release_sequence(seq, i, now)
        self.num_generated_tokens += len(advanced)
        return advanced

    def schedule(self) -> list[Sequence]:
        num_decodes = sum(len(seq) - seq.num_cached_tokens == 1 for seq in self.running)
        budget = self.max_num_batched_tokens - self.model_runner.decode_rows(num_decodes) + num_decodes
        running, self.running = self.running, deque()
        while running:   
            seq = running.popleft()     # oldest first
            while not self.block_manager.can_allocate(seq) and running:
                self.preempt(running.pop())
            if self.block_manager.can_allocate(seq):
                budget -= self.admit(seq, budget)
            else:
                self.preempt(seq)
        while (self.waiting 
               and budget 
               and len(self.running) < self.max_num_seqs
               and self.block_manager.can_allocate(self.waiting[0])):
            budget -= self.admit(self.waiting.popleft(), budget)
        return list(self.running)

    def release_sequence(self, seq: Sequence, i: int, now: float):
        del seq.token_ids[i + 1:]                                   # drop placeholder
        for queue in (self.running, self.waiting):                  # remove from running or waiting queue
            if seq in queue:
                queue.remove(seq)
        self.block_manager.deallocate(seq)                          # deallocate KV cache
        self.request_metrics.append(self.request_stats(seq, now))   # Add metrics
        

    def preempt(self, seq: Sequence):
        self.block_manager.deallocate(seq)
        seq.enqueue_time = time.perf_counter()
        self.waiting.appendleft(seq)

    def admit(self, seq: Sequence, budget: int) -> int:
        self.block_manager.allocate(seq)
        self.running.append(seq)
        seq.num_new_tokens = min(len(seq) - seq.num_cached_tokens, budget)
        return seq.num_new_tokens

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
