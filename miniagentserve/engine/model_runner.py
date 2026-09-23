"""
model_runner.py
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from flashinfer.sampling import sampling_from_probs, softmax
from tqdm import tqdm

from miniagentserve.engine.sequence import Sequence
from miniagentserve.models.devstral2.devstral2 import Devstral2ForCausalLM

# Graphs are keyed by (decode rows, total tokens): each step is padded up to the nearest captured pair.
GRAPH_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]    # decode rows, one token each
TOKEN_STEP = 64                                 # steps with prompts: total tokens at each multiple, so GEMM M stays tile-aligned
PREFILL_ROWS = 17                               # prompt rows: up to 16 prompts per step, plus a spare for padding

torch._dynamo.config.recompile_limit = 64       # one compiled variant per decode/prompt mix the graphs capture

for _msg in (
    "Dynamo detected a call to a `functools.lru_cache`-wrapped function",
    "Dynamo does not know how to trace the builtin",
):
    warnings.filterwarnings("ignore", message=_msg, category=UserWarning)

@dataclass
class ModelRunnerGraphInputs:

    input_ids: torch.Tensor         # (num_tokens,) every sequence's new tokens, back to back
    positions: torch.Tensor         # (num_tokens,)
    batch_indices: torch.Tensor     # (num_tokens,) the sequence each token belongs to
    seq_lens: torch.Tensor          # (batch,) cached plus new tokens; decode rows first
    cu_seqlens_q: torch.Tensor      # (num_prefills + 1,) where each prompt row's tokens start, after the decode tokens
    block_tables: torch.Tensor      # (batch, max_blocks)
    temperatures: torch.Tensor      # (batch,) for the sampler, not the model

    @staticmethod
    def layout(num_tokens: int, batch: int, num_prefills: int, max_blocks: int) -> list[tuple[torch.dtype, tuple[int, ...]]]:
        """Each field's dtype and shape, in field order: widest dtype first, so views stay aligned."""
        return [(torch.long, (num_tokens,)), (torch.int32, (num_tokens,)), (torch.int32, (num_tokens,)),
                (torch.int32, (batch,)), (torch.int32, (num_prefills + 1,)), (torch.int32, (batch, max_blocks)),
                (torch.float32, (batch,))]

    @staticmethod
    def nbytes(*dims: int) -> int:
        return sum(dtype.itemsize * math.prod(shape) for dtype, shape in ModelRunnerGraphInputs.layout(*dims))

    @staticmethod
    def unpack(buffer: torch.Tensor, *dims: int) -> ModelRunnerGraphInputs:
        """Carves a byte buffer into the fields, so one copy moves them all."""
        views, offset = [], 0
        for dtype, shape in ModelRunnerGraphInputs.layout(*dims):
            views.append(buffer[offset:offset + dtype.itemsize * math.prod(shape)].view(dtype).view(shape))
            offset += dtype.itemsize * math.prod(shape)
        return ModelRunnerGraphInputs(*views)

    def model_kwargs(self) -> dict[str, torch.Tensor]:
        return {name: value for name, value in vars(self).items() if name != "temperatures"}


class ModelRunner:

    def __init__(self, 
                 path: str, 
                 num_blocks: int, 
                 block_size: int, 
                 max_num_seqs: int = 64,
                 max_num_batched_tokens: int = 512, 
                 max_model_len: int = 8192):
        self.model = Devstral2ForCausalLM.from_pretrained(path)
        config = self.model.config
        self.device = next(self.model.parameters()).device
        self.host_buffer = torch.empty(0, dtype=torch.uint8, pin_memory=True)   # staging for `prepare`
        self.max_blocks = min(num_blocks, -(-max_model_len // block_size))      # block table width the graphs capture
        self.philox = torch.zeros(2, dtype=torch.int64, device=self.device)     # sampling RNG: seed, offset
        
        self.pad_block_id = num_blocks  # one past what the block manager hands out: a sink for padding rows
        kv_shape = (config.num_hidden_layers, num_blocks + 1, block_size, config.num_key_value_heads, config.head_dim)
        self.k_cache = torch.zeros(kv_shape, dtype=config.kv_dtype, device=self.device)
        self.v_cache = torch.zeros(kv_shape, dtype=config.kv_dtype, device=self.device)
        
        self.graph_shapes = [(d, d) for d in GRAPH_BATCH_SIZES] + [   # decode only, then with prompts
            (d, t) for d in [0, *GRAPH_BATCH_SIZES] for t in range(TOKEN_STEP, max_num_batched_tokens + TOKEN_STEP, TOKEN_STEP) if t > d
        ]
        self.graph_buffers = {
            shape: torch.zeros(ModelRunnerGraphInputs.nbytes(*self.dims(*shape), self.max_blocks), dtype=torch.uint8, device=self.device)
            for shape in self.graph_shapes
        }
        # warmup flashinfer
        for shape in ((1, 1), (16, 16), (1, 64)):
            self.forward(self.prepare([], [], shape), self.model)
        
        self.compiled_model = torch.compile(self.model, dynamic=True)
        self.compile_graph(self.graph_shapes)

    @torch.inference_mode()
    def compile_graph(self, shapes: list[tuple[int, int]]):
        pool = torch.cuda.graph_pool_handle()
        side = torch.cuda.Stream()
        self.graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = dict()
        self.graph_tokens: dict[tuple[int, int], torch.Tensor] = dict()
        for shape in tqdm(shapes, desc="Capturing CUDA graphs"):
            inputs = self.prepare([], [], shape)   # all padding rows: valid, and harmless to run

            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self.forward(inputs, self.compiled_model)
            torch.cuda.current_stream().wait_stream(side)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                self.graph_tokens[shape] = self.forward(inputs, self.compiled_model)
            self.graphs[shape] = g

    def forward(self, inputs: ModelRunnerGraphInputs, model: torch.nn.Module) -> torch.Tensor:
        logits = model(**inputs.model_kwargs(), k_cache=self.k_cache, v_cache=self.v_cache)
        return self.sample(logits, inputs.temperatures)

    @staticmethod
    def dims(decode_rows: int, num_tokens: int) -> tuple[int, int, int]:
        prefill_rows = PREFILL_ROWS if num_tokens > decode_rows else 0
        return num_tokens, decode_rows + prefill_rows, prefill_rows

    @staticmethod
    def decode_rows(num_decodes: int) -> int | None:
        return next((b for b in [0, *GRAPH_BATCH_SIZES] if b >= num_decodes), None)

    def graph_shape(self, decodes: list[Sequence], prefills: list[Sequence]) -> tuple[int, int] | None:
        # eager mode if any sequence is longer than graphs are built for, or too many prompts leave no spare row
        if max(len(seq.block_table) for seq in decodes + prefills) > self.max_blocks or len(prefills) >= PREFILL_ROWS:
            return None
        decode_rows = self.decode_rows(len(decodes))
        if decode_rows is None:
            return None
        num_tokens = decode_rows + sum(seq.num_new_tokens for seq in prefills)
        if prefills:
            num_tokens = -(-num_tokens // TOKEN_STEP) * TOKEN_STEP
        shape = (decode_rows, num_tokens)
        return shape if shape in self.graphs else None

    def prepare(self, decodes: list[Sequence], prefills: list[Sequence], graph_shape: tuple[int, int] | None) -> ModelRunnerGraphInputs:
        """Rows are decodes, their padding, prompts, then theirs; tokens follow the same order."""
        nd, npf = len(decodes), len(prefills)
        if graph_shape is None:
            decode_rows, prefill_rows = nd, npf
            num_tokens = nd + sum(seq.num_new_tokens for seq in prefills)
            width = max(len(seq.block_table) for seq in decodes + prefills)
        else:
            decode_rows = graph_shape[0]
            num_tokens, _, prefill_rows = self.dims(*graph_shape)
            width = self.max_blocks
        rows = decode_rows + prefill_rows

        dims = (num_tokens, rows, prefill_rows, width)
        size = ModelRunnerGraphInputs.nbytes(*dims)
        if size > self.host_buffer.numel():
            self.host_buffer = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        host = self.host_buffer[:size]
        # assigning a list into a numpy view of the pinned buffer skips the intermediate CPU tensor
        ids, pos, idx, lens, cu, tables, temps = (v.numpy() for v in vars(ModelRunnerGraphInputs.unpack(host, *dims)).values())

        def fill(row: int, t: int, seq: Sequence):
            start, n = seq.num_cached_tokens, seq.num_new_tokens
            ids[t:t + n] = seq.token_ids[start:start + n]
            pos[t:t + n] = range(start, start + n)
            idx[t:t + n] = row
            lens[row] = start + n
            tables[row, :len(seq.block_table)] = seq.block_table    # the rest of the row sits past seq_lens and is never read
            temps[row] = max(seq.temperature, 1e-5)                 # greedy is a one-hot softmax, no separate argmax

        for i, seq in enumerate(decodes):   # decode row i owns token i
            fill(i, i, seq)
        # decode padding: one token each, at position 0 of the sink block
        ids[nd:decode_rows], pos[nd:decode_rows], idx[nd:decode_rows] = 0, 0, np.arange(nd, decode_rows)
        lens[nd:decode_rows], tables[nd:decode_rows, 0], temps[nd:decode_rows] = 1, self.pad_block_id, 1

        t = 0   # prompt tokens so far, counted from the first one
        for j, seq in enumerate(prefills):
            cu[j] = t
            fill(decode_rows + j, decode_rows + t, seq)
            t += seq.num_new_tokens
        if prefill_rows > npf:   # prompt padding: the first spare row takes every leftover token, the rest stay empty
            first, start = decode_rows + npf, decode_rows + t
            ids[start:], pos[start:], idx[start:] = 0, np.arange(num_tokens - start), first
            lens[first:], lens[first] = 1, max(num_tokens - start, 1)
            cu[npf], cu[npf + 1:] = t, num_tokens - decode_rows
            tables[first], tables[first + 1:, 0], temps[first:] = self.pad_block_id, self.pad_block_id, 1   # all in the sink block
        cu[prefill_rows] = num_tokens - decode_rows

        buffer = self.graph_buffers[graph_shape] if graph_shape else torch.empty(size, dtype=torch.uint8, device=self.device)
        buffer.copy_(host, non_blocking=True)                    # the step's one transfer
        return ModelRunnerGraphInputs.unpack(buffer, *dims)

    def sample(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        self.philox[1:].add_(32 * logits.shape[0])   # inside the graph, so every replay draws afresh
        return sampling_from_probs(softmax(logits, temperatures), seed=self.philox[:1], offset=self.philox[1:])

    @torch.inference_mode()
    def run(self, seqs: list[Sequence]) -> list[int]:
        decodes = [seq for seq in seqs if seq.num_new_tokens == 1]
        prefills = [seq for seq in seqs if seq.num_new_tokens > 1]
        shape = self.graph_shape(decodes, prefills)
        inputs = self.prepare(decodes, prefills, shape)
        if shape:
            self.graphs[shape].replay()
            tokens = self.graph_tokens[shape]
        else:
            tokens = self.forward(inputs, self.model)
        sampled = tokens.tolist()   # .tolist() is where the step blocks on the GPU
        decode_rows = shape[0] if shape else len(decodes)
        row = {id(seq): i for i, seq in enumerate(decodes)} | {id(seq): decode_rows + j for j, seq in enumerate(prefills)}
        return [sampled[row[id(seq)]] for seq in seqs]
