"""
model_runner.py
"""
from __future__ import annotations

import bisect
import math
import warnings
from dataclasses import dataclass

import torch
from flashinfer.sampling import sampling_from_probs, softmax
from tqdm import tqdm

from miniagentserve.engine.sequence import Sequence
from miniagentserve.models.devstral2.devstral2 import Devstral2ForCausalLM

GRAPH_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]

for _msg in (
    "Dynamo detected a call to a `functools.lru_cache`-wrapped function",
    "Dynamo does not know how to trace the builtin",
):
    warnings.filterwarnings("ignore", message=_msg, category=UserWarning)

@dataclass
class ModelRunnerGraphInputs:

    input_ids: torch.Tensor         # (batch, num_new_tokens), (batch, 1) while decoding
    block_tables: torch.Tensor      # (batch, max_blocks)
    cache_lens: torch.Tensor        # tokens per sequence already in the cache

    def kwargs(self) -> dict[str, torch.Tensor]:
        return dict(vars(self))


@dataclass
class ModelRunnerGraphOutputs:
    logits: torch.Tensor    # (batch, num_kept_tokens, vocab_size)
    tokens: torch.Tensor    # (batch,) sampled, so a replay leaves nothing for the CPU to launch
        

class ModelRunner:

    def __init__(self, path: str, num_blocks: int, block_size: int, max_model_len: int = 8192):
        self.model = Devstral2ForCausalLM.from_pretrained(path)
        config = self.model.config
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.device = next(self.model.parameters()).device
        self.host_buffer = torch.empty(0, dtype=torch.uint8, pin_memory=True)   # staging for `prepare`
        self.max_blocks = min(num_blocks, -(-max_model_len // block_size))      # block table width the graphs capture
        self.philox = torch.zeros(2, dtype=torch.int64, device=self.device)     # sampling RNG: seed, offset
        
        self.pad_block_id = num_blocks  # one past what the block manager hands out: a sink for padding rows
        # (num_layers, num_blocks + 1, block_size, kv_heads, head_dim)
        self.kv_shape = (config.num_hidden_layers, num_blocks + 1, block_size, config.num_key_value_heads, config.head_dim)
        self.kv_dtype = config.kv_dtype
        self.k_cache = torch.zeros(self.kv_shape, dtype=self.kv_dtype, device=self.device)
        self.v_cache = torch.zeros(self.kv_shape, dtype=self.kv_dtype, device=self.device)
        
        # warmup flashinfer
        for m in (1, 16):
            inputs, temperatures = self.unpack(self.device_buffer(m), m, 1, self.max_blocks)
            logits = self.model(**inputs.kwargs(), k_cache=self.k_cache, v_cache=self.v_cache, logits_to_keep=1)
            self.sample(logits[:, -1], temperatures.fill_(1))   # allocates the sampler's workspace, before any capture
        
        self.compiled_model = torch.compile(self.model, dynamic=True)
        self.compile_graph(GRAPH_BATCH_SIZES)

    @torch.inference_mode()
    def compile_graph(self, fixed_m: list[int]):
        pool = torch.cuda.graph_pool_handle()
        side = torch.cuda.Stream()
        self.graphs: dict[int, torch.cuda.CUDAGraph] = dict()
        self.graph_buffers: dict[int, torch.Tensor] = dict()
        self.graph_output_buffers: dict[int, ModelRunnerGraphOutputs] = dict()
        for m in tqdm(fixed_m, desc="Capturing CUDA graphs"):
            self.graph_buffers[m] = self.device_buffer(m)
            inputs, temperatures = self.unpack(self.graph_buffers[m], m, 1, self.max_blocks)
            temperatures.fill_(1)   # a captured softmax must not divide by the zeroed buffer

            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self.forward(inputs, temperatures)
            torch.cuda.current_stream().wait_stream(side)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                self.graph_output_buffers[m] = self.forward(inputs, temperatures)
            self.graphs[m] = g

    def forward(self, inputs: ModelRunnerGraphInputs, temperatures: torch.Tensor) -> ModelRunnerGraphOutputs:
        model = self.compiled_model if inputs.input_ids.shape[1] == 1 or inputs.input_ids.shape[0] > 4 else self.model
        logits = model(
            **inputs.kwargs(), k_cache=self.k_cache, v_cache=self.v_cache, logits_to_keep=1
        )
        return ModelRunnerGraphOutputs(logits=logits, tokens=self.sample(logits[:, -1], temperatures))

    @staticmethod
    def size(m: int, t: int, max_blocks: int) -> int:
        return 8 * m * t + 8 * m + 4 * m * max_blocks + 4 * m

    @staticmethod
    def unpack(buffer: torch.Tensor, m: int, t: int, max_blocks: int) -> tuple[ModelRunnerGraphInputs, torch.Tensor]:
        """Carves a byte buffer into the step's fields, widest dtype first so every view stays aligned."""
        views, offset = [], 0
        for dtype, shape in ((torch.long, (m, t)), (torch.long, (m,)), (torch.int32, (m, max_blocks)), (torch.float32, (m,))):
            views.append(buffer[offset:offset + dtype.itemsize * math.prod(shape)].view(dtype).view(shape))
            offset += dtype.itemsize * math.prod(shape)
        input_ids, cache_lens, block_tables, temperatures = views
        return ModelRunnerGraphInputs(input_ids=input_ids, block_tables=block_tables, cache_lens=cache_lens), temperatures

    def device_buffer(self, m: int) -> torch.Tensor:
        return torch.zeros(self.size(m, 1, self.max_blocks), dtype=torch.uint8, device=self.device)

    def prepare(self, seqs: list[Sequence], t: int, max_blocks: int, graph_m: int | None) -> tuple[ModelRunnerGraphInputs, torch.Tensor]:
        """Builds ids, lens, tables, temps from seqs on Host and transfer to Device in one transaction"""
        m, rows = len(seqs), graph_m or len(seqs)
        width = self.max_blocks if graph_m else max_blocks
        size = self.size(rows, t, width)
        if size > self.host_buffer.numel():
            self.host_buffer = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        host = self.host_buffer[:size]
        inputs, temperatures = self.unpack(host, rows, t, width)
        # assigning a list into a numpy view of the pinned buffer skips the intermediate CPU tensor
        ids, lens, tables, temps = (v.numpy() for v in (inputs.input_ids, inputs.cache_lens, inputs.block_tables, temperatures))
        for i, seq in enumerate(seqs):
            ids[i] = seq.token_ids[seq.num_cached_tokens:]
            lens[i] = seq.num_cached_tokens
            tables[i, :len(seq.block_table)] = seq.block_table   # the rest of the row sits past cache_lens and is never read
            temps[i] = max(seq.temperature, 1e-5)   # greedy is a one-hot softmax, no separate argmax
        lens[m:], tables[m:, 0], temps[m:] = 0, self.pad_block_id, 1   # rows padding the batch up to the captured size
        buffer = self.graph_buffers[graph_m] if graph_m else torch.empty(size, dtype=torch.uint8, device=self.device)
        buffer.copy_(host, non_blocking=True)                    # the step's one transfer
        return self.unpack(buffer, rows, t, width)

    def sample(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        """Two fused kernels. Greedy needs no branch of its own: `prepare` clamps its temperature to
        near zero, which makes the softmax one-hot at the argmax."""
        self.philox[1:].add_(32 * logits.shape[0])   # inside the graph, so every replay draws afresh
        return sampling_from_probs(softmax(logits, temperatures), seed=self.philox[:1], offset=self.philox[1:])

    @torch.inference_mode()
    def run(self, seqs: list[Sequence]) -> list[int]:
        m = len(seqs)
        num_new_tokens = len(seqs[0]) - seqs[0].num_cached_tokens
        max_blocks = max(len(seq.block_table) for seq in seqs)
        decoding = num_new_tokens == 1
        i = bisect.bisect_left(GRAPH_BATCH_SIZES, m)  # pad the batch up to the next captured size
        graph_m = GRAPH_BATCH_SIZES[i] if decoding and i < len(GRAPH_BATCH_SIZES) and max_blocks <= self.max_blocks else None

        inputs, temperatures = self.prepare(seqs, num_new_tokens, max_blocks, graph_m)

        if graph_m:
            self.graphs[graph_m].replay()
            tokens = self.graph_output_buffers[graph_m].tokens
        else:
            tokens = self.forward(inputs, temperatures).tokens

        return tokens[:m].tolist()   # .tolist() is where the step blocks on the GPU
