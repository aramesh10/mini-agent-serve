"""
model_runner.py
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
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

    @classmethod
    def alloc(cls, m: int, num_blocks: int, device: str | torch.device) -> ModelRunnerGraphInputs:
        # `block_tables` is as wide as the cache: no sequence can own more blocks than exist.
        return cls(
            input_ids=torch.zeros((m, 1), dtype=torch.long, device=device),
            block_tables=torch.zeros((m, num_blocks), dtype=torch.int32, device=device),
            cache_lens=torch.zeros(m, dtype=torch.long, device=device),
        )

    def copy(self, src: ModelRunnerGraphInputs) -> None:
        b, t = src.input_ids.shape
        self.input_ids[:b, :t].copy_(src.input_ids)
        self.block_tables[:b, :src.block_tables.shape[1]].copy_(src.block_tables)
        self.cache_lens[:b].copy_(src.cache_lens)

    def kwargs(self) -> dict[str, torch.Tensor]:
        return dict(vars(self))


@dataclass
class ModelRunnerGraphOutputs:
    logits: torch.Tensor    # (batch, num_kept_tokens, vocab_size)
        

class ModelRunner:

    def __init__(self, path: str, num_blocks: int, block_size: int):
        self.model = Devstral2ForCausalLM.from_pretrained(path)
        config = self.model.config
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.device = next(self.model.parameters()).device
        
        # (num_layers, num_blocks, block_size, kv_heads, head_dim)
        self.kv_shape = (config.num_hidden_layers, num_blocks, block_size, config.num_key_value_heads, config.head_dim)
        self.kv_dtype = config.kv_dtype
        self.k_cache = torch.zeros(self.kv_shape, dtype=self.kv_dtype, device=self.device)
        self.v_cache = torch.zeros(self.kv_shape, dtype=self.kv_dtype, device=self.device)
        
        # warmup flashinfer
        warmup_decode  = ModelRunnerGraphInputs.alloc( 1, self.num_blocks, self.device)
        warmup_prefill = ModelRunnerGraphInputs.alloc(16, self.num_blocks, self.device)
        self.model(**warmup_decode.kwargs(), k_cache=self.k_cache, v_cache=self.v_cache, logits_to_keep=1)
        self.model(**warmup_prefill.kwargs(), k_cache=self.k_cache, v_cache=self.v_cache, logits_to_keep=1)
        
        self.compiled_model = torch.compile(self.model, dynamic=True)
        self.compile_graph(GRAPH_BATCH_SIZES)

    @torch.inference_mode()
    def compile_graph(self, fixed_m: list[int]):
        pool = torch.cuda.graph_pool_handle()
        side = torch.cuda.Stream()
        self.graphs: dict[int, torch.cuda.CUDAGraph] = dict()
        self.graph_input_buffers: dict[int, ModelRunnerGraphInputs] = dict()
        self.graph_output_buffers: dict[int, ModelRunnerGraphOutputs] = dict()
        for m in tqdm(fixed_m, desc="Capturing CUDA graphs"):
            inputs = ModelRunnerGraphInputs.alloc(m, self.num_blocks, self.device)
            self.graph_input_buffers[m] = inputs

            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self.forward(inputs)
            torch.cuda.current_stream().wait_stream(side)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                self.graph_output_buffers[m] = self.forward(inputs)
            self.graphs[m] = g

    def forward(self, inputs: ModelRunnerGraphInputs) -> ModelRunnerGraphOutputs:
        model = self.compiled_model if inputs.input_ids.shape[1] == 1 or inputs.input_ids.shape[0] > 4 else self.model
        logits = model(
            **inputs.kwargs(), k_cache=self.k_cache, v_cache=self.v_cache, logits_to_keep=1
        )
        return ModelRunnerGraphOutputs(logits=logits)

    def prepare(self, seqs: list[Sequence]) -> ModelRunnerGraphInputs:
        input_ids, cache_lens = [], []
        max_blocks = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [0] * (max_blocks - len(seq.block_table)) for seq in seqs]
        for seq in seqs:
            start = seq.num_cached_tokens
            input_ids.append(seq.token_ids[start:])
            cache_lens.append(start)
        return ModelRunnerGraphInputs(
            input_ids      = torch.tensor(     input_ids, dtype=torch.long,  device=self.device),
            block_tables   = torch.tensor(  block_tables, dtype=torch.int32, device=self.device),
            cache_lens     = torch.tensor(    cache_lens, dtype=torch.long,  device=self.device),
        )

    def sample(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float()
        greedy = logits.argmax(-1)
        probs = torch.softmax(logits / temperatures.clamp(min=1e-5)[:, None], -1)
        sampled = probs.div_(torch.empty_like(probs).exponential_(1)).argmax(-1)  # Gumbel-max
        return torch.where(temperatures == 0, greedy, sampled)

    @torch.inference_mode()
    def run(self, seqs: list[Sequence]) -> list[int]:
        inputs = self.prepare(seqs)
        m, num_new_tokens = inputs.input_ids.shape
        if num_new_tokens == 1 and m in self.graphs:
            self.graph_input_buffers[m].copy(inputs)
            self.graphs[m].replay()
            outputs = self.graph_output_buffers[m]
        else:
            outputs = self.forward(inputs)
        temperatures = torch.tensor([seq.temperature for seq in seqs], device=self.device)
        return self.sample(outputs.logits[:, -1], temperatures).tolist()
