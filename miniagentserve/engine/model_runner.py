"""
model_runner.py
"""
import torch

from miniagentserve.engine.sequence import Sequence
from miniagentserve.models.devstral2.devstral2 import Devstral2ForCausalLM

class ModelRunner:

    def __init__(self, path: str, num_blocks: int, block_size: int):
        self.model = Devstral2ForCausalLM.from_pretrained(path)
        config = self.model.config
        self.block_size = block_size
        self.num_slots = num_blocks * block_size
        self.device = next(self.model.parameters()).device
        # (num_layers, 1, kv_heads, num_slots, head_dim)
        shape = (config.num_hidden_layers, 1, config.num_key_value_heads, self.num_slots, config.head_dim)
        self.k_cache = torch.zeros(shape, dtype=config.kv_dtype, device=self.device)
        self.v_cache = torch.zeros(shape, dtype=config.kv_dtype, device=self.device)

    def prepare(self, seqs: list[Sequence]):
        bs = self.block_size
        input_ids, cache_lens, kv_write_index = [], [], []
        kv_read_index = torch.full((len(seqs), self.num_slots), -1, dtype=torch.int32)
        for b, seq in enumerate(seqs):
            start = seq.num_cached_tokens
            input_ids.append(seq.token_ids[start:])
            cache_lens.append(start)
            kv_write_index.extend(seq.block_table[p // bs] * bs + p % bs for p in range(start, len(seq)))
            slots = (torch.tensor(seq.block_table)[:, None] * bs + torch.arange(bs)).flatten()
            kv_read_index[b, slots] = torch.arange(len(slots), dtype=torch.int32)
        input_ids       = torch.tensor(     input_ids, dtype=torch.long, device=self.device)
        cache_lens      = torch.tensor(    cache_lens, dtype=torch.long, device=self.device)
        kv_write_index  = torch.tensor(kv_write_index, dtype=torch.long, device=self.device)
        kv_read_index   = kv_read_index.to(self.device)
        return input_ids, cache_lens, kv_write_index, kv_read_index

    def sample(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float()
        greedy = logits.argmax(-1)
        probs = torch.softmax(logits / temperatures.clamp(min=1e-5)[:, None], -1)
        sampled = probs.div_(torch.empty_like(probs).exponential_(1)).argmax(-1)  # Gumbel-max
        return torch.where(temperatures == 0, greedy, sampled)

    @torch.inference_mode()
    def run(self, seqs: list[Sequence]) -> list[int]:
        input_ids, cache_lens, kv_write_index, kv_read_index = self.prepare(seqs)
        logits = self.model(
            input_ids,
            k_cache=self.k_cache,
            v_cache=self.v_cache,
            kv_write_index=kv_write_index,
            kv_read_index=kv_read_index,
            cache_lens=cache_lens,
            logits_to_keep=1,
        )
        temperatures = torch.tensor([seq.temperature for seq in seqs], device=self.device)
        return self.sample(logits[:, -1], temperatures).tolist()
