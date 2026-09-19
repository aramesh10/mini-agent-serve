"""
devstral2.py
Devstral 2 Model implementation

Paper: https://arxiv.org/pdf/2601.08584
HuggingFace Transformer: https://github.com/huggingface/transformers/blob/main/src/transformers/models/ministral3/modeling_ministral3.py 
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
from safetensors import safe_open

from miniagentserve.models.devstral2.configs import Devstral2Config
from miniagentserve.models.devstral2.utils import _convert_key, _shard_files
from miniagentserve.models.model import Model
from miniagentserve.models.utils import _resolve_checkpoint_dir

__all__ = ["Devstral2Model", "Devstral2ForCausalLM"]

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary position embeddings to `(batch, heads, seq, head_dim)` tensors."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class Devstral2RotaryEmbedding(nn.Module):
    """YaRN rotary embeddings."""

    def __init__(self, config: Devstral2Config):
        super().__init__()
        dim = config.head_dim
        base = config.rope_theta
        factor = config.rope_factor
        original_max = config.rope_original_max_position_embeddings

        def get_mscale(scale: float, mscale: float = 1.0) -> float:
            if scale <= 1:
                return 1.0
            return 0.1 * mscale * math.log(scale) + 1.0

        def find_correction_dim(num_rotations: float) -> float:
            """Inverse dimension formula: which dim performs `num_rotations` rotations."""
            return (dim * math.log(original_max / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

        low = find_correction_dim(config.rope_beta_fast)
        high = find_correction_dim(config.rope_beta_slow)
        if config.rope_truncate:
            low, high = math.floor(low), math.ceil(high)
        low, high = max(low, 0), min(high, dim - 1)
        if low == high:
            high += 0.001  # prevent singularity

        pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float, device="cpu") / dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (factor * pos_freqs)

        ramp = torch.clamp(
            (torch.arange(dim // 2, dtype=torch.float, device="cpu") - low) / (high - low), 0, 1
        )
        extrapolation_factor = 1 - ramp
        inv_freq = (
            inv_freq_interpolation * (1 - extrapolation_factor)
            + inv_freq_extrapolation * extrapolation_factor
        )

        if config.rope_mscale and config.rope_mscale_all_dim:
            self.attention_scaling = get_mscale(factor, config.rope_mscale) / get_mscale(
                factor, config.rope_mscale_all_dim
            )
        else:
            self.attention_scaling = get_mscale(factor)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Computed in fp32: cos/sin of huge positions are not representable in bf16.
        freqs = position_ids[..., None].float() * self.inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(x.dtype), sin.to(x.dtype)


@dataclass
class AttentionMetadata:
    """Per-step inputs shared by every attention layer."""

    position_ids: torch.Tensor
    position_embeddings: tuple[torch.Tensor, torch.Tensor]
    kv_write_index: torch.Tensor
    block_mask: BlockMask


class Devstral2Attention(nn.Module):
    """Grouped-query attention with rope and llama-4 style query scaling."""

    def __init__(self, config: Devstral2Config):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.llama_4_scaling_beta = config.llama_4_scaling_beta
        self.original_max_position_embeddings = config.rope_original_max_position_embeddings

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, attn: AttentionMetadata
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        n_kv = self.num_key_value_heads

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = attn.position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Scatter the new K/V into their physical slots. 
        # `(batch, heads, q_len, dim)` -> `(heads, batch * q_len, dim)`
        k_cache[0].index_copy_(1, attn.kv_write_index, key_states.permute(1, 0, 2, 3).reshape(n_kv, -1, self.head_dim))
        v_cache[0].index_copy_(1, attn.kv_write_index, value_states.permute(1, 0, 2, 3).reshape(n_kv, -1, self.head_dim))

        # Attention temperature grows with the absolute position (llama-4 scaling).
        if self.llama_4_scaling_beta:
            attn_scale = 1 + self.llama_4_scaling_beta * torch.log1p(
                torch.floor(attn.position_ids.float() / self.original_max_position_embeddings)
            )
            query_states = query_states * attn_scale[:, None, :, None].to(query_states.dtype)

        attn_output = flex_attention(
            query_states,
            k_cache,
            v_cache,
            block_mask=attn.block_mask,
            scale=self.scaling,
            enable_gqa=True,
        )

        attn_output = attn_output.transpose(1, 2).flatten(2)
        return self.o_proj(attn_output)


class Devstral2MLP(nn.Module):
    def __init__(self, config: Devstral2Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Devstral2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Devstral2DecoderLayer(nn.Module):
    def __init__(self, config: Devstral2Config):
        super().__init__()
        self.self_attn = Devstral2Attention(config)
        self.mlp = Devstral2MLP(config)
        self.input_layernorm = Devstral2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Devstral2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, attn: AttentionMetadata
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, k_cache, v_cache, attn)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Devstral2Model(nn.Module):
    def __init__(self, config: Devstral2Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Devstral2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = Devstral2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Devstral2RotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        kv_write_index: torch.Tensor,
        kv_read_index: torch.Tensor,
        cache_lens: torch.Tensor,
    ) -> torch.Tensor:
        """`k_cache`/`v_cache` are the paged cache, `(num_layers, 1, kv_heads, num_slots,
        head_dim)`, written in place.

        `kv_write_index` is the physical slot per new token in row-major `(batch, q_len)` order
        and `kv_read_index` is each slot's logical position per sequence (-1 = unused); both come
        from `KVBlockManager` via `ModelRunner`. `cache_lens` is the number of tokens
        already cached per sequence.
        """
        q_pos = cache_lens[:, None] + torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden_states = self.embed_tokens(input_ids)
        attn = AttentionMetadata(
            position_ids=q_pos,
            position_embeddings=self.rotary_emb(hidden_states, q_pos),
            kv_write_index=kv_write_index,
            block_mask=self.block_mask(kv_read_index, q_pos),
        )
        for layer, k, v in zip(self.layers, k_cache, v_cache):
            hidden_states = layer(hidden_states, k, v, attn)
        return self.norm(hidden_states)

    @staticmethod
    def block_mask(kv_read_index: torch.Tensor, q_pos: torch.Tensor) -> BlockMask:
        """Paged causal mask: query `(b, q)` sees slots whose logical position `p` in
        sequence `b` satisfies `0 <= p <= q_pos[b, q]`."""

        def mask_mod(b, h, q_idx, kv_idx):
            p = kv_read_index[b, kv_idx]
            return (p >= 0) & (p <= q_pos[b, q_idx])

        return create_block_mask(
            mask_mod, B=q_pos.shape[0], H=None, Q_LEN=q_pos.shape[1], KV_LEN=kv_read_index.shape[1], device=q_pos.device
        )


class Devstral2ForCausalLM(Model):
    def __init__(self, config: Devstral2Config):
        super().__init__()
        self.config = config
        self.model = Devstral2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, *, logits_to_keep: int = 0, **kwargs: torch.Tensor) -> torch.Tensor:
        """Returns logits. `logits_to_keep=n` computes only the last `n` positions.

        `kwargs` go to `Devstral2Model.forward`, which documents the cache layout and the
        caller's obligations.
        """
        hidden_states = self.model(input_ids, **kwargs)
        if logits_to_keep:
            hidden_states = hidden_states[:, -logits_to_keep:]
        return self.lm_head(hidden_states)

    # ---- checkpoint loading -------------------------------------------------

    @classmethod
    def from_pretrained(
        cls, path: str, dtype: torch.dtype | None = None, device: str | torch.device = "cuda"
    ) -> Devstral2ForCausalLM:
        """Load a released checkpoint (local directory or HF repo id).

        The released checkpoints wrap the decoder in a multimodal model, so weights are
        prefixed `language_model.` and the vision tower is skipped. Linear weights are
        stored as fp8 (e4m3) with a scalar `weight_scale_inv`; they are dequantized to
        `dtype`, which defaults to the precision recorded in the config.
        """
        path = _resolve_checkpoint_dir(path)
        config = Devstral2Config.from_json(os.path.join(path, "config.json"))
        if dtype is not None:
            config = replace(config, dtype=dtype)
        dtype = config.dtype

        with torch.device("meta"):
            model = cls(config)

        state_dict: dict[str, torch.Tensor] = {}
        for shard in _shard_files(path):
            with safe_open(shard, framework="pt", device="cpu") as f:
                for key in f.keys():
                    name = _convert_key(key)
                    if name is None:
                        continue
                    weight = f.get_tensor(key)
                    if weight.dtype == torch.float8_e4m3fn:
                        scale = f.get_tensor(key + "_scale_inv").to(dtype)
                        weight = weight.to(device=device, dtype=dtype) * scale.to(device)
                    else:
                        weight = weight.to(device=device, dtype=dtype)
                    state_dict[name] = weight

        if config.tie_word_embeddings:
            # `assign=True` installs Parameters as-is, so one Parameter under both keys stays tied.
            embed = nn.Parameter(state_dict["model.embed_tokens.weight"])
            state_dict["model.embed_tokens.weight"] = state_dict["lm_head.weight"] = embed

        model.load_state_dict(state_dict, strict=True, assign=True)
        model.to(device)  # non-persistent rope buffers are not part of the state dict
        return model.eval()

