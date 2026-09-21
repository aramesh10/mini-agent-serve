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
from flashinfer.activation import silu_and_mul
from flashinfer.decode import trtllm_batch_decode_with_kv_cache
from flashinfer.norm import fused_add_rmsnorm, fused_add_rmsnorm_quant
from flashinfer.page import append_paged_kv_cache
from flashinfer.prefill import trtllm_batch_context_with_kv_cache
from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
from safetensors import safe_open

from miniagentserve.models.devstral2.configs import Devstral2Config
from miniagentserve.models.devstral2.utils import _convert_key, _shard_files
from miniagentserve.models.model import Model
from miniagentserve.models.utils import _resolve_checkpoint_dir

__all__ = ["Devstral2Model", "Devstral2ForCausalLM"]

FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max

_WORKSPACE_BYTES = 128 * 1024 * 1024
_workspaces: dict[torch.device, torch.Tensor] = {}

def _workspace(device: torch.device) -> torch.Tensor:
    if device not in _workspaces:
        _workspaces[device] = torch.zeros(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    return _workspaces[device]

def _yarn_cos_sin_cache(config: Devstral2Config) -> torch.Tensor:
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
        attention_scaling = get_mscale(factor, config.rope_mscale) / get_mscale(
            factor, config.rope_mscale_all_dim
        )
    else:
        attention_scaling = get_mscale(factor)

    # fp32 throughout: cos/sin of huge positions are not representable in bf16.
    positions = torch.arange(config.max_position_embeddings, dtype=torch.float, device="cpu")
    freqs = positions[:, None] * inv_freq
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1) * attention_scaling


def quantize_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (x / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8)


class FP8Linear(nn.Module):

    def __init__(self, in_features: int, out_features: int, dtype: torch.dtype):
        super().__init__()
        self.out_dtype = dtype
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=FP8))
        self.register_buffer("weight_scale_inv", torch.empty(()))
        self.register_buffer("activation_scale", torch.empty(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x` is either activations to quantise, or ones a fused norm already quantised with this
        layer's `activation_scale`."""
        if x.dtype is not FP8:
            x = quantize_fp8(x, self.activation_scale)
        return torch._scaled_mm(
            x, self.weight.t(), scale_a=self.activation_scale, scale_b=self.weight_scale_inv,
            out_dtype=self.out_dtype,
        )


@dataclass
class AttentionMetadata:
    """Per-step inputs shared by every attention layer."""

    position_ids: torch.Tensor             
    cos_sin_cache: torch.Tensor             
    block_tables: torch.Tensor              
    batch_indices: torch.Tensor             
    kv_indptr: torch.Tensor                 
    kv_last_page_len: torch.Tensor          
    seq_lens: torch.Tensor
    max_seq_len: int                        
    q_len: int
    cum_seq_lens_q: torch.Tensor | None   
    cum_seq_lens_kv: torch.Tensor | None
    workspace: torch.Tensor


class Devstral2Attention(nn.Module):
    """Grouped-query attention with rope and llama-4 style query scaling."""

    def __init__(self, config: Devstral2Config):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.llama_4_scaling_beta = config.llama_4_scaling_beta
        self.original_max_position_embeddings = config.rope_original_max_position_embeddings

        heads, kv_heads, dtype = config.num_attention_heads, config.num_key_value_heads, config.dtype
        self.q_size, self.kv_size = heads * self.head_dim, kv_heads * self.head_dim
        self.qkv_proj = FP8Linear(config.hidden_size, self.q_size + 2 * self.kv_size, dtype)
        self.o_proj = FP8Linear(heads * self.head_dim, config.hidden_size, dtype)

    def forward(
        self, hidden_states: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, attn: AttentionMetadata
    ) -> torch.Tensor:
        query_states, key_states, value_states = self.qkv_proj(hidden_states).split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        value_states = value_states.unflatten(-1, (-1, self.head_dim))

        apply_rope_with_cos_sin_cache_inplace(
            attn.position_ids, query_states, key_states, self.head_dim, attn.cos_sin_cache,
            is_neox=True,
        )
        query_states = query_states.unflatten(-1, (-1, self.head_dim))
        key_states = key_states.unflatten(-1, (-1, self.head_dim))

        append_paged_kv_cache(
            key_states, value_states, attn.batch_indices, attn.position_ids, (k_cache, v_cache),
            attn.block_tables.flatten(), attn.kv_indptr, attn.kv_last_page_len, kv_layout="NHD",
        )

        if self.llama_4_scaling_beta:
            attn_scale = 1 + self.llama_4_scaling_beta * torch.log1p(
                torch.floor(attn.position_ids.float() / self.original_max_position_embeddings)
            )
            query_states = query_states * attn_scale[:, None, None].to(query_states.dtype)

        kv = (k_cache, v_cache)
        if attn.q_len == 1:
            attn_output = trtllm_batch_decode_with_kv_cache(
                query=query_states, kv_cache=kv, workspace_buffer=attn.workspace,
                block_tables=attn.block_tables, seq_lens=attn.seq_lens,
                max_seq_len=attn.max_seq_len, bmm1_scale=self.scaling, bmm2_scale=1.0,
                kv_layout="NHD",
            )
        else:
            attn_output = trtllm_batch_context_with_kv_cache(
                query=query_states, kv_cache=kv, workspace_buffer=attn.workspace,
                block_tables=attn.block_tables, seq_lens=attn.seq_lens,
                max_q_len=attn.q_len, max_kv_len=attn.max_seq_len, bmm1_scale=self.scaling,
                bmm2_scale=1.0, batch_size=attn.seq_lens.numel(),
                cum_seq_lens_q=attn.cum_seq_lens_q, cum_seq_lens_kv=attn.cum_seq_lens_kv,
                kv_layout="NHD",
            )
        return self.o_proj(attn_output.flatten(1))


class Devstral2MLP(nn.Module):
    def __init__(self, config: Devstral2Config):
        super().__init__()
        self.gate_up_proj = FP8Linear(config.hidden_size, 2 * config.intermediate_size, config.dtype)
        self.down_proj = FP8Linear(config.intermediate_size, config.hidden_size, config.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(silu_and_mul(self.gate_up_proj(x), enable_pdl=True))


class Devstral2DecoderLayer(nn.Module):
    def __init__(self, config: Devstral2Config):
        super().__init__()
        self.self_attn = Devstral2Attention(config)
        self.mlp = Devstral2MLP(config)

        self.input_layernorm = nn.Parameter(torch.ones(config.hidden_size))
        self.post_attention_layernorm = nn.Parameter(torch.ones(config.hidden_size))
        self.eps = config.rms_norm_eps

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, k_cache: torch.Tensor,
        v_cache: torch.Tensor, attn: AttentionMetadata
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.empty(hidden_states.shape, dtype=FP8, device=hidden_states.device)
        fused_add_rmsnorm_quant(
            x, hidden_states, residual, self.input_layernorm,
            self.self_attn.qkv_proj.activation_scale, self.eps,
            enable_pdl=True,
        )
        hidden_states = self.self_attn(x, k_cache, v_cache, attn)
        fused_add_rmsnorm_quant(
            x, hidden_states, residual, self.post_attention_layernorm,
            self.mlp.gate_up_proj.activation_scale, self.eps,
            enable_pdl=True,
        )
        return self.mlp(x), residual


class Devstral2Model(nn.Module):
    def __init__(self, config: Devstral2Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Devstral2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.Parameter(torch.ones(config.hidden_size))
        self.register_buffer("cos_sin_cache", _yarn_cos_sin_cache(config), persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_tables: torch.Tensor,
        cache_lens: torch.Tensor,
    ) -> torch.Tensor:
        batch, q_len = input_ids.shape
        device = input_ids.device
        q_pos = (cache_lens[:, None] + torch.arange(q_len, device=device)).to(torch.int32)
        # Token-major from here on: that is the layout the fused norms and attention kernels take.
        hidden_states = self.embed_tokens(input_ids).flatten(0, 1)
        seq_lens = (cache_lens + q_len).to(torch.int32)

        cum_seq_lens_q = cum_seq_lens_kv = None
        if q_len == 1:
            max_seq_len = k_cache.shape[1] * k_cache.shape[2]
        else:
            max_seq_len = block_tables.shape[1] * k_cache.shape[2]  # upper bound, no device sync
            cum_seq_lens_q = torch.arange(batch + 1, dtype=torch.int32, device=device) * q_len
            cum_seq_lens_kv = F.pad(seq_lens.cumsum(0, dtype=torch.int32), (1, 0))

        attn = AttentionMetadata(
            position_ids=q_pos.flatten(), 
            cos_sin_cache=self.cos_sin_cache,
            block_tables=block_tables,
            batch_indices=torch.arange(batch, dtype=torch.int32, device=device).repeat_interleave(q_len),
            kv_indptr=torch.arange(batch + 1, dtype=torch.int32, device=device) * block_tables.shape[1],
            kv_last_page_len=(seq_lens - 1) % k_cache.shape[2] + 1,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            q_len=q_len,
            cum_seq_lens_q=cum_seq_lens_q,
            cum_seq_lens_kv=cum_seq_lens_kv,
            workspace=_workspace(device),
        )

        residual = torch.zeros_like(hidden_states)
        for layer, k, v in zip(self.layers, k_cache, v_cache):
            hidden_states, residual = layer(hidden_states, residual, k, v, attn)
        fused_add_rmsnorm(
            hidden_states, residual, self.norm, self.config.rms_norm_eps, enable_pdl=True
        )
        return hidden_states.unflatten(0, (batch, q_len))


class Devstral2ForCausalLM(Model):
    def __init__(self, config: Devstral2Config):
        super().__init__()
        self.config = config
        self.model = Devstral2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, *, logits_to_keep: int = 0, **kwargs: torch.Tensor) -> torch.Tensor:
        hidden_states = self.model(input_ids, **kwargs)
        if logits_to_keep:
            hidden_states = hidden_states[:, -logits_to_keep:]
        return self.lm_head(hidden_states)

    # ---- checkpoint loading -------------------------------------------------

    @classmethod
    def from_pretrained(
        cls, path: str, dtype: torch.dtype | None = None, device: str | torch.device = "cuda"
    ) -> Devstral2ForCausalLM:
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
                    if key.endswith(("_scale_inv", "_scale")):
                        weight = weight.float()
                    elif weight.dtype != FP8:
                        weight = weight.to(dtype)
                    state_dict[name] = weight.to(device)

        for name in [k for k in state_dict if k.endswith("mlp.gate_proj.weight")]:
            mlp = name[: -len("gate_proj.weight")]
            for suffix in ("weight", "weight_scale_inv", "activation_scale"):
                gate, up = (state_dict.pop(f"{mlp}{proj}_proj.{suffix}") for proj in ("gate", "up"))
                assert suffix == "weight" or torch.equal(gate, up), f"{mlp}{suffix}: gate != up"
                state_dict[f"{mlp}gate_up_proj.{suffix}"] = torch.cat([gate, up]) if suffix == "weight" else gate

        for name in [k for k in state_dict if k.endswith("self_attn.q_proj.weight")]:
            attn = name[: -len("q_proj.weight")]
            parts = {
                suffix: [state_dict.pop(f"{attn}{proj}_proj.{suffix}") for proj in "qkv"]
                for suffix in ("weight", "weight_scale_inv", "activation_scale")
            }
            act = parts["activation_scale"]
            assert all(torch.equal(act[0], a) for a in act), f"{attn}activation_scale: q/k/v differ"
            scales = parts["weight_scale_inv"]
            if all(torch.equal(scales[0], s) for s in scales):
                scale, weights = scales[0], parts["weight"]
            else:  # requantise to the largest scale so a single per-tensor scale is exact-ish
                scale = torch.stack(scales).max()
                weights = [(w.float() * (s / scale)).clamp(-FP8_MAX, FP8_MAX).to(FP8)
                           for w, s in zip(parts["weight"], scales)]
            state_dict[f"{attn}qkv_proj.weight"] = torch.cat(weights)
            state_dict[f"{attn}qkv_proj.weight_scale_inv"] = scale
            state_dict[f"{attn}qkv_proj.activation_scale"] = act[0]

        if config.tie_word_embeddings:
            embed = nn.Parameter(state_dict["model.embed_tokens.weight"])
            state_dict["model.embed_tokens.weight"] = state_dict["lm_head.weight"] = embed

        model.load_state_dict(state_dict, strict=True, assign=True)
        model.to(device)
        return model.eval()

