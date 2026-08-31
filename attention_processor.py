# attention_processor.py
# URAE attention processor patched for FLUX.1-Kontext-dev
#
# Changes vs original URAE:
#   1. FluxAttnProcessor2_0: train_seq_len is now None by default — it is set
#      dynamically from the actual sequence length at the first call, so
#      proportional attention auto-calibrates to Kontext's longer sequences
#      (ref-image tokens + target tokens + text tokens).
#   2. FluxAttnAdaptationProcessor2_0 gets the same dynamic default.
#   3. Both processors accept a `train_seq_len` override if you want to pin
#      the baseline explicitly (e.g. for consistency across batches).

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from diffusers.models.attention_processor import Attention
from typing import Optional
from diffusers.models.embeddings import apply_rotary_emb


class FluxAttnProcessor2_0:
    """
    Attention processor for FLUX / URAE, patched for Kontext compatibility.

    train_seq_len controls the proportional attention baseline:
      - None  (default): auto-set to the actual sequence length on the first
                         forward pass, then frozen.  This means proportional
                         attention is a no-op at standard FLUX resolution and
                         naturally compensates at higher resolutions.
      - int   : pin to a fixed value (original URAE behaviour: 512 + 64*64).
    """

    def __init__(self, train_seq_len: Optional[int] = None):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "FluxAttnProcessor2_0 requires PyTorch 2.0+."
            )
        self._train_seq_len = train_seq_len  # None → auto-detect on first call
        self._auto_set = False

    @property
    def train_seq_len(self):
        return self._train_seq_len

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        proportional_attention: bool = False,
    ) -> torch.FloatTensor:

        batch_size, _, _ = (
            hidden_states.shape if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        query = attn.to_q(hidden_states)
        key   = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim  = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key   = key.view(batch_size,   -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if encoder_hidden_states is not None:
            enc_q = attn.add_q_proj(encoder_hidden_states)
            enc_k = attn.add_k_proj(encoder_hidden_states)
            enc_v = attn.add_v_proj(encoder_hidden_states)

            enc_q = enc_q.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_k = enc_k.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_v = enc_v.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_added_q is not None:
                enc_q = attn.norm_added_q(enc_q)
            if attn.norm_added_k is not None:
                enc_k = attn.norm_added_k(enc_k)

            query = torch.cat([enc_q, query], dim=2)
            key   = torch.cat([enc_k, key],   dim=2)
            value = torch.cat([enc_v, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key   = apply_rotary_emb(key,   image_rotary_emb)

        # --- proportional attention scale -----------------------------------
        # Auto-detect train_seq_len from the actual sequence on first call.
        # This makes proportional=True safe to pass unconditionally: it is a
        # no-op at the training resolution and scales up correctly beyond it.
        actual_seq = key.size(2)
        if proportional_attention:
            if self._train_seq_len is None:
                # First call: freeze to current seq_len so ratio = 1 at this res
                self._train_seq_len = actual_seq
                self._auto_set = True
            attention_scale = math.sqrt(
                math.log(actual_seq, self._train_seq_len) / head_dim
            )
        else:
            attention_scale = math.sqrt(1.0 / head_dim)
        # --------------------------------------------------------------------

        hidden_states = F.scaled_dot_product_attention(
            query, key, value,
            dropout_p=0.0, is_causal=False,
            scale=attention_scale,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, :encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1]:],
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states

        return hidden_states


class FluxAttnAdaptationProcessor2_0(nn.Module):
    """
    Minor-component adapter for URAE 4K stage.
    Same dynamic train_seq_len behaviour as FluxAttnProcessor2_0 above.
    """

    def __init__(
        self,
        rank: int = 16,
        dim: int = 3072,
        to_out: bool = False,
        train_seq_len: Optional[int] = None,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "FluxAttnAdaptationProcessor2_0 requires PyTorch 2.0+."
            )
        self.to_q_a = nn.Linear(dim, rank, bias=False)
        self.to_q_b = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.to_q_b.weight)

        self.to_k_a = nn.Linear(dim, rank, bias=False)
        self.to_k_b = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.to_k_b.weight)

        self.to_v_a = nn.Linear(dim, rank, bias=False)
        self.to_v_b = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.to_v_b.weight)

        if to_out:
            self.to_out_a = nn.Linear(dim, rank, bias=False)
            self.to_out_b = nn.Linear(rank, dim, bias=False)
            nn.init.zeros_(self.to_out_b.weight)

        self._train_seq_len = train_seq_len
        self._auto_set = False

    @property
    def train_seq_len(self):
        return self._train_seq_len

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        proportional_attention: bool = False,
    ) -> torch.FloatTensor:

        batch_size, _, _ = (
            hidden_states.shape if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        query = attn.to_q(hidden_states) + self.to_q_b(self.to_q_a(hidden_states))
        key   = attn.to_k(hidden_states) + self.to_k_b(self.to_k_a(hidden_states))
        value = attn.to_v(hidden_states) + self.to_v_b(self.to_v_a(hidden_states))

        inner_dim = key.shape[-1]
        head_dim  = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key   = key.view(batch_size,   -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if encoder_hidden_states is not None:
            enc_q = attn.add_q_proj(encoder_hidden_states)
            enc_k = attn.add_k_proj(encoder_hidden_states)
            enc_v = attn.add_v_proj(encoder_hidden_states)

            enc_q = enc_q.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_k = enc_k.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_v = enc_v.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_added_q is not None:
                enc_q = attn.norm_added_q(enc_q)
            if attn.norm_added_k is not None:
                enc_k = attn.norm_added_k(enc_k)

            query = torch.cat([enc_q, query], dim=2)
            key   = torch.cat([enc_k, key],   dim=2)
            value = torch.cat([enc_v, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key   = apply_rotary_emb(key,   image_rotary_emb)

        actual_seq = key.size(2)
        if proportional_attention:
            if self._train_seq_len is None:
                self._train_seq_len = actual_seq
                self._auto_set = True
            attention_scale = math.sqrt(
                math.log(actual_seq, self._train_seq_len) / head_dim
            )
        else:
            attention_scale = math.sqrt(1.0 / head_dim)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value,
            dropout_p=0.0, is_causal=False,
            scale=attention_scale,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, :encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1]:],
            )
            if hasattr(self, "to_out_a"):
                hidden_states = (
                    attn.to_out[0](hidden_states)
                    + self.to_out_b(self.to_out_a(hidden_states))
                )
            else:
                hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states

        return hidden_states
