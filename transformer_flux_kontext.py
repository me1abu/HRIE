# transformer_flux_kontext.py
# URAE FluxTransformer2DModel patched for FLUX.1-Kontext-dev compatibility.
#
# Changes vs original URAE transformer_flux.py:
#   1. forward() accepts ntk_factor and proportional_attention, and strips
#      them from joint_attention_kwargs before passing to attention blocks.
#   2. __module__ is set to the diffusers path so FluxKontextPipeline's
#      type-check accepts this class without warnings.

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import (
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    AdaLayerNormZeroSingle,
)
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from attention_processor import FluxAttnProcessor2_0

logger = logging.get_logger(__name__)


@maybe_allow_in_graph
class FluxSingleTransformerBlock(nn.Module):
    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=4.0):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm     = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp  = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=FluxAttnProcessor2_0(),
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
        )

    def forward(self, hidden_states, temb, image_rotary_emb=None,
                joint_attention_kwargs=None):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden = self.act_mlp(self.proj_mlp(norm_hidden_states))
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_out = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )
        hidden_states = torch.cat([attn_out, mlp_hidden], dim=2)
        hidden_states = gate.unsqueeze(1) * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        return hidden_states


@maybe_allow_in_graph
class FluxTransformerBlock(nn.Module):
    def __init__(self, dim, num_attention_heads, attention_head_dim,
                 qk_norm="rms_norm", eps=1e-6):
        super().__init__()
        self.norm1         = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ValueError("PyTorch 2.0+ required.")
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=FluxAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=eps,
        )
        self.norm2         = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff            = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context    = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def forward(self, hidden_states, encoder_hidden_states, temb,
                image_rotary_emb=None, joint_attention_kwargs=None):
        norm_hs,  gate_msa,   shift_mlp,   scale_mlp,   gate_mlp   = self.norm1(hidden_states, emb=temb)
        norm_enc, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder_hidden_states, emb=temb)
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_out, ctx_attn_out = self.attn(
            hidden_states=norm_hs,
            encoder_hidden_states=norm_enc,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )
        hidden_states       = hidden_states       + gate_msa.unsqueeze(1)   * attn_out
        encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * ctx_attn_out

        norm_hs  = self.norm2(hidden_states)         * (1 + scale_mlp[:, None])   + shift_mlp[:, None]
        norm_enc = self.norm2_context(encoder_hidden_states) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        hidden_states         = hidden_states         + gate_mlp.unsqueeze(1)   * self.ff(norm_hs)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * self.ff_context(norm_enc)
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        return encoder_hidden_states, hidden_states


class FluxPosEmbed(nn.Module):
    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        self.theta    = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor, ntk_factor: float = 1.0) -> Tuple:
        n_axes = ids.shape[-1]
        cos_out, sin_out = [], []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        freqs_dtype = torch.float32 if is_mps else torch.float64
        for i in range(n_axes):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i], pos[:, i],
                repeat_interleave_real=True, use_real=True,
                freqs_dtype=freqs_dtype, ntk_factor=ntk_factor,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        return (
            torch.cat(cos_out, dim=-1).to(ids.device),
            torch.cat(sin_out, dim=-1).to(ids.device),
        )


class FluxTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
    URAE FluxTransformer2DModel with Kontext-compatible forward().
    forward() accepts ntk_factor and proportional_attention, strips them
    from joint_attention_kwargs before handing off to attention blocks.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int] = (16, 56, 56),
    ):
        super().__init__()
        self.out_channels = in_channels
        self.inner_dim    = self.config.num_attention_heads * self.config.attention_head_dim
        self.pos_embed    = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        text_time_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds
            else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed  = text_time_cls(embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim)
        self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.inner_dim)
        self.x_embedder       = nn.Linear(self.config.in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList([
            FluxTransformerBlock(dim=self.inner_dim, num_attention_heads=self.config.num_attention_heads, attention_head_dim=self.config.attention_head_dim)
            for _ in range(self.config.num_layers)
        ])
        self.single_transformer_blocks = nn.ModuleList([
            FluxSingleTransformerBlock(dim=self.inner_dim, num_attention_heads=self.config.num_attention_heads, attention_head_dim=self.config.attention_head_dim)
            for _ in range(self.config.num_single_layers)
        ])
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)
        self.gradient_checkpointing = False

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        processors = {}
        def _recurse(name, module):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()
            for sub_name, child in module.named_children():
                _recurse(f"{name}.{sub_name}", child)
        for name, module in self.named_children():
            _recurse(name, module)
        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        count = len(self.attn_processors.keys())
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(f"Passed {len(processor)} processors for {count} attention layers.")
        def _recurse(name, module, proc):
            if hasattr(module, "set_processor"):
                module.set_processor(proc if not isinstance(proc, dict) else proc.pop(f"{name}.processor"))
            for sub_name, child in module.named_children():
                _recurse(f"{name}.{sub_name}", child, proc)
        for name, module in self.named_children():
            _recurse(name, module, processor)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        pooled_projections: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        img_ids: Optional[torch.Tensor] = None,
        txt_ids: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        ntk_factor: float = 1.0,
        proportional_attention: bool = False,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:

        if txt_ids is not None and txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids is not None and img_ids.ndim == 3:
            img_ids = img_ids[0]

        if joint_attention_kwargs is not None:
            joint_attention_kwargs   = dict(joint_attention_kwargs)
            lora_scale               = joint_attention_kwargs.pop("scale", 1.0)
            # Strip URAE keys so they don't reach Attention.forward()
            ntk_factor               = joint_attention_kwargs.pop("ntk_factor", ntk_factor)
            proportional_attention   = joint_attention_kwargs.pop("proportional_attention", proportional_attention)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        hidden_states             = self.x_embedder(hidden_states)
        timestep                  = timestep.to(hidden_states.dtype) * 1000
        guidance                  = guidance.to(hidden_states.dtype) * 1000 if guidance is not None else None
        temb = (
            self.time_text_embed(timestep, pooled_projections) if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids, ntk_factor=ntk_factor)

        # Block kwargs: pass proportional_attention but NOT ntk_factor (already consumed)
        block_jak = dict(joint_attention_kwargs or {})
        if proportional_attention:
            block_jak["proportional_attention"] = True

        for index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:
                def make_fw(m):
                    def fw(*a): return m(*a)
                    return fw
                ckpt_kw = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    make_fw(block), hidden_states, encoder_hidden_states, temb, image_rotary_emb, block_jak, **ckpt_kw
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=block_jak,
                )
            if controlnet_block_samples is not None:
                interval = int(np.ceil(len(self.transformer_blocks) / len(controlnet_block_samples)))
                if controlnet_blocks_repeat:
                    hidden_states = hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval]

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            if self.training and self.gradient_checkpointing:
                def make_fw(m):
                    def fw(*a): return m(*a)
                    return fw
                ckpt_kw = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    make_fw(block), hidden_states, temb, image_rotary_emb, block_jak, **ckpt_kw
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=block_jak,
                )
            if controlnet_single_block_samples is not None:
                interval = int(np.ceil(len(self.single_transformer_blocks) / len(controlnet_single_block_samples)))
                hidden_states[:, encoder_hidden_states.shape[1]:, ...] += controlnet_single_block_samples[index_block // interval]

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]
        hidden_states = self.norm_out(hidden_states, temb)
        output        = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


# ── Make the pipeline type-check happy ───────────────────────────────────────
# FluxKontextPipeline checks isinstance(transformer, diffusers FluxTransformer2DModel)
# by comparing __module__. Registering ours under the same path silences the warning.
FluxTransformer2DModel.__module__ = "diffusers.models.transformers.transformer_flux"
