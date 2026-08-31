# pipeline_flux_kontext_urae.py
# FLUX.1-Kontext-dev + URAE high-resolution pipeline.
#
# Design notes:
#   - load_urae_lora() deliberately does NOT call fuse_lora(). Fusing bakes the
#     adapter into the base weights and makes a second adapter impossible to
#     weight independently. Left unfused, URAE and a task LoRA can be stacked
#     and blended at inference time via set_adapters().
#   - load_custom_lora() loads a second, arbitrary LoRA (local file or HF repo).
#   - set_lora_weights() blends the two adapters.

import math
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from diffusers import FluxKontextPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.utils import logging

logger = logging.get_logger(__name__)


def _ntk_factor_for_resolution(target_h: int, target_w: int, base_res: int = 1024) -> float:
    """area-ratio NTK factor: (h/1024) * (w/1024)"""
    return (target_h / base_res) * (target_w / base_res)


class FluxKontextURAEPipeline(FluxKontextPipeline):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        pipe = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        pipe.scheduler.config.use_dynamic_shifting = False
        pipe.scheduler.config.time_shift = 10
        logger.info("FluxKontextURAEPipeline: URAE scheduler settings applied.")
        return pipe

    def load_urae_lora(
        self,
        repo_id: str = "Huage001/URAE",
        weight_name: str = "urae_2k_adapter.safetensors",
        adapter_name: str = "urae_2k",
    ):
        """
        Load the URAE 2K LoRA. Does NOT fuse — keeps it stackable with
        a second custom LoRA via set_adapters().
        repo_id can be a HuggingFace repo ID or a local .safetensors file path.
        """
        import os
        if os.path.isfile(repo_id):
            # Local file — same loading pattern as load_custom_lora
            weight_dir  = os.path.dirname(repo_id) or "."
            weight_name = os.path.basename(repo_id)
            self.load_lora_weights(weight_dir, weight_name=weight_name, adapter_name=adapter_name)
        else:
            self.load_lora_weights(repo_id, weight_name=weight_name, adapter_name=adapter_name)
        logger.info(f"URAE LoRA loaded as adapter '{adapter_name}': {repo_id}")

    def load_custom_lora(
        self,
        lora_path: str,
        adapter_name: str = "custom",
    ):
        """
        Load a second, task-specific LoRA from a local path or HF repo.
        lora_path can be:
          - "/path/to/your_lora.safetensors"
          - "your-hf-username/your-repo" (weight_name auto-detected)
        """
        import os
        if os.path.isfile(lora_path):
            # Local .safetensors file
            weight_dir  = os.path.dirname(lora_path) or "."
            weight_name = os.path.basename(lora_path)
            self.load_lora_weights(weight_dir, weight_name=weight_name, adapter_name=adapter_name)
        else:
            # HF repo ID
            self.load_lora_weights(lora_path, adapter_name=adapter_name)
        logger.info(f"Custom LoRA loaded as adapter '{adapter_name}': {lora_path}")

    def set_lora_weights(
        self,
        urae_scale: float = 1.0,
        custom_scale: float = 0.8,
        urae_adapter: str = "urae_2k",
        custom_adapter: str = "custom",
    ):
        """
        Blend URAE and custom LoRA adapters. Call after loading both.
        urae_scale:   strength of URAE resolution adapter  (keep >= 0.7)
        custom_scale: strength of your fine-tuned style LoRA
        """
        self.set_adapters(
            [urae_adapter, custom_adapter],
            adapter_weights=[urae_scale, custom_scale],
        )
        logger.info(
            f"LoRA blend: '{urae_adapter}'={urae_scale}  '{custom_adapter}'={custom_scale}"
        )

    def __call__(
        self,
        image: PipelineImageInput = None,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 2.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        ntk_factor: Optional[float] = None,
        proportional_attention: Optional[bool] = None,
        **kwargs,
    ):
        # resolve dimensions
        if height is None or width is None:
            if image is not None:
                _img = image[0] if isinstance(image, (list, tuple)) else image
                if hasattr(_img, "size"):
                    _w, _h = _img.size
                elif isinstance(_img, torch.Tensor):
                    _h, _w = _img.shape[-2], _img.shape[-1]
                else:
                    _h, _w = 1024, 1024
                height = height or _h
                width  = width  or _w
            else:
                height = height or 1024
                width  = width  or 1024

        if ntk_factor is None:
            ntk_factor = _ntk_factor_for_resolution(height, width)
        if proportional_attention is None:
            proportional_attention = ntk_factor > 1.0

        jak = dict(joint_attention_kwargs or {})
        jak["ntk_factor"] = ntk_factor
        if proportional_attention:
            jak["proportional_attention"] = True

        logger.debug(
            f"URAE: {width}x{height}  ntk_factor={ntk_factor:.2f}  "
            f"prop_attn={proportional_attention}"
        )

        return super().__call__(
            image=image,
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            true_cfg_scale=true_cfg_scale,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            output_type=output_type,
            return_dict=return_dict,
            joint_attention_kwargs=jak,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
            max_area=height * width,
            _auto_resize=False,
            **kwargs,
        )
