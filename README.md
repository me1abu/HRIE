# HRIE: High-Resolution Image Editing

**with FLUX.1-Kontext-dev**

<!-- Add teaser.jpg here: a 3-panel strip of input | stock Kontext | HRIE at the same crop -->

> **Edit images at their own resolution.** FLUX.1-Kontext-dev natively works around 1 megapixel, so `diffusers` silently downscales anything larger before editing and hands you back a smaller image. HRIE removes that ceiling: feed it a 2048×1536 photo and you get an edited 2048×1536 photo.

## 🪶Features

- **Resolution-preserving editing😊**: output matches your input resolution (snapped to a multiple of 16), instead of being resized down to Kontext's ~1MP envelope and back up again.
- **No structural artifacts at high resolution🎯**: an ultra-resolution adapter plus NTK-scaled RoPE keeps geometry stable well past the native resolution, where stock Kontext duplicates structures and smears texture.
- **Stackable LoRAs🎨**: the resolution adapter stays unfused, so your own task LoRA loads alongside it with an independent weight.
- **Minimal code changes🚀**: three swapped imports and two extra lines. Everything else is the `FluxKontextPipeline` API you already use.
- **Batch-ready⚡**: an optimized batch runner with `torch.compile`, flash SDPA, and BF16 end to end.

## Quick Start

- Install [PyTorch](https://pytorch.org/get-started/locally/), [diffusers](https://huggingface.co/docs/diffusers/index) (0.35.0+), [transformers](https://huggingface.co/docs/transformers/index), and [peft](https://huggingface.co/docs/peft/index).

- Clone this repo into your project directory:

```bash
git clone https://github.com/me1abu/HRIE.git
cd HRIE
```

- **You only need minimal modifications!**

```diff
  import torch
  from PIL import Image
- from diffusers import FluxKontextPipeline
+ from pipeline_flux_kontext_urae import FluxKontextURAEPipeline
+ from transformer_flux_kontext import FluxTransformer2DModel

  bfl_repo = "black-forest-labs/FLUX.1-Kontext-dev"
+ transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=torch.bfloat16)
- pipe = FluxKontextPipeline.from_pretrained(bfl_repo, torch_dtype=torch.bfloat16)
+ pipe = FluxKontextURAEPipeline.from_pretrained(bfl_repo, transformer=transformer, torch_dtype=torch.bfloat16)
  pipe.to("cuda")

+ pipe.load_urae_lora()

  image = Image.open("input.jpg").convert("RGB")   # e.g. 2048x1536
  result = pipe(
      image=image,
      prompt="your edit instruction",
-     height=1024,
-     width=1024,
+     height=1536,
+     width=2048,
      guidance_scale=4.0,
      num_inference_steps=20,
      generator=torch.Generator("cuda").manual_seed(0),
  ).images[0]
  result.save("edited.png")
```

`ntk_factor` and `proportional_attention` are derived automatically from the requested resolution — pass them explicitly only if you want to override.

⚠️ **~48GB of GPU memory is recommended at 2K.** Below that, call `pipe.enable_sequential_cpu_offload()` (or pass `--offload` to the batch script) at the cost of speed.

## Installation

```bash
git clone https://github.com/me1abu/HRIE.git
cd HRIE
conda create -n hrie python=3.12
conda activate hrie
pip install -r requirements.txt
```

Tested on `torch==2.4` and `diffusers==0.35`, on an RTX Pro 6000 (Blackwell). `FluxKontextPipeline` first shipped in diffusers 0.35.0, so anything older will fail on import.

Base FLUX.1-Kontext-dev weights are gated on the Hub — run `huggingface-cli login` and accept the model terms first.

### Adapter weights

The 2K adapter downloads automatically on the first `load_urae_lora()` call:

**[`urae_2k_adapter.safetensors`](https://huggingface.co/Huage001/URAE/blob/main/urae_2k_adapter.safetensors)** (`Huage001/URAE`)

To use a local copy, pass the path directly:

```python
pipe.load_urae_lora(repo_id="/path/to/urae_2k_adapter.safetensors")
```

## Inference

#### Resolution behaviour

Stock `FluxKontextPipeline` resizes your input to the nearest of its preferred ~1MP resolutions (`1024×1024`, `1248×832`, `1568×672`, …). HRIE passes `_auto_resize=False` and sets `max_area` from your actual target, so the requested resolution is what gets denoised and what comes back. The batch script snaps dimensions down to the nearest multiple of 16, which is a VAE constraint, not a resolution cap.

#### Stacking your own LoRA

The resolution adapter is loaded **unfused**, so a second adapter can be blended against it with an independent weight:

```python
pipe.load_urae_lora()                                    # resolution
pipe.load_custom_lora("/path/to/your_lora.safetensors")  # style / task
pipe.set_lora_weights(urae_scale=1.0, custom_scale=0.8)
```

If your LoRA was trained with a trigger token, include it in the prompt.

#### Batch

```bash
python batch_urae_kontext.py \
  --input_dir ./in --output_dir ./out \
  --prompt "your edit instruction" \
  --custom_lora /path/to/your_lora.safetensors
```

`--steps 15` for previews, `--offload` for smaller GPUs, `--no_compile` to skip the compile warmup.

## How It Works

The high-resolution technique comes from URAE, which targets FLUX.1-dev text-to-image. Kontext is a different model in the ways that matter here: its attention sequence concatenates reference-image tokens, target-image tokens, and text tokens, so the sequence-length and positional assumptions don't transfer unchanged. Three things had to change.

**1. Proportional attention became resolution-relative instead of hardcoded.**
The upstream implementation scales attention against a fixed training sequence length (`512 + 64*64`, the text-to-image layout). Kontext's sequence is longer and varies with the reference image, making that constant wrong at every resolution including the native one. Here `train_seq_len` defaults to `None`, calibrates from the actual sequence length on the first forward pass, then freezes — a no-op at whatever resolution you start from, scaling correctly above it. Pin it explicitly if you need a fixed baseline across batches.

**2. NTK-scaled RoPE is routed through a pipeline that doesn't know it exists.**
`FluxKontextPipeline` forwards `joint_attention_kwargs` straight to `Attention.forward()`, which rejects unknown keys. The transformer's `forward()` intercepts `ntk_factor` and `proportional_attention`, consumes the first for positional embedding, passes the second to the attention processors, and strips both before they reach attention. The NTK factor is derived from the target area ratio, `(h/1024) * (w/1024)`.

**3. Adapters stay unfused so a second LoRA can stack.**
The upstream loader fuses its adapter into the base weights, which makes independent weighting of a task LoRA impossible. Both stay unfused and blend via `set_adapters()`.

Two smaller things: the scheduler needs `use_dynamic_shifting=False` with `time_shift=10`, which Kontext does not default to, and the transformer's `__module__` is re-registered under the diffusers path so `FluxKontextPipeline`'s `isinstance` check accepts it.

No forked diffusers required — stock `get_1d_rotary_pos_embed` already accepts `ntk_factor`.

## Performance

The batch runner applies `torch.compile` (`reduce-overhead`, `fullgraph`, static shapes), the flash SDPA backend, BF16 end to end with no FP32 VAE upcast, and a reduced step count. Combined, this took one 2K workload from roughly 80s/image to ~15s/image in steady state on an RTX Pro 6000, excluding a one-time ~2–3 min compile warmup.

Per-optimization attribution is **not** measured here. The `--no_compile`, `--no_fa3`, and `--no_cuda_graphs` flags exist so you can isolate contributions on your own hardware. Treat the aggregate as a single data point, one GPU, one resolution.

⚠️ Do not disable the math or mem-efficient SDPA backends to force flash. The VAE runs attention at spatial sizes where flash has no valid kernel, and PyTorch raises `No available kernel` rather than falling back. Enable flash on top and leave the others available; SDPA selects per call.

## Files

| File | Purpose |
|---|---|
| `attention_processor.py` | Attention processors with dynamic `train_seq_len` |
| `transformer_flux_kontext.py` | Transformer with Kontext-compatible `forward()` |
| `pipeline_flux_kontext_urae.py` | NTK derivation, resolution passthrough, stackable LoRA loading |
| `batch_urae_kontext.py` | Optimized batch runner |

## Acknowledgement

- [URAE](https://github.com/Huage001/URAE) (Apache-2.0) for the ultra-resolution adaptation method and the 2K adapter weights.
- [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) by [Black Forest Labs](https://blackforestlabs.ai/) for the base model.
- [diffusers](https://github.com/huggingface/diffusers) for the pipeline and attention code base.

Model weights are not redistributed here. FLUX.1-Kontext-dev ships under Black Forest Labs' non-commercial license — check it before any commercial use, as the license on this repo grants you no rights to the model.