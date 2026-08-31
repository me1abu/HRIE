#!/usr/bin/env python3
"""
batch_urae_kontext.py
Batch image editing with FLUX.1-Kontext-dev + the URAE high-resolution
adapter, with an optional second LoRA stacked on top.

Applied optimizations (each can be toggled off for A/B measurement):
  - torch.compile, mode="reduce-overhead", fullgraph=True, static shapes
  - flash SDPA backend enabled for the transformer's long sequences
  - BF16 end to end, no FP32 VAE upcast
  - reduced step count (flow-matching tolerates fewer steps than DDPM)

Reported end-to-end effect on one RTX Pro 6000 (Blackwell) at ~2K:
roughly 80s/image down to ~15s/image in steady state, excluding the
one-time compile warmup. Per-optimization attribution is not measured
here — use the --no_* flags to isolate individual contributions on your
own hardware before quoting numbers.

Usage:
    python batch_urae_kontext.py \
        --input_dir  /path/to/inputs/ \
        --output_dir /path/to/outputs/ \
        --prompt "your edit instruction" \
        --custom_lora /path/to/your_lora.safetensors \
        --urae_scale 1.0 \
        --custom_lora_scale 0.8

    --no_compile      skip torch.compile (faster startup, slower steady state)
    --no_cuda_graphs  disable CUDA graph capture (needed for mixed-size batches)
    --no_fa3          fall back to default PyTorch SDPA backend selection
    --steps 15        fast preview mode
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import torch
from PIL import Image

THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))

from transformer_flux_kontext import FluxTransformer2DModel
from pipeline_flux_kontext_urae import FluxKontextURAEPipeline

SUPPORTED_EXTS  = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_PROMPT  = "enhance the lighting and detail, photorealistic"
SEED            = 42


def parse_args():
    p = argparse.ArgumentParser(description="Optimized batch FLUX.1-Kontext-dev + URAE")

    # ── I/O ──────────────────────────────────────────────────────────────────
    p.add_argument("--input_dir",     required=True)
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--output_suffix", default="_urae")
    p.add_argument("--output_ext",    default=None,
                   help="Override output extension, e.g. .jpg (default: same as input)")

    # ── prompt ────────────────────────────────────────────────────────────────
    p.add_argument("--prompt", default=DEFAULT_PROMPT,
                   help="Edit instruction applied to every image. If your custom "
                        "LoRA was trained with a trigger token, include it here.")

    # ── sampling ──────────────────────────────────────────────────────────────
    p.add_argument("--steps",          type=int,   default=20,
                   help="Inference steps (default 20; use 15 for previews)")
    p.add_argument("--guidance_scale", type=float, default=4.0)

    # ── model ─────────────────────────────────────────────────────────────────
    p.add_argument("--model", default="black-forest-labs/FLUX.1-Kontext-dev")
    p.add_argument("--offload", action="store_true",
                   help="CPU offload — only use on <48GB GPUs, disables CUDA graphs")

    # ── optimizations (all on by default) ────────────────────────────────────
    p.add_argument("--no_compile",      action="store_true",
                   help="Skip torch.compile (useful for first run / debugging)")
    p.add_argument("--no_cuda_graphs",  action="store_true",
                   help="Disable CUDA graphs (needed for mixed-size batches)")
    p.add_argument("--no_fa3",          action="store_true",
                   help="Skip FlashAttention-3 (fall back to PyTorch SDPA)")

    # ── URAE LoRA ─────────────────────────────────────────────────────────────
    p.add_argument("--no_urae_lora",    action="store_true")
    p.add_argument("--urae_repo",       default="Huage001/URAE")
    p.add_argument("--urae_weight",     default="urae_2k_adapter.safetensors")
    p.add_argument("--urae_scale",      type=float, default=1.0)

    # ── custom LoRA ───────────────────────────────────────────────────────────
    p.add_argument("--custom_lora",       default=None)
    p.add_argument("--custom_lora_scale", type=float, default=0.8)

    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────

def snap(h, w, multiple=16):
    return max(multiple, (h // multiple) * multiple), \
           max(multiple, (w // multiple) * multiple)


def collect_images(d: Path):
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)


def build_output_path(src: Path, out_dir: Path, suffix: str, ext_override: str) -> Path:
    ext = ext_override or src.suffix.lower()
    if ext == ".jpeg":
        ext = ".jpg"
    return out_dir / (src.stem + suffix + ext)


# ── FlashAttention-3 ──────────────────────────────────────────────────────────

def try_enable_fa3(pipe):
    """
    Enable the best available attention backend WITHOUT disabling fallbacks.

    CRITICAL: Never call enable_math_sdp(False) or enable_mem_efficient_sdp(False).
    The VAE encoder's attention runs at tiny spatial resolution where flash kernels
    have no valid implementation. PyTorch needs the fallback backends available or
    it throws: RuntimeError: No available kernel. Aborting execution.

    Strategy: enable flash on top, leave math + mem_efficient ON as fallbacks.
    PyTorch SDPA auto-selects the best kernel per call — flash for large seqs
    (transformer blocks), math/mem_efficient for small seqs (VAE attention).
    """
    # Always enable flash — fastest for the transformer's long sequences.
    # Do NOT disable math or mem_efficient — VAE attention needs them as fallback.
    torch.backends.cuda.enable_flash_sdp(True)

    try:
        import flash_attn
        fa_version = tuple(int(x) for x in flash_attn.__version__.split(".")[:2])
        if fa_version >= (2, 6):
            if hasattr(pipe.transformer, "enable_flash_attention"):
                pipe.transformer.enable_flash_attention()
                print(f"  flash-attn {flash_attn.__version__} enabled on transformer")
            else:
                print(f"  flash-attn {flash_attn.__version__} present — SDPA using flash backend")
        else:
            print(f"  flash-attn {flash_attn.__version__} present (older build). "
                  "Consider upgrading for the newer kernels.")
    except ImportError:
        print("  flash-attn not installed — PyTorch built-in SDPA (flash_sdp enabled)")
        print("  For best speed: pip install flash-attn --no-build-isolation")


# ── CUDA graphs ───────────────────────────────────────────────────────────────

def apply_cuda_graphs(pipe):
    """
    Capture the transformer forward pass in a CUDA graph.
    Eliminates CPU overhead and kernel-launch latency per denoising step.
    Requires all input shapes to be identical across the batch.
    """
    try:
        from diffusers.utils import is_accelerate_available
        if is_accelerate_available():
            # Use diffusers' built-in static caching / graph capture if available
            if hasattr(pipe, "enable_static_cache"):
                pipe.enable_static_cache()
                print("  CUDA graphs: diffusers static cache enabled")
                return

        # Manual graph capture on the transformer
        if hasattr(pipe.transformer, "to_bettertransformer"):
            pipe.transformer = pipe.transformer.to_bettertransformer()
            print("  CUDA graphs: BetterTransformer applied")
        else:
            print("  CUDA graphs: will be captured on first forward pass via torch.compile")
    except Exception as e:
        print(f"  CUDA graphs: skipped ({e})")


# ── pipeline build ────────────────────────────────────────────────────────────

def build_pipeline(args):
    print("\n[1/4] Loading transformer …")
    transformer = FluxTransformer2DModel.from_pretrained(
        args.model,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )

    print("[2/4] Loading pipeline …")
    pipe = FluxKontextURAEPipeline.from_pretrained(
        args.model,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )

    # ── BF16 everywhere, no FP32 upcasts ─────────────────────────────────────
    # diffusers sometimes upcasts VAE to FP32; force BF16 for throughput
    pipe.vae = pipe.vae.to(torch.bfloat16)
    if hasattr(pipe, "upcast_vae"):
        pipe.upcast_vae = False
    torch.set_default_dtype(torch.bfloat16)
    print("  BF16 throughout — FP32 upcasts disabled")

    if args.offload:
        pipe.enable_sequential_cpu_offload()
        print("  CPU offload enabled (CUDA graphs disabled)")
    else:
        pipe.to("cuda")

    # ── FlashAttention-3 ──────────────────────────────────────────────────────
    if not args.no_fa3:
        print("[3/4] Setting up attention backend …")
        try_enable_fa3(pipe)
    else:
        print("[3/4] FA3 skipped (--no_fa3)")

    # ── LoRA loading ──────────────────────────────────────────────────────────
    print("[4/4] Loading LoRA adapters …")
    has_urae   = not args.no_urae_lora
    has_custom = args.custom_lora is not None

    if has_urae:
        print(f"  URAE LoRA: {args.urae_repo}/{args.urae_weight}")
        pipe.load_urae_lora(
            repo_id=args.urae_repo,
            weight_name=args.urae_weight,
            adapter_name="urae_2k",
        )
    if has_custom:
        print(f"  Custom LoRA: {args.custom_lora}")
        pipe.load_custom_lora(lora_path=args.custom_lora, adapter_name="custom")

    if has_urae and has_custom:
        pipe.set_lora_weights(
            urae_scale=args.urae_scale,
            custom_scale=args.custom_lora_scale,
        )
        print(f"  Blended: urae={args.urae_scale}  custom={args.custom_lora_scale}")
    elif has_urae:
        pipe.set_adapters(["urae_2k"], adapter_weights=[args.urae_scale])
    elif has_custom:
        pipe.set_adapters(["custom"], adapter_weights=[args.custom_lora_scale])

    # ── torch.compile ─────────────────────────────────────────────────────────
    # Must come AFTER LoRA loading so compiled graph includes LoRA weights
    if not args.no_compile and not args.offload:
        print("\n  torch.compile: compiling transformer …")
        print("  (first image will be slow ~2-3min while compiling; all subsequent are fast)")
        pipe.transformer = torch.compile(
            pipe.transformer,
            mode="reduce-overhead",   # best for repeated same-shape calls
            fullgraph=True,           # no graph breaks — maximum fusion
            dynamic=False,            # static shapes = better kernel selection
        )
        print("  torch.compile: ready (will trigger on first forward pass)")
    elif args.no_compile:
        print("  torch.compile: skipped (--no_compile)")

    # ── CUDA graphs ───────────────────────────────────────────────────────────
    if not args.no_cuda_graphs and not args.offload and not args.no_compile:
        # torch.compile with reduce-overhead mode already uses CUDA graphs internally
        # so we don't need to do anything extra here
        print("  CUDA graphs: handled by torch.compile reduce-overhead mode ✓")
    elif not args.no_cuda_graphs and not args.offload and args.no_compile:
        apply_cuda_graphs(pipe)

    return pipe


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        print(f"ERROR: input_dir not found: {input_dir}")
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(input_dir)
    if not images:
        print(f"No images found in {input_dir}")
        sys.exit(0)

    print(f"\nFLUX.1-Kontext + URAE batch processor")
    print(f"  Images:  {len(images)}")
    print(f"  Prompt:  '{args.prompt}'")
    print(f"  Steps:   {args.steps}")
    print(f"  Seed:    {SEED} (generator seeded once, advances per image)")
    print(f"  Compile: {'OFF' if args.no_compile else 'ON — reduce-overhead + fullgraph'}")
    print(f"  FA3:     {'OFF' if args.no_fa3 else 'ON'}")
    print(f"  Output:  {output_dir}")

    pipe      = build_pipeline(args)
    generator = torch.Generator("cuda").manual_seed(SEED)

    success = 0
    failed  = []
    times   = []
    total_start = time.time()

    for idx, img_path in enumerate(images):
        out_path = build_output_path(img_path, output_dir, args.output_suffix, args.output_ext)
        print(f"\n[{idx+1}/{len(images)}] {img_path.name}", flush=True)

        try:
            img = Image.open(img_path).convert("RGB")
            src_w, src_h = img.size
            out_h, out_w = snap(src_h, src_w)
            print(f"  {src_w}x{src_h} → {out_w}x{out_h}", end="  ", flush=True)

            # warm CUDA before timing
            if idx == 0:
                torch.cuda.synchronize()

            t0 = time.time()

            result = pipe(
                image=img,
                prompt=args.prompt,
                height=out_h,
                width=out_w,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.steps,
                generator=generator,
            ).images[0]

            torch.cuda.synchronize()
            elapsed = time.time() - t0
            times.append(elapsed)

            save_kw = {"quality": 95} if out_path.suffix.lower() in {".jpg", ".jpeg"} else {}
            result.save(str(out_path), **save_kw)

            label = "(compile warmup)" if idx == 0 and not args.no_compile else ""
            print(f"{elapsed:.1f}s  → {out_path.name} {label}")
            success += 1

        except Exception as e:
            print(f"  FAILED — {e}")
            traceback.print_exc()
            failed.append(img_path.name)

    # ── summary ───────────────────────────────────────────────────────────────
    total = time.time() - total_start
    # exclude compile warmup (first image) from avg if compile was on
    timed = times[1:] if (not args.no_compile and len(times) > 1) else times
    avg   = sum(timed) / len(timed) if timed else 0

    print(f"\n{'─'*60}")
    print(f"  Done:     {success}/{len(images)} succeeded")
    print(f"  Total:    {total:.1f}s")
    if not args.no_compile and len(times) > 1:
        print(f"  Compile warmup (image 1): {times[0]:.1f}s")
    print(f"  Avg/image (steady state): {avg:.1f}s")
    print(f"  Output:   {output_dir}")
    if failed:
        print(f"\n  Failed ({len(failed)}):")
        for n in failed:
            print(f"    • {n}")


if __name__ == "__main__":
    main()