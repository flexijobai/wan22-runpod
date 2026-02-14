#!/usr/bin/env python3
"""RunPod Serverless Handler for Wan2.x Video Generation

Architecture:
  - Models download to /runpod-volume/huggingface/ on first use
  - Model swapping: only one model in VRAM at a time
  - Subsequent jobs use cached models from Network Volume

Actions:
  - flf2v:  First-Last-Frame to Video (Wan2.1 FLF2V 14B 720P) - PRIMARY
  - t2v:    Text-to-Video (Wan2.2 T2V A14B MoE) - SECONDARY
  - health: Health check (GPU info, model status, cached models)

Models:
  FLF2V: Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers (story segments)
  T2V:   Wan-AI/Wan2.2-T2V-A14B-Diffusers (text-only scenes)
"""
import runpod
import base64
import os
import json
import time
import random
import tempfile
import subprocess
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────
VOLUME_DIR = "/runpod-volume"

MODEL_FLF2V = "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers"
MODEL_T2V = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"

DEFAULT_FPS = 16
DEFAULT_NEG_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, "
    "style, works, paintings, images, static, overall gray, worst quality, "
    "low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, "
    "three legs, many people in the background, walking backwards"
)

# ── Global State ────────────────────────────────────────────────────────
LOADED_MODEL = None  # "flf2v" or "t2v"
PIPE = None


def get_cache_dir():
    """Get HuggingFace cache directory on Network Volume."""
    cache_dir = os.path.join(VOLUME_DIR, "huggingface")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = cache_dir
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(cache_dir, "hub")
    return cache_dir


def unload_model():
    """Unload current model from VRAM."""
    global PIPE, LOADED_MODEL
    if PIPE is not None:
        import torch
        del PIPE
        PIPE = None
        LOADED_MODEL = None
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("MODEL: Unloaded from VRAM")


def load_flf2v():
    """Load Wan2.1 FLF2V 14B pipeline."""
    global PIPE, LOADED_MODEL
    if LOADED_MODEL == "flf2v":
        return PIPE

    if LOADED_MODEL is not None:
        print(f"MODEL: Swapping {LOADED_MODEL} → flf2v")
        unload_model()

    import torch
    from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
    from transformers import CLIPVisionModel

    cache_dir = get_cache_dir()

    print("=" * 60)
    print(f"INIT: Loading FLF2V ({MODEL_FLF2V})")
    init_start = time.time()

    # VAE and image_encoder MUST be float32 for quality
    print("INIT: [1/3] Loading image_encoder (float32)...")
    image_encoder = CLIPVisionModel.from_pretrained(
        MODEL_FLF2V,
        subfolder="image_encoder",
        torch_dtype=torch.float32,
        cache_dir=cache_dir,
    )

    print("INIT: [2/3] Loading VAE (float32)...")
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_FLF2V,
        subfolder="vae",
        torch_dtype=torch.float32,
        cache_dir=cache_dir,
    )

    print("INIT: [3/3] Loading pipeline (bfloat16)...")
    PIPE = WanImageToVideoPipeline.from_pretrained(
        MODEL_FLF2V,
        vae=vae,
        image_encoder=image_encoder,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
    )
    PIPE.to("cuda")
    LOADED_MODEL = "flf2v"

    print(f"INIT: FLF2V loaded in {time.time() - init_start:.1f}s")
    print("=" * 60)
    return PIPE


def load_t2v():
    """Load Wan2.2 T2V A14B pipeline."""
    global PIPE, LOADED_MODEL
    if LOADED_MODEL == "t2v":
        return PIPE

    if LOADED_MODEL is not None:
        print(f"MODEL: Swapping {LOADED_MODEL} → t2v")
        unload_model()

    import torch
    from diffusers import WanPipeline

    cache_dir = get_cache_dir()

    print("=" * 60)
    print(f"INIT: Loading T2V ({MODEL_T2V})")
    init_start = time.time()

    PIPE = WanPipeline.from_pretrained(
        MODEL_T2V,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
    )
    PIPE.to("cuda")
    LOADED_MODEL = "t2v"

    print(f"INIT: T2V loaded in {time.time() - init_start:.1f}s")
    print("=" * 60)
    return PIPE


# ── Helpers ─────────────────────────────────────────────────────────────

def calc_dimensions(width, height, pipe):
    """Ensure dimensions are divisible by model requirements."""
    mod = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    width = (width // mod) * mod
    height = (height // mod) * mod
    return width, height


def validate_num_frames(num_frames):
    """Ensure num_frames follows 4k+1 formula."""
    if (num_frames - 1) % 4 != 0:
        num_frames = ((num_frames - 1) // 4) * 4 + 1
    return num_frames


def frames_to_mp4(frames, fps, output_path):
    """Convert list of PIL/numpy frames to MP4."""
    import imageio
    from PIL import Image
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
    for frame in frames:
        if isinstance(frame, Image.Image):
            writer.append_data(np.array(frame))
        elif hasattr(frame, 'cpu'):
            writer.append_data(frame.cpu().numpy())
        else:
            writer.append_data(np.asarray(frame))
    writer.close()


def make_result(frames, fps, generation_time, seed, width, height):
    """Encode frames to MP4 and build response dict."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    frames_to_mp4(frames, fps, tmp_path)

    with open(tmp_path, "rb") as f:
        video_bytes = f.read()

    duration = len(frames) / fps
    os.unlink(tmp_path)

    print(f"GEN: Video {duration:.1f}s, {len(video_bytes)/1024:.0f}KB, seed={seed}")

    return {
        "video_base64": base64.b64encode(video_bytes).decode("utf-8"),
        "duration": round(duration, 2),
        "generation_time": round(generation_time, 2),
        "seed": seed,
        "width": width,
        "height": height,
        "num_frames": len(frames),
        "fps": fps,
        "video_size_kb": round(len(video_bytes) / 1024, 1),
    }


# ── Action Handlers ────────────────────────────────────────────────────

def handle_health():
    """Health check - GPU info, loaded model, cached models."""
    import torch
    result = {
        "status": "ok",
        "version": "1.0",
        "loaded_model": LOADED_MODEL,
        "models": {
            "flf2v": MODEL_FLF2V,
            "t2v": MODEL_T2V,
        },
    }
    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        result["vram_total_gb"] = round(props.total_mem / 1024**3, 1)
        result["vram_used_gb"] = round(torch.cuda.memory_allocated(0) / 1024**3, 1)

    cache_hub = os.path.join(VOLUME_DIR, "huggingface", "hub")
    if os.path.isdir(cache_hub):
        result["cached_models"] = os.listdir(cache_hub)
    else:
        result["cached_models"] = []

    return result


def handle_flf2v(job_input):
    """First-Last-Frame to Video - PRIMARY action for story segments.

    Input:
        image_base64:      str - First frame (required)
        last_image_base64: str - Last frame (required)
        prompt:            str - Motion/scene description
        width:             int - Default 1280 (720P landscape) or 720 (portrait)
        height:            int - Default 720 (landscape) or 1280 (portrait)
        num_frames:        int - Must be 4k+1, default 81 (~5s at 16fps)
        seed:              int - -1 for random
        cfg:               float - Guidance scale, default 5.0
        steps:             int - Inference steps, default 50
        fps:               int - Output FPS, default 16
    """
    import torch
    from PIL import Image
    from io import BytesIO

    # ── Validate inputs ──
    image_b64 = job_input.get("image_base64")
    last_image_b64 = job_input.get("last_image_base64")

    if not image_b64:
        return {"error": "image_base64 (first frame) is required"}
    if not last_image_b64:
        return {"error": "last_image_base64 (last frame) is required"}

    prompt = job_input.get("prompt", "")
    negative_prompt = job_input.get("negative_prompt", DEFAULT_NEG_PROMPT)
    width = job_input.get("width", 1280)
    height = job_input.get("height", 720)
    num_frames = validate_num_frames(job_input.get("num_frames", 81))
    seed = job_input.get("seed", -1)
    guidance_scale = job_input.get("cfg", 5.0)
    num_inference_steps = job_input.get("steps", 50)
    fps = job_input.get("fps", DEFAULT_FPS)

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    # ── Decode images ──
    try:
        first_image = Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGB")
        last_image = Image.open(BytesIO(base64.b64decode(last_image_b64))).convert("RGB")
    except Exception as e:
        return {"error": f"Failed to decode images: {str(e)}"}

    # ── Load pipeline ──
    pipe = load_flf2v()

    # ── Adjust dimensions ──
    width, height = calc_dimensions(width, height, pipe)
    first_image = first_image.resize((width, height), Image.LANCZOS)
    last_image = last_image.resize((width, height), Image.LANCZOS)

    # ── Generate ──
    print(f"GEN [flf2v]: {width}x{height}, {num_frames}f, seed={seed}, cfg={guidance_scale}, steps={num_inference_steps}")
    generator = torch.Generator(device="cuda").manual_seed(seed)

    start = time.time()
    output = pipe(
        image=first_image,
        last_image=last_image,
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        width=width,
        height=height,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )
    generation_time = time.time() - start

    frames = output.frames[0]
    print(f"GEN [flf2v]: {len(frames)} frames in {generation_time:.1f}s")

    return make_result(frames, fps, generation_time, seed, width, height)


def handle_t2v(job_input):
    """Text-to-Video - SECONDARY action for text-only scenes.

    Input:
        prompt:          str - Scene description (required)
        width:           int - Default 1280 (720P landscape) or 720 (portrait)
        height:          int - Default 720 (landscape) or 1280 (portrait)
        num_frames:      int - Must be 4k+1, default 81 (~5s at 16fps)
        seed:            int - -1 for random
        cfg:             float - Guidance scale, default 3.5 (Wan2.2)
        steps:           int - Inference steps, default 40 (Wan2.2)
        fps:             int - Output FPS, default 16
    """
    import torch

    prompt = job_input.get("prompt", "")
    if not prompt:
        return {"error": "prompt is required for t2v"}

    negative_prompt = job_input.get("negative_prompt", DEFAULT_NEG_PROMPT)
    width = job_input.get("width", 1280)
    height = job_input.get("height", 720)
    num_frames = validate_num_frames(job_input.get("num_frames", 81))
    seed = job_input.get("seed", -1)
    guidance_scale = job_input.get("cfg", 3.5)
    num_inference_steps = job_input.get("steps", 40)
    fps = job_input.get("fps", DEFAULT_FPS)

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    # ── Load pipeline ──
    pipe = load_t2v()

    # ── Adjust dimensions ──
    width, height = calc_dimensions(width, height, pipe)

    # ── Generate ──
    print(f"GEN [t2v]: {width}x{height}, {num_frames}f, seed={seed}, cfg={guidance_scale}, steps={num_inference_steps}")
    generator = torch.Generator(device="cuda").manual_seed(seed)

    start = time.time()
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        width=width,
        height=height,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )
    generation_time = time.time() - start

    frames = output.frames[0]
    print(f"GEN [t2v]: {len(frames)} frames in {generation_time:.1f}s")

    return make_result(frames, fps, generation_time, seed, width, height)


# ── Main Handler ────────────────────────────────────────────────────────

def handler(job):
    """RunPod job handler - routes to action handlers."""
    try:
        job_input = job["input"]
        action = job_input.get("action", "flf2v")

        if action == "health":
            return handle_health()
        elif action == "flf2v":
            return handle_flf2v(job_input)
        elif action == "t2v":
            return handle_t2v(job_input)
        elif action == "avatar":
            return {"error": "avatar not yet implemented - planned for v2"}
        else:
            return {"error": f"Unknown action: {action}. Available: health, flf2v, t2v"}

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# ── Startup ─────────────────────────────────────────────────────────────
print("=" * 60)
print("Wan2.x Video Pipeline - RunPod Serverless Handler v1.0")
print(f"PRIMARY:   FLF2V  - {MODEL_FLF2V}")
print(f"SECONDARY: T2V    - {MODEL_T2V}")
print(f"Volume: {VOLUME_DIR}")
print("Actions: health, flf2v (primary), t2v (secondary)")
print("=" * 60)

runpod.serverless.start({"handler": handler})
