"""Generates an image with a diffusers pipeline.

Invoked as: python scripts/py/generate.py <model_dir> <out.png> <device> <steps> <w>x<h>
Reads the prompt on stdin and any negative prompt in NEGATIVE_PROMPT. Prints one
JSON line on stdout; progress goes to stderr so it can be streamed to the UI.
"""
import gc
import json
import os
import sys

def pick_device(requested):
    """Resolve a requested device to ('cuda'|'cpu', index).

    PyTorch's ROCm build deliberately aliases ``torch.cuda`` onto HIP, so this
    same code drives both vendors — the only difference is what to call it in
    the output. ``rocm`` is accepted as an explicit request because callers
    pass the vendor they detected.
    """
    import torch
    want = (requested or "cpu").lower()
    available = torch.cuda.is_available()
    if want == "cpu":
        return "cpu", None
    if want in ("auto", "rocm", "cuda", "gpu", "both"):
        if not available:
            return "cpu", None
        if want == "rocm" and "ROCM_HOME" not in __import__("os").environ:
            # ROCm builds set this; a CUDA build does not.
            return "cuda", 0
        return "cuda", 0
    return "cpu", None


def vendor_name():
    import torch
    try:
        if "ROCM_VERSION" in getattr(torch, "__config__", type("c", (), {"show": lambda *a: ""})).show():
            return "rocm"
    except Exception:
        pass
    try:
        import os
        return "rocm" if os.environ.get("ROCM_HOME") or "rocm" in getattr(torch.version, "hip", "") else "cuda"
    except Exception:
        return "cuda"


def weight_variant(model_dir):
    """Which safetensors variant the tree actually ships, or None.

    The catalog's diffusers bundles download fp16 weights only — they halve
    the bytes, and a CPU load upcasts them to float32 anyway. Without
    ``variant="fp16"`` diffusers looks for ``diffusion_pytorch_model.bin``
    beside each config, finds only the fp16 twin, and reports "no file named
    ..." — which reads like a torn download rather than a filename mismatch.
    A tree with plain weights keeps the default variant; mixed trees too, so
    a repo that was fetched outside the catalog still loads.
    """
    import os
    has_plain = has_fp16 = False
    for _root, _dirs, files in os.walk(model_dir):
        for name in files:
            if not name.endswith(".safetensors"):
                continue
            if name.endswith(".fp16.safetensors"):
                has_fp16 = True
            else:
                has_plain = True
    return "fp16" if has_fp16 and not has_plain else None


def guidance_for(model_dir):
    """Classifier-free guidance for this tree, or None to use the default.

    sd-turbo and sdxl-turbo are distilled to skip the unconditional pass:
    their model cards ask for ``guidance_scale=0`` with one or two steps. The
    pipeline default is 7.5, which is guidance the distilled model was never
    trained to follow — at one step it produces coloured noise. The folder
    name is the only signal a downloaded tree carries, and every
    guidance-free model in the catalog has "turbo" in it.
    """
    return 0.0 if "turbo" in os.path.basename(os.path.normpath(model_dir)).lower() else None


def physical_ram_bytes():
    """Total RAM in bytes, or 0 when this platform will not say."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return 0


def cpu_has_native_bf16():
    """True when bfloat16 arithmetic runs on the chip, not in a fallback.

    A CPU without AVX512-BF16 or AMX still *runs* bf16 — by converting to
    fp32 for every operation and back — and the conversion is what kills it.
    Measured on a Skylake i5: 0.3 GFLOPS against 55 for plain fp32, so one
    64x64 step took four and a half minutes instead of a second. That is not
    slow, and waiting does not turn it into a useful image.
    """
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if line.startswith("flags") or line.startswith("Features"):
                    flags = f" {line.split(':', 1)[-1]} "
                    return "avx512_bf16" in flags or "amx_bf16" in flags
    except OSError:
        pass
    try:
        import torch
        return "BF16" in torch.backends.cpu.get_cpu_capability().upper()
    except Exception:
        return False


def cpu_dtype():
    """The dtype a CPU pipeline should load in, and the reason in one line.

    fp32 weights for a model this size are about 4.5 GB. Under 7 GB of total
    RAM the smaller build is the only one that loads — slowly, because the
    CPU cannot run bf16 quickly — and over it fp32 is both faster and safe.
    """
    import torch
    if cpu_has_native_bf16():
        return torch.bfloat16, "the CPU runs bf16 natively"
    total = physical_ram_bytes()
    if total and total < 7 * 1024**3:
        return torch.bfloat16, "no native bf16 and under 7 GB of RAM, so the smaller build"
    return torch.float32, "no native bf16, so fp32 rather than its slow fallback"


def main() -> int:
    if len(sys.argv) < 6:
        print("usage: generate.py <model_dir> <out.png> <device> <steps> <WxH>", file=sys.stderr)
        return 2
    model_dir, out_path, device, steps, size = sys.argv[1:6]
    steps = int(steps)
    width, height = (int(x) for x in size.lower().split("x"))
    guidance = guidance_for(model_dir)
    prompt = sys.stdin.read().strip()
    negative = os.environ.get("NEGATIVE_PROMPT", "").strip()

    if not prompt:
        print(json.dumps({"error": "no prompt supplied"}))
        return 1

    print(json.dumps({"stage": "loading"}), flush=True)
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
    except ImportError as exc:
        print(json.dumps({"error": f"missing dependency: {exc.name}"}))
        return 1

    kind, _index = pick_device(device)
    use_gpu = kind != "cpu"
    dtype, why = (
        (torch.float16, "the GPU's native half precision")
        if use_gpu
        else cpu_dtype()
    )
    # Said out loud before anything loads: the server shows this in the
    # resource manager, where the request ("auto") is not the answer. The
    # dtype rides along because on a CPU it decides whether the wait is
    # seconds or an hour.
    print(
        json.dumps(
            {
                "device": vendor_name() if use_gpu else "cpu",
                "dtype": str(dtype).replace("torch.", ""),
                "why": why,
            }
        ),
        flush=True,
    )
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            # None when the tree has plain weights; "fp16" for the fp16-only
            # bundles the catalog downloads.
            variant=weight_variant(model_dir),
        )
        if use_gpu:
            # Keeps VRAM down on small cards by moving each module onto the GPU
            # as it runs. This is a method on the pipeline, not a load kwarg —
            # passed to from_pretrained it was silently ignored, and the whole
            # pipeline still went to the card at once. Needs `accelerate`; a
            # plain .to() stays as the fallback.
            try:
                pipe.enable_model_cpu_offload()
            except Exception:
                pipe = pipe.to(f"cuda:{_index}")
        else:
            pipe = pipe.to("cpu")
        pipe.set_progress_bar_config(disable=True)

        print(json.dumps({"stage": "generating", "steps": steps}), flush=True)
        prompt_embeds = negative_embeds = None
        # The same rule the pipeline applies: guidance above 1 means the
        # unconditional half is needed too. A turbo model runs without it.
        do_cfg = guidance is None or guidance > 1
        if not use_gpu:
            # The text encoder runs once, to turn the prompt into embeddings;
            # after that its weights are dead weight. On a laptop freeing them
            # is what leaves room for the denoiser. When guidance is on, both
            # halves are built here because the pipeline would otherwise reach
            # for the encoder again for the unconditional one — after it has
            # been dropped.
            with torch.no_grad():
                prompt_embeds, negative_embeds = pipe.encode_prompt(
                    prompt,
                    device="cpu",
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=do_cfg,
                    negative_prompt=negative or None,
                )
            pipe.text_encoder = None
            gc.collect()
        # Only one of each pair may be forwarded: the pipeline refuses both a
        # prompt and its embeddings, and a text prompt would sent it back to
        # the encoder that was just freed.
        image = pipe(
            prompt=None if prompt_embeds is not None else prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt=None if negative_embeds is not None else (negative or None),
            negative_prompt_embeds=negative_embeds,
            num_inference_steps=steps,
            width=width,
            height=height,
            **({} if guidance is None else {"guidance_scale": guidance}),
        ).images[0]
        image.save(out_path)
        print(
            json.dumps(
                {
                    "path": out_path,
                    "width": image.width,
                    "height": image.height,
                    "device": vendor_name() if use_gpu else "cpu",
                }
            ),
            flush=True,
        )
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())