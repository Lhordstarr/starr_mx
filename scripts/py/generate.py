"""Generates an image with a diffusers pipeline.

Invoked as: python scripts/py/generate.py <model_dir> <out.png> <device> <steps> <w>x<h>
Reads the prompt on stdin and any negative prompt in NEGATIVE_PROMPT. Prints one
JSON line on stdout; progress goes to stderr so it can be streamed to the UI.
"""
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


def main() -> int:
    if len(sys.argv) < 6:
        print("usage: generate.py <model_dir> <out.png> <device> <steps> <WxH>", file=sys.stderr)
        return 2
    model_dir, out_path, device, steps, size = sys.argv[1:6]
    steps = int(steps)
    width, height = (int(x) for x in size.lower().split("x"))
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
    # Said out loud before anything loads: the server shows this in the
    # resource manager, where the request ("auto") is not the answer.
    print(json.dumps({"device": vendor_name() if use_gpu else "cpu"}), flush=True)
    try:
        # GPU: fp16 is the standard half-precision. CPU: fp32 would hold ~5 GB
        # of weights for a model this size alone — more than a small laptop has
        # — and the kernel OOM-kills the loader before the first step. bfloat16
        # keeps every weight at two bytes and modern PyTorch has real CPU
        # kernels for it, so the whole pipeline fits in about half the RAM.
        dtype = torch.float16 if use_gpu else torch.bfloat16
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
        image = pipe(
            prompt=prompt,
            negative_prompt=negative or None,
            num_inference_steps=steps,
            width=width,
            height=height,
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