"""Synthesises speech with the Kokoro 82M model.

Invoked as: python scripts/py/speak.py <model_dir> <voice> <out.wav> <device> [speed] [repo]
Reads the text to speak on stdin, so an arbitrarily long prompt never has to fit
in an argv entry. Prints one JSON line on stdout.

The weights come from <model_dir> itself — the copy the Models page downloaded —
rather than from the Hugging Face hub cache, which used to hold a second, hidden
327 MB copy that the "downloaded" badge never mentioned. The small files that sit
beside the weights (config.json, voices/<voice>.pt) are fetched into <model_dir>
on first use if they are missing, so the directory becomes self-contained and
later runs touch no network at all.
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
        if want == "rocm" and "ROCM_HOME" not in os.environ:
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
        return "rocm" if os.environ.get("ROCM_HOME") or "rocm" in getattr(torch.version, "hip", "") else "cuda"
    except Exception:
        return "cuda"


def local_file(repo, model_dir, rel):
    """Path to ``rel`` inside ``model_dir``, fetching it there once if missing.

    Returns None when it cannot be produced (no network and nothing cached),
    which leaves the caller to the interpreters' own hub fallback — the run
    then fails the way it did before this file was self-contained, with the
    hub's own error, rather than a confusing crash.
    """
    local = os.path.join(model_dir, rel)
    if os.path.isfile(local):
        return local
    try:
        from huggingface_hub import hf_hub_download
        print(json.dumps({"stage": "fetching", "file": rel}), flush=True)
        return hf_hub_download(repo_id=repo, filename=rel, local_dir=model_dir)
    except Exception as exc:
        print(json.dumps({"stage": "fetching", "file": rel, "without": str(exc)}), flush=True)
        return None


def main() -> int:
    if len(sys.argv) < 5:
        print("usage: speak.py <model_dir> <voice> <out.wav> <device> [speed] [repo]", file=sys.stderr)
        return 2
    model_dir, voice, out_path, device = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    speed = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    repo = sys.argv[6] if len(sys.argv) > 6 else "hexgrad/Kokoro-82M"
    text = sys.stdin.read().strip()

    if not text:
        print(json.dumps({"error": "no text supplied"}))
        return 1

    print(json.dumps({"stage": "loading"}), flush=True)
    try:
        import torch
        from kokoro import KModel, KPipeline
    except ImportError as exc:
        print(json.dumps({"error": f"missing dependency: {exc.name}"}))
        return 1

    kind, _index = pick_device(device)
    use_gpu = kind != "cpu"
    # Told to the server as soon as it is known, so the resource manager shows
    # where this is really running rather than what was requested.
    print(json.dumps({"device": vendor_name() if use_gpu else "cpu"}), flush=True)
    try:
        # Everything the 82M checkpoint needs, anchored on the downloaded weights.
        weights = os.path.join(model_dir, KModel.MODEL_NAMES.get(repo, "kokoro-v1_0.pth"))
        config = local_file(repo, model_dir, "config.json")
        voice_path = local_file(repo, model_dir, f"voices/{voice}.pt")

        model = KModel(
            repo_id=repo,
            config=config or None,
            model=weights if os.path.isfile(weights) else None,
        )
        pipeline = KPipeline(lang_code="a", repo_id=repo, model=model)
        print(json.dumps({"stage": "synthesising"}), flush=True)
        chunks = []
        for _, _, audio in pipeline(text, voice=voice_path or voice, speed=speed):
            chunks.append(audio)
        if not chunks:
            print(json.dumps({"error": "kokoro produced no audio"}))
            return 1

        import numpy as np
        import soundfile as sf

        merged = np.concatenate(chunks)
        sf.write(out_path, merged, 24000)
        print(
            json.dumps(
                {
                    "path": out_path,
                    "samples": int(merged.shape[0]),
                    "seconds": round(merged.shape[0] / 24000, 2),
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
