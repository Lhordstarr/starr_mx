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

This file is also the engine behind a voice that stays loaded. `tts_engine.py`
imports `build_engine` and `render` from here, loads the checkpoint once, and
then renders as many utterances as it is asked for; the command line above is
the one-shot form of those same two calls. One implementation of how a Kokoro
voice is made, so the warm path and the one-shot path cannot drift apart.
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


def cli_report(event):
    """The command line's reporter: one JSON line per event, as it always was."""
    print(json.dumps(event), flush=True)


def local_file(repo, model_dir, rel, report=None):
    """Path to ``rel`` inside ``model_dir``, fetching it there once if missing.

    Returns None when it cannot be produced (no network and nothing cached),
    which leaves the caller to the interpreters' own hub fallback — the run
    then fails the way it did before this file was self-contained, with the
    hub's own error, rather than a confusing crash.
    """
    emit = report or cli_report
    local = os.path.join(model_dir, rel)
    if os.path.isfile(local):
        return local
    try:
        from huggingface_hub import hf_hub_download

        emit({"stage": "fetching", "file": rel})
        return hf_hub_download(repo_id=repo, filename=rel, local_dir=model_dir)
    except Exception as exc:
        emit({"stage": "fetching", "file": rel, "without": str(exc)})
        return None


def build_engine(model_dir, device, repo, report=None):
    """Load the checkpoint and the pipeline. Returns the engine.

    The expensive half: reading 327 MB of weights and building the pipeline
    around it. The engine holds both, so a caller that keeps it can render
    many utterances without paying this again.
    """
    import torch
    from kokoro import KModel, KPipeline

    kind, _index = pick_device(device)
    # Everything the 82M checkpoint needs, anchored on the downloaded weights.
    weights = os.path.join(model_dir, KModel.MODEL_NAMES.get(repo, "kokoro-v1_0.pth"))
    config = local_file(repo, model_dir, "config.json", report)

    model = KModel(
        repo_id=repo,
        config=config or None,
        model=weights if os.path.isfile(weights) else None,
    )
    # Move it here, and move it explicitly.
    #
    # KModel builds its submodules on the CPU and loads the weights with
    # map_location='cpu'. KPipeline has two branches for its model argument:
    # given a bool it constructs KModel(repo_id=...).to(device).eval(), but
    # given a KModel instance it takes it verbatim and never touches
    # `device` at all. This script builds its own KModel so it can point at
    # the downloaded weights, so it always took the second branch — and
    # Kokoro ran on the CPU on every machine while the process row said GPU.
    # It was not a slow GPU; it was not a GPU at all.
    #
    # `.to()` before anything is read, so a move that fails is a real error
    # rather than a synthesis that quietly comes out on the wrong device.
    model = model.to(kind).eval()

    # Where it actually ended up, read off the model rather than off the
    # request. This is what the resource manager is told, so a move that did
    # not take, or a torch build that quietly refused, cannot be reported as
    # the device that was asked for — the row would say GPU again and the
    # job would be slow for exactly the same reason.
    landed = next(model.parameters()).device.type

    pipeline = KPipeline(lang_code="a", repo_id=repo, model=model, device=kind)
    return {
        "kind": kind,
        "landed": landed,
        "pipeline": pipeline,
        "repo": repo,
        "model_dir": model_dir,
    }


def render(engine, text, out_path, voice, speed, report=None):
    """Say one thing with a loaded engine. Returns the result object.

    The voice is resolved per call, not per engine: the checkpoint is shared by
    every voice, and the style file beside it is a few hundred kilobytes.
    """
    import numpy as np
    import soundfile as sf

    voice_path = local_file(engine["repo"], engine["model_dir"], f"voices/{voice}.pt", report)
    chunks = []
    for _, _, audio in engine["pipeline"](text, voice=voice_path or voice, speed=speed):
        chunks.append(audio)
    if not chunks:
        raise RuntimeError("kokoro produced no audio")

    merged = np.concatenate(chunks)
    sf.write(out_path, merged, 24000)
    return {
        "path": out_path,
        "samples": int(merged.shape[0]),
        "seconds": round(merged.shape[0] / 24000, 2),
        "device": engine["landed"],
    }


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
        import torch  # noqa: F401
        from kokoro import KModel, KPipeline  # noqa: F401
    except ImportError as exc:
        print(json.dumps({"error": f"missing dependency: {exc.name}"}))
        return 1

    try:
        engine = build_engine(model_dir, device, repo)
        landed = engine["landed"]
        print(json.dumps({"device": landed}), flush=True)
        print(json.dumps({"stage": "synthesising"}), flush=True)
        result = render(engine, text, out_path, voice, speed)
        print(json.dumps(result), flush=True)
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
