"""Synthesises speech with a transformers text-to-speech pipeline.

Invoked as: python scripts/py/speak_hf.py <model_dir> <voice> <out.wav> <device> [speed] [repo]
Reads the text to speak on stdin. Prints one JSON line per stage on stdout —
notes go to stderr, which the server streams into the run log.

The argument list is the same as speak.py's, so the server can call either
runner the same way. What differs is the model: speak.py is Kokoro, a styletts2
checkpoint the `kokoro` package loads from a .pth beside its voices/, while this
one runs anything the transformers text-to-speech pipeline loads — VITS (which
is what every MMS-TTS checkpoint is), SpeechT5, Bark, FastSpeech2.

`voice` and `speed` are accepted and ignored: these checkpoints carry one voice
and the pipeline has no speed control. That is said out loud on stderr rather
than silently dropped. A setting that does nothing and says nothing is how a
working feature reads as a broken one — and the alternative, refusing the run
because the caller sent the voice the app always sends, would mean a model that
works but can never be reached.

The weights come from <model_dir> itself — the copy the Models page downloaded —
never from the hub cache, so the directory stays the single source of truth for
what is installed.

Like speak.py, this file is both a command-line runner and a loadable engine:
`tts_engine.py` imports `build_engine` and `render` from here, so a pipeline
that is loaded once can speak many times without reloading the weights.
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
    """The command line's reporter: one JSON line per event."""
    print(json.dumps(event), flush=True)


def split_text(text, limit=400):
    """The text as sentence-sized pieces of at most ``limit`` characters.

    VITS predicts a duration for the whole input at once, and it was trained on
    single utterances. Handing it a 20,000-character reply in one call puts the
    duration predictor far outside anything it saw in training, which comes back
    as a stretched or clipped reading rather than an error. Splitting on sentence
    ends keeps every call inside the range it knows; a wall of text with no
    punctuation is cut at a word boundary so nothing is dropped.
    """
    import re
    pieces = [p.strip() for p in re.split(r"(?<=[.!?…])\s+", text) if p.strip()]
    out = []
    for piece in pieces:
        while len(piece) > limit:
            cut = piece.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit
            out.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if piece:
            out.append(piece)
    return out


def write_wav(path, samples, rate):
    """16-bit PCM, written with the standard library.

    numpy ships with transformers, but soundfile does not — and unlike Kokoro's
    environment this one is the shared Python venv, where a missing audio
    library must not be the reason a model cannot speak.
    """
    import numpy as np
    import wave

    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    np.clip(audio, -1.0, 1.0, out=audio)
    pcm = (audio * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(int(rate))
        fh.writeframes(pcm.tobytes())
    return int(pcm.shape[0])


def build_engine(model_dir, device, report=None):
    """Load the pipeline. Returns the engine.

    The expensive half, and the one worth not repeating: reading the weights
    and building the pipeline around them. The engine holds the pipeline, so
    the next utterance does not pay this again.
    """
    import numpy as np  # noqa: F401
    import torch  # noqa: F401
    from transformers import pipeline

    kind, _index = pick_device(device)
    use_gpu = kind != "cpu"
    pipe = pipeline("text-to-speech", model=model_dir, device=f"cuda:{_index}" if use_gpu else "cpu")

    # Where the weights actually ended up, read off the model rather than off
    # the request.
    landed = next(pipe.model.parameters()).device.type
    return {"pipe": pipe, "landed": landed, "kind": kind}


def render(engine, text, out_path, voice, speed, report=None):
    """Say one thing with a loaded engine. Returns the result object."""
    import numpy as np

    emit = report or cli_report
    if voice or speed != 1.0:
        print(
            "This model has a single voice and no speed control — "
            f"the voice ({voice or 'default'}) and speed ({speed:g}) settings do not apply.",
            file=sys.stderr,
            flush=True,
        )

    pieces = split_text(text)
    emit({"stage": "synthesising", "chunks": len(pieces)})

    rate = None
    rendered = []
    for i, piece in enumerate(pieces):
        out = engine["pipe"](piece)
        if not isinstance(out, dict) or "audio" not in out:
            raise RuntimeError(f"the pipeline returned {type(out).__name__}, not audio")
        if rate is None:
            rate = int(out.get("sampling_rate") or 16000)
        elif int(out.get("sampling_rate") or rate) != rate:
            # Mixing rates in one file would be a pitch shift partway
            # through, which is worse than refusing.
            raise RuntimeError(
                f"the pipeline changed sample rate mid-run ({rate} → {out['sampling_rate']})"
            )
        rendered.append(np.asarray(out["audio"], dtype=np.float32).reshape(-1))
        print(f"chunk {i + 1}/{len(pieces)}", file=sys.stderr, flush=True)

    if not rendered:
        raise RuntimeError("no text survived splitting")

    merged = np.concatenate(rendered)
    samples = write_wav(out_path, merged, rate)
    return {
        "path": out_path,
        "samples": samples,
        "seconds": round(samples / rate, 2),
        "device": engine["landed"],
    }


def main() -> int:
    if len(sys.argv) < 5:
        print("usage: speak_hf.py <model_dir> <voice> <out.wav> <device> [speed] [repo]", file=sys.stderr)
        return 2
    model_dir, voice, out_path, device = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    speed = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    text = sys.stdin.read().strip()

    if not text:
        print(json.dumps({"error": "no text supplied"}))
        return 1

    print(json.dumps({"stage": "loading"}), flush=True)
    try:
        import numpy as np  # noqa: F401
        import torch  # noqa: F401
        from transformers import pipeline  # noqa: F401
    except ImportError as exc:
        print(json.dumps({"error": f"missing dependency: {exc.name}"}))
        return 1

    try:
        engine = build_engine(model_dir, device)
        landed = engine["landed"]
        print(json.dumps({"device": landed}), flush=True)
        result = render(engine, text, out_path, voice, speed)
        print(json.dumps(result), flush=True)
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
