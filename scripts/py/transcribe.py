"""Transcribes an audio file with Whisper via transformers.

Invoked by the server as:  python scripts/py/transcribe.py <model_dir> <audio> <device>
Prints one JSON line on stdout; all logging goes to stderr.
"""
import json
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


def main() -> int:
    if len(sys.argv) < 4:
        print("usage: transcribe.py <model_dir> <audio_path> <device>", file=sys.stderr)
        return 2
    model_dir, audio, device = sys.argv[1], sys.argv[2], sys.argv[3]

    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:
        print(json.dumps({"error": f"missing dependency: {exc.name}"}))
        return 1

    kind, _index = pick_device(device)
    use_gpu = kind != "cpu"
    # Resolved before the model is built; the server reads this line out of the
    # running job's output so the resource manager can label it truthfully.
    print(json.dumps({"device": vendor_name() if use_gpu else "cpu"}), flush=True)
    print(json.dumps({"stage": "loading"}), flush=True)
    try:
        asr = pipeline(
            "automatic-speech-recognition",
            model=model_dir,
            device=f"cuda:{_index}" if use_gpu else "cpu",
            dtype=torch.float16 if use_gpu else torch.float32,
        )
        print(json.dumps({"stage": "transcribing"}), flush=True)

        # Timestamps are not portable across ASR architectures.
        #
        # `True` is Whisper's segment timestamps, and only Whisper's: a CTC model
        # — wav2vec2, HuBERT, WavLM, MMS — rejects it outright with "CTC can
        # either predict character level timestamps, or word level timestamps",
        # which is a hard failure rather than a missing extra. Those models take
        # 'word'. Anything else is asked without timestamps; the transcript is
        # the point, and a model that cannot report them should still transcribe.
        kind = type(asr.model).__name__
        try:
            if kind.endswith("ForCTC"):
                out = asr(audio, return_timestamps="word")
            elif "Whisper" in kind:
                out = asr(audio, return_timestamps=True)
            else:
                out = asr(audio)
        except ValueError as exc:
            # A CTC tokenizer that cannot do word alignment raises the same class
            # of error as the one above. Losing the word times is acceptable;
            # losing the transcription is not.
            if "timestamp" not in str(exc).lower():
                raise
            out = asr(audio)
        print(
            json.dumps(
                {
                    "text": out.get("text", "").strip(),
                    "chunks": [
                        {"start": c.get("timestamp", [None, None])[0],
                         "end": c.get("timestamp", [None, None])[1],
                         "text": c.get("text", "").strip()}
                        for c in out.get("chunks", [])
                    ],
                }
            ),
            flush=True,
        )
    except Exception as exc:  # surfaced to the browser, not swallowed
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())