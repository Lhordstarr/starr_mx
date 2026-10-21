"""A speech engine that stays loaded.

Invoked as::

    python scripts/py/tts_engine.py --runner kokoro|kokoro-onnx|transformers \
        --model-dir DIR --device auto [--repo REPO] [--model-file onnx/x.onnx]

The voice used to be made by a process that lived for exactly as long as the
utterance: it loaded the weights, said one thing, and exited. Every reply paid
for the load again, and the resource manager showed a job that was gone before
it could be read. This is the other half of that story — the same runners, kept
in one process, so the load is paid once and the engine is there for the next
reply.

The protocol is one JSON object per line in both directions.

On stdout:

    {"ready": true, "device": "cpu"}      once the model is loaded
    {"id": 1, "stage": "synthesising"}     progress, while working
    {"id": 1, "ok": true, "result": {...}} the answer
    {"id": 1, "ok": false, "error": "..."}  a failure, not a crash

On stdin:

    {"id": 1, "text": "...", "voice": "af_heart", "speed": 1.0, "out": "/abs/out.wav"}

The engine is the runner's own code, imported rather than reimplemented: this
file only knows how to keep it alive and how to talk to it, so there is one
implementation of how a voice is made.
"""
import argparse
import json
import os
import sys

# The runners live beside this file, and a script's own directory is the first
# thing on the path — but only when it is run as a script, which is how this is
# always started. Said out loud rather than assumed.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import speak  # noqa: E402
import speak_hf  # noqa: E402
import speak_onnx  # noqa: E402

RUNNERS = {
    "kokoro": speak,
    "kokoro-onnx": speak_onnx,
    "transformers": speak_hf,
}

DEFAULT_REPO = {
    "kokoro": "hexgrad/Kokoro-82M",
    "kokoro-onnx": "onnx-community/Kokoro-82M-v1.0-ONNX",
    "transformers": "",
}


def emit(payload):
    """One JSON line, flushed. The engine's only way of speaking."""
    print(json.dumps(payload), flush=True)


def build(runner, args):
    """Load the model and return the engine plus the runner module."""
    module = RUNNERS[runner]
    repo = args.repo or DEFAULT_REPO[runner]
    if runner == "kokoro":
        return module.build_engine(args.model_dir, args.device, repo), module
    if runner == "kokoro-onnx":
        return module.build_engine(args.model_dir, args.device, repo, args.model_file), module
    return module.build_engine(args.model_dir, args.device), module


def main() -> int:
    parser = argparse.ArgumentParser(description="A speech engine that stays loaded.")
    parser.add_argument("--runner", required=True, choices=sorted(RUNNERS))
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--repo", default="")
    parser.add_argument("--model-file", default="")
    args = parser.parse_args()

    try:
        engine, module = build(args.runner, args)
    except Exception as exc:
        # A failure to load is an answer, not a crash: the caller falls back to
        # the one-shot runner rather than losing the feature.
        emit({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1

    emit({"ready": True, "device": engine["landed"]})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except Exception as exc:
            emit({"ok": False, "error": f"bad request: {exc}"})
            continue
        if not isinstance(request, dict):
            emit({"ok": False, "error": "bad request: not an object"})
            continue
        rid = request.get("id")
        voice = str(request.get("voice") or "af_heart")
        try:
            speed = float(request.get("speed") or 1.0)
        except (TypeError, ValueError):
            speed = 1.0
        out_path = str(request.get("out") or "")
        text = str(request.get("text") or "")

        def progress(event, _rid=rid):
            emit({"id": _rid, **event})

        if not text.strip():
            emit({"id": rid, "ok": False, "error": "no text supplied"})
            continue
        if not out_path:
            emit({"id": rid, "ok": False, "error": "no output path supplied"})
            continue
        try:
            result = module.render(engine, text, out_path, voice, speed, progress)
            emit({"id": rid, "ok": True, "result": result})
        except Exception as exc:
            # One bad utterance must not take the engine down: the next reply
            # should not have to reload the model to find out.
            emit({"id": rid, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
