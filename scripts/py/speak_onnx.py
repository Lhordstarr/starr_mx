"""Synthesises speech with the ONNX export of Kokoro 82M.

Invoked as: python scripts/py/speak_onnx.py <model_dir> <voice> <out.wav> <device> [speed] [repo] [model_file]
Reads the text to speak on stdin, so an arbitrarily long prompt never has to fit
in an argv entry. Prints one JSON line per progress step, then one answer.

The ONNX graph itself is only the acoustic model. Two things around it are still
needed and are deliberately taken from the directory the Models page downloaded:

  * G2P and chunking come from the `kokoro` package, the same English front end
    the PyTorch runner uses. Running the graph without it would mean shipping a
    second phonemiser and getting a different pronunciation of the same text.
  * The voice is `voices/<voice>.bin`, a (510, 1, 256) float32 table exported by
    the same repository. Row `len(phonemes)` is the style vector for a chunk —
    the index is the phoneme count, not the token count, which is easy to get
    backwards.

The model file is found next to this script's `model_dir`; the caller may pass
the exact repo-relative path as the seventh argument, but the fallback order
means a directory with any of the repository's quantisations still works.

Like the other two runners, this file is both a command-line runner and a
loadable engine: `tts_engine.py` imports `build_engine` and `render` from here,
so the ONNX session — which takes seconds to build — is paid for once rather
than once per utterance.
"""
import glob
import json
import os
import sys


def pick_providers(requested):
    """Map the app's device vocabulary onto ONNX Runtime providers.

    The shared ONNX Runtime wheel is CPU-only; ``onnxruntime-gpu`` adds CUDA or
    ROCm. Reporting the provider the session actually got is the same rule the
    PyTorch runner follows: never print the request back as if it were a fact.
    """
    import onnxruntime as ort

    available = ort.get_available_providers()
    want = (requested or "cpu").lower()
    if want != "cpu":
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "ROCMExecutionProvider" in available:
            return ["ROCMExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def provider_label(providers):
    first = providers[0] if providers else "CPUExecutionProvider"
    if first == "CUDAExecutionProvider":
        return "cuda"
    if first == "ROCMExecutionProvider":
        return "rocm"
    return "cpu"


def cli_report(event):
    """The command line's reporter: one JSON line per event."""
    print(json.dumps(event), flush=True)


def local_file(repo, model_dir, rel, report=None):
    """Path to ``rel`` inside ``model_dir``, fetching it there once if missing."""
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


def find_model(model_dir, requested):
    """The ONNX file to load, preferring what the catalog names."""
    if requested:
        p = requested if os.path.isabs(requested) else os.path.join(model_dir, requested)
        if os.path.isfile(p):
            return p
    onnx_dir = os.path.join(model_dir, "onnx")
    for name in (
        "model_q4f16.onnx",
        "model_q4.onnx",
        "model_quantized.onnx",
        "model_uint8.onnx",
        "model.onnx",
        "model_fp16.onnx",
    ):
        p = os.path.join(onnx_dir, name)
        if os.path.isfile(p):
            return p
    found = sorted(glob.glob(os.path.join(onnx_dir, "*.onnx")))
    return found[0] if found else None


def build_engine(model_dir, device, repo, model_file="", report=None):
    """Load the ONNX session and the Kokoro front end. Returns the engine.

    Building an inference session takes seconds — several on a cold cache — so
    this is the half worth doing once and keeping.
    """
    import numpy as np  # noqa: F401
    import onnxruntime as ort
    import soundfile as sf  # noqa: F401
    from kokoro import KPipeline

    model_path = find_model(model_dir, model_file)
    if not model_path:
        raise RuntimeError("no ONNX model found beside this download")

    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    if not os.path.isfile(tokenizer_path):
        raise RuntimeError("tokenizer.json is missing — download the whole set")
    with open(tokenizer_path, "r", encoding="utf-8") as fh:
        vocab = json.load(fh).get("model", {}).get("vocab", {})
    if not vocab:
        raise RuntimeError("tokenizer.json carries no phoneme vocabulary")

    providers = pick_providers(device)
    sess = ort.InferenceSession(model_path, providers=providers)
    landed = provider_label(sess.get_providers())

    # model=False asks only for the front end: G2P, sentence chunking and
    # the 510-phoneme limit. It never constructs a torch model, so this
    # path needs the kokoro package's text code but no checkpoint.
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", model=False)

    # A caller may hand us the PyTorch repository id by mistake; the voice and
    # the graph only exist in the ONNX repository, so use the export's id for
    # any on-demand voice fetch either way.
    voice_repo = repo if "ONNX" in repo.upper() or "onnx" in repo else "onnx-community/Kokoro-82M-v1.0-ONNX"
    return {
        "sess": sess,
        "landed": landed,
        "vocab": vocab,
        "pipeline": pipeline,
        "model_dir": model_dir,
        "voice_repo": voice_repo,
    }


def render(engine, text, out_path, voice, speed, report=None):
    """Say one thing with a loaded engine. Returns the result object."""
    import numpy as np
    import soundfile as sf

    emit = report or cli_report
    voice_path = local_file(engine["voice_repo"], engine["model_dir"], f"voices/{voice}.bin", report)
    if not voice_path:
        raise RuntimeError(f"voice {voice} is not downloaded")
    voices = np.fromfile(voice_path, dtype=np.float32)
    if voices.size % 256:
        raise RuntimeError("voice file is not a float32 style table")
    voices = voices.reshape(-1, 1, 256)

    vocab = engine["vocab"]
    chunks = []
    for result in engine["pipeline"](text, voice=None, speed=speed):
        phonemes = result.phonemes
        if not phonemes:
            continue
        ids = [vocab[p] for p in phonemes if p != "$" and p in vocab]
        if not ids:
            continue
        if len(phonemes) >= voices.shape[0]:
            raise RuntimeError(f"phoneme chunk too long ({len(phonemes)}); no style row")
        style = voices[len(phonemes)]
        audio = engine["sess"].run(
            None,
            {
                "input_ids": np.array([[0, *ids, 0]], dtype=np.int64),
                "style": style.astype(np.float32),
                "speed": np.array([speed], dtype=np.float32),
            },
        )[0]
        chunks.append(np.asarray(audio).reshape(-1))

    if not chunks:
        raise RuntimeError("Kokoro produced no audio")
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
        print(
            "usage: speak_onnx.py <model_dir> <voice> <out.wav> <device> "
            "[speed] [repo] [model_file]",
            file=sys.stderr,
        )
        return 2
    model_dir, voice, out_path, device = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    speed = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    repo = sys.argv[6] if len(sys.argv) > 6 else "onnx-community/Kokoro-82M-v1.0-ONNX"
    requested = sys.argv[7] if len(sys.argv) > 7 else ""
    text = sys.stdin.read().strip()

    if not text:
        print(json.dumps({"error": "no text supplied"}))
        return 1

    print(json.dumps({"stage": "loading"}), flush=True)
    try:
        import numpy as np  # noqa: F401
        import onnxruntime as ort  # noqa: F401
        import soundfile as sf  # noqa: F401
        from kokoro import KPipeline  # noqa: F401
    except ImportError as exc:
        print(json.dumps({"error": f"missing dependency: {exc.name}"}))
        return 1

    try:
        engine = build_engine(model_dir, device, repo, requested)
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
