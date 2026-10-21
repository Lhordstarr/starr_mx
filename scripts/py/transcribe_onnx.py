"""Transcribes audio with ONNX Whisper.

Invoked by the server as:  python scripts/py/transcribe_onnx.py <model_dir> <audio> <device>
Prints one JSON line per progress step, then one answer.

The repository is the transformers.js-style ONNX export: an encoder graph and a
single merged decoder graph, plus the usual Whisper tokenizer and feature
extractor files. This runner deliberately does **not** use ONNX Runtime's
cache branch. The merged graph's past-key-value inputs are an optimisation, and
handing them back with the wrong layout makes the model lose the audio after the
first few tokens — it degenerates into "the, the, the" while still looking like a
successful transcription. Recomputing the decoder sequence each step is slower
but correct, and a short dictation clip is small enough that the difference is
seconds rather than minutes.

Audio is decoded to 16 kHz mono through ffmpeg (or PyAV when ffmpeg is absent).
Long clips are split into consecutive 30-second windows, the length Whisper was
trained on; each window gets its own transcript chunk.
"""
import json
import os
import shutil
import subprocess
import sys
import glob


CHUNK_SECONDS = 30.0
SAMPLE_RATE = 16000


def pick_providers(requested):
    """Map the app's device vocabulary onto ONNX Runtime providers."""
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


def decode_ffmpeg(path):
    exe = shutil.which("ffmpeg")
    if not exe:
        return None
    proc = subprocess.run(
        [
            exe,
            "-v",
            "error",
            "-i",
            path,
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "pipe:1",
        ],
        capture_output=True,
        timeout=180,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(detail[-1] if detail else "ffmpeg could not decode this audio")
    import numpy as np

    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def decode_pyav(path):
    """Fallback for machines where the transformers runner has PyAV but no ffmpeg."""
    import av
    import numpy as np

    frames = []
    with av.open(path) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                frames.append(out.to_ndarray().reshape(-1))
        try:
            for out in resampler.resample(None):
                frames.append(out.to_ndarray().reshape(-1))
        except Exception:
            pass
    return np.concatenate(frames).astype(np.float32) if frames else np.zeros(0, np.float32)


def decode_audio(path):
    audio = decode_ffmpeg(path)
    if audio is None:
        audio = decode_pyav(path)
    if audio.size == 0:
        raise RuntimeError("the audio file decoded to zero samples")
    return audio


def find_graphs(model_dir):
    """Encoder + merged decoder, preferring the int8 export the catalog ships."""
    onnx_dir = os.path.join(model_dir, "onnx")
    pairs = [
        ("encoder_model_quantized.onnx", "decoder_model_merged_quantized.onnx"),
        ("encoder_model.onnx", "decoder_model_merged.onnx"),
        ("encoder_model_int8.onnx", "decoder_model_merged_int8.onnx"),
    ]
    for enc, dec in pairs:
        ep, dp = os.path.join(onnx_dir, enc), os.path.join(onnx_dir, dec)
        if os.path.isfile(ep) and os.path.isfile(dp):
            return ep, dp
    encoders = sorted(glob.glob(os.path.join(onnx_dir, "encoder_model*.onnx")))
    decoders = sorted(glob.glob(os.path.join(onnx_dir, "decoder_model_merged*.onnx")))
    if encoders and decoders:
        return encoders[0], decoders[0]
    return None, None


def main() -> int:
    if len(sys.argv) < 4:
        print(
            "usage: transcribe_onnx.py <model_dir> <audio_path> <device>",
            file=sys.stderr,
        )
        return 2
    model_dir, audio_path, device = sys.argv[1], sys.argv[2], sys.argv[3]

    print(json.dumps({"stage": "loading"}), flush=True)
    try:
        import numpy as np
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor, WhisperTokenizer
    except ImportError as exc:
        print(json.dumps({"error": f"missing dependency: {exc.name}"}))
        return 1

    try:
        encoder_path, decoder_path = find_graphs(model_dir)
        if not encoder_path or not decoder_path:
            print(
                json.dumps(
                    {
                        "error": (
                            "no ONNX Whisper graphs found (want onnx/encoder_model*.onnx "
                            "and onnx/decoder_model_merged*.onnx)"
                        )
                    }
                )
            )
            return 1

        providers = pick_providers(device)
        enc_sess = ort.InferenceSession(encoder_path, providers=providers)
        dec_sess = ort.InferenceSession(decoder_path, providers=providers)
        landed = provider_label(enc_sess.get_providers())
        # Read before any model work so the resource row is right even when the
        # tokenizer or the audio later fails.
        print(json.dumps({"device": landed}), flush=True)

        feature_extractor = WhisperFeatureExtractor.from_pretrained(model_dir)
        tokenizer = WhisperTokenizer.from_pretrained(model_dir)
        with open(os.path.join(model_dir, "generation_config.json"), "r", encoding="utf-8") as fh:
            generation = json.load(fh)
        start = int(generation.get("decoder_start_token_id", 50257))
        eos = int(generation.get("eos_token_id", 50256))
        forced = []
        for pair in generation.get("forced_decoder_ids") or []:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                forced.append(int(pair[1]))
        max_length = int(generation.get("max_length", 448))
        suppress = {int(t) for t in generation.get("suppress_tokens") or [] if isinstance(t, int)}
        begin_suppress = {
            int(t) for t in generation.get("begin_suppress_tokens") or [] if isinstance(t, int)
        }

        # The no-cache branch still requires the past inputs to exist. Build
        # them once, with zero-length sequence axes; the graph's model-past
        # branch ignores them, and rebuilding them per token would be waste.
        past = {}
        for inp in dec_sess.get_inputs():
            if not inp.name.startswith("past_key_values."):
                continue
            dims = inp.shape
            heads = dims[1] if len(dims) > 1 and isinstance(dims[1], int) else 6
            head_dim = dims[3] if len(dims) > 3 and isinstance(dims[3], int) else 64
            past[inp.name] = np.zeros((1, heads, 0, head_dim), dtype=np.float32)

        def decode_window(audio):
            features = feature_extractor(
                audio, sampling_rate=SAMPLE_RATE, return_tensors="np"
            ).input_features.astype(np.float32)
            encoded = enc_sess.run(None, {"input_features": features})[0]
            ids = [start, *forced]
            generated = []
            max_new = max(1, max_length - len(ids))
            for step in range(max_new):
                feeds = {
                    "input_ids": np.array([ids], dtype=np.int64),
                    "encoder_hidden_states": encoded,
                    "use_cache_branch": np.array([False], dtype=bool),
                    **past,
                }
                outputs = dec_sess.run(None, feeds)
                logits = outputs[0][0, -1].astype(np.float32)
                if step == 0:
                    for token in begin_suppress:
                        if 0 <= token < logits.shape[0]:
                            logits[token] = -np.inf
                for token in suppress:
                    if 0 <= token < logits.shape[0]:
                        logits[token] = -np.inf
                nxt = int(np.argmax(logits))
                if nxt == eos:
                    break
                generated.append(nxt)
                ids.append(nxt)
            return tokenizer.decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ).strip()

        print(json.dumps({"stage": "transcribing"}), flush=True)
        audio = decode_audio(audio_path)
        window = int(CHUNK_SECONDS * SAMPLE_RATE)
        chunks = []
        text_parts = []
        for offset in range(0, audio.size, window):
            part = audio[offset : offset + window]
            if part.size < SAMPLE_RATE // 10:
                continue
            text = decode_window(part)
            start_s = round(offset / SAMPLE_RATE, 2)
            end_s = round(min(audio.size, offset + part.size) / SAMPLE_RATE, 2)
            chunks.append({"start": start_s, "end": end_s, "text": text})
            if text:
                text_parts.append(text)
        print(
            json.dumps(
                {
                    "text": " ".join(text_parts).strip(),
                    "chunks": chunks,
                    "device": landed,
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
