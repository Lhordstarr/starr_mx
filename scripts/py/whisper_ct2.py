"""Transcribes audio with a CTranslate2 Whisper model — no PyTorch required.

Invoked by the server as:
    python scripts/py/whisper_ct2.py <model_dir> <audio_path> <device>

Prints one JSON line on stdout; progress and logging go to stderr.

Why this exists alongside ``transcribe.py``:

* ``transcribe.py`` drives the weights through transformers, which means
  PyTorch — 529 MB before a single model is downloaded. CTranslate2 does the
  same job in roughly 40 MB and runs the whole Whisper family on CPU.
* It is the only STT path that installs on a machine whose only interpreter is
  Python 3.14, because every Kokoro/transformers TTS package caps below that.
* PyAV is deliberately not used. ``faster_whisper.audio.decode_audio`` calls
  ``av.open(metadata_errors=...)``, an argument PyAV removed in v15, and no
  PyAV release with cp314 wheels predates the removal. Decoding with the
  standard library sidesteps the ABI split entirely, and a browser recording
  arrives as WebM/Opus anyway — see ``decode_audio`` below.
"""
from __future__ import annotations

import array
import json
import os
import sys
import wave

TARGET_RATE = 16_000


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _site_packages_dirs() -> list[str]:
    """The interpreter's package directories, without importing anything big.

    This runs before CTranslate2 is imported: the loader reads
    ``LD_LIBRARY_PATH`` once at process start, so a Python assignment made
    later cannot help a ``dlopen``. Discovering the directories first lets
    ``ensure_cuda_libraries`` re-exec with the right environment instead.
    """
    import sysconfig

    found: list[str] = []
    try:
        paths = sysconfig.get_paths()
        for key in ("purelib", "platlib"):
            value = paths.get(key)
            if value:
                found.append(value)
    except Exception:
        pass
    # ``sys.path`` also sees a venv's site-packages when the interpreter was
    # started through a wrapper that does not report it in sysconfig. Checking
    # it is cheap, and it gives the tests a way to supply a stand-in tree.
    for entry in sys.path:
        if entry and ("site-packages" in entry or "dist-packages" in entry):
            found.append(entry)

    unique: list[str] = []
    for entry in found:
        full = os.path.abspath(entry)
        if full not in unique and os.path.isdir(full):
            unique.append(full)
    return unique


def _has_any(dirname: str, names: tuple[str, ...]) -> bool:
    return any(os.path.exists(os.path.join(dirname, name)) for name in names)


def _cuda_library_dirs() -> list[str]:
    """Directories that contain the CUDA libraries CTranslate2 dlopens.

    The PyPI CTranslate2 wheel is CUDA-enabled but does not ship cuBLAS or
    cuDNN. Those arrive with PyTorch (``torch/lib``) or with the ``nvidia-*``
    wheels. They were present on the GPU box and still not found: CTranslate2
    enumerated the card through libcudart, then failed when it tried to load
    cuBLAS. Keeping the directories explicit here is the difference between a
    working GPU job and an error that names a library the user can see on disk.
    """
    if os.name == "nt":
        cublas = ("cublas64_12.dll",)
        cudnn = ("cudnn64_9.dll",)
    else:
        cublas = ("libcublas.so.12",)
        cudnn = ("libcudnn.so.9",)

    dirs: list[str] = []
    for packages in _site_packages_dirs():
        torch_lib = os.path.join(packages, "torch", "lib")
        if _has_any(torch_lib, cublas + cudnn):
            dirs.append(torch_lib)
        for package, names in (("cublas", cublas), ("cudnn", cudnn)):
            lib = os.path.join(packages, "nvidia", package, "lib")
            if _has_any(lib, names):
                dirs.append(lib)
    return dirs


def ensure_cuda_libraries(device_request: str) -> None:
    """Put the venv's CUDA libraries on the loader path, once.

    A CUDA model request on a machine that has torch but no system CUDA is the
    normal case on this app: torch carries cuBLAS/cuDNN inside its own wheel,
    and CTranslate2 does not know to look there. ``os.execve`` is deliberate —
    glibc has already read ``LD_LIBRARY_PATH`` by the time Python code runs, so
    changing the variable in place would be too late for the next ``dlopen``.
    The re-exec keeps the same pid and stdio, so the server's job tracking does
    not notice.
    """
    if os.name != "posix":
        return
    if (device_request or "cpu").lower() in ("cpu", "rocm", "amd"):
        return

    dirs = _cuda_library_dirs()
    if not dirs:
        return

    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [part for part in current.split(os.pathsep) if part]
    have = {os.path.normcase(os.path.abspath(part)) for part in parts}
    missing = [d for d in dirs if os.path.normcase(os.path.abspath(d)) not in have]
    if not missing:
        return

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.pathsep.join(missing + parts)
    log(f"adding CUDA libraries from {', '.join(missing)}")
    try:
        os.execve(
            sys.executable,
            [sys.executable, os.path.abspath(__file__), *sys.argv[1:]],
            env,
        )
    except OSError as exc:
        # Not fatal on its own: the model load below will fail in a way the
        # caller can see, and the CPU retry still gets a chance to work.
        log(f"could not re-exec with the CUDA library path: {exc}")


def _resample(samples, src_rate: int):
    """Linear resampling to 16 kHz. Good enough for speech features, and it
    avoids depending on scipy just to change a sample rate."""
    import numpy as np

    if src_rate == TARGET_RATE or src_rate <= 0:
        return np.asarray(samples, dtype=np.float32)
    n_out = max(1, int(len(samples) * TARGET_RATE / src_rate))
    x_old = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, np.asarray(samples, dtype=np.float32)).astype(np.float32)


def decode_audio(path: str):
    """Read a WAV file to a float32 numpy array at 16 kHz.

    Returns None when the file is not something the standard library can read
    (a browser recording is usually WebM/Opus). The caller then decodes it with
    PyAV if that is importable, and reports a clear error if not.

    The result must be a numpy array, not a list: ``WhisperModel.transcribe``
    only skips its own PyAV decode when it is handed an ndarray, and a plain
    list is treated as a path-like and sent straight back to ``decode_audio``,
    which is precisely the call that breaks on PyAV 15+.
    """
    try:
        with wave.open(path, "rb") as w:
            channels = w.getnchannels()
            width = w.getsampwidth()
            rate = w.getframerate()
            raw = w.readframes(w.getnframes())
    except (wave.Error, EOFError, OSError):
        return None

    if width == 2:
        data = array.array("h")
        data.frombytes(raw[: len(raw) - (len(raw) % 2)])
        samples = [s / 32768.0 for s in data]
    elif width == 1:
        data = array.array("B")
        data.frombytes(raw)
        samples = [(b - 128) / 128.0 for b in data]
    elif width == 4:
        data = array.array("i")
        data.frombytes(raw[: len(raw) - (len(raw) % 4)])
        samples = [s / 2147483648.0 for s in data]
    else:
        return None

    if channels > 1:
        # Downmix by averaging channels.
        samples = [
            sum(samples[i : i + channels]) / channels
            for i in range(0, len(samples) - channels + 1, channels)
        ]

    return _resample(samples, rate)


def decode_with_pyav(path: str):
    """Fallback for compressed formats. PyAV is imported lazily so that its
    absence is an ordinary error message rather than an import-time crash."""
    try:
        import av  # noqa: F401
        import numpy as np
    except Exception as exc:  # pragma: no cover - depends on the install
        raise RuntimeError(
            "This recording is not a WAV file and PyAV is not available to "
            "decode it. Record as WAV, or install the ffmpeg-backed decoder."
        ) from exc

    import faster_whisper.audio as fwa

    try:
        pcm = fwa.decode_audio(path, sampling_rate=TARGET_RATE)
    except TypeError:
        # The av>=15 keyword incompatibility faster-whisper 1.2.x still has.
        # Decoding by hand avoids it.
        try:
            container = av.open(path, mode="r")
        except Exception as exc:
            raise RuntimeError(f"Could not open {os.path.basename(path)}: {exc}") from exc
        chunks = []
        # The sample rate has to be read while the container is still open.
        # Reading it after close silently yielded nothing, so no resampling
        # happened and a 48 kHz Opus recording reached Whisper as 48 kHz —
        # stretched threefold, which turned an 11-second sentence into 33
        # seconds of noise and garbage output.
        src_rate = 0
        try:
            if container.streams.audio:
                src_rate = int(container.streams.audio[0].rate) or 0
            for frame in container.decode(audio=0):
                arr = frame.to_ndarray().reshape(-1)
                # PyAV hands back float32 already normalised to [-1, 1] for
                # Opus and AAC. Scaling that by 32768 — the right conversion
                # for int16 — left an RMS of 4e-6, which Whisper reads as
                # silence, so every browser recording came back empty.
                if arr.dtype.kind == 'f':
                    chunks.append(arr.astype("float32"))
                else:
                    chunks.append(arr.astype("float32") / 32768.0)
        finally:
            container.close()
        if not chunks:
            return np.zeros(0, dtype="float32")
        pcm = np.concatenate(chunks)
        if src_rate and src_rate != TARGET_RATE:
            pcm = _resample(pcm.tolist(), src_rate)
    return pcm


def pick_device(requested: str) -> tuple[str, int]:
    """Resolve a device request to (device, index) for CTranslate2.

    CTranslate2 has no ROCm backend, so an AMD request degrades to CPU with a
    note on stderr rather than pretending the card was used.
    """
    want = (requested or "cpu").lower()
    try:
        import ctranslate2
    except Exception:
        return "cpu", 0

    try:
        have_cuda = ctranslate2.get_cuda_device_count() > 0
    except Exception as exc:
        # Enumeration can itself fail on a wheel with no usable CUDA runtime.
        # That is a CPU answer, not a traceback before the runner can report
        # what happened.
        log(f"could not enumerate CUDA devices: {exc}")
        have_cuda = False
    if want == "cpu":
        return "cpu", 0
    if want in ("rocm", "amd"):
        if have_cuda:
            log("rocm requested but CTranslate2 has no ROCm backend; using CUDA if present")
        else:
            log("rocm requested but CTranslate2 cannot drive an AMD GPU; running on CPU")
        return "cpu", 0
    if want in ("auto", "gpu", "cuda", "nvidia", "both"):
        if have_cuda:
            return "cuda", 0
        log("no CUDA device visible; running on CPU")
    return "cpu", 0


def main() -> int:
    if len(sys.argv) < 3:
        print(json.dumps({"error": "usage: whisper_ct2.py <model_dir> <audio> [device]"}))
        return 2
    model_dir = sys.argv[1]
    audio_path = sys.argv[2]
    device_request = sys.argv[3] if len(sys.argv) > 3 else "cpu"

    if not os.path.isdir(model_dir):
        print(json.dumps({"error": f"model directory not found: {model_dir}"}))
        return 1

    # Must happen before importing CTranslate2: its first CUDA call is what
    # dlopens libcublas, and the loader has already fixed its search path.
    ensure_cuda_libraries(device_request)

    try:
        import ctranslate2
        from faster_whisper import WhisperModel
    except Exception as exc:
        print(json.dumps({"error": f"missing dependency: {exc}"}))
        return 1

    device, index = pick_device(device_request)
    if device == "cpu":
        compute = "int8"
    else:
        try:
            supported = ctranslate2.get_supported_compute_types("cuda")
        except Exception as exc:
            log(f"could not list CUDA compute types: {exc}")
            supported = set()
        compute = "float16" if "float16" in supported else "float32"

    def load(target_device: str, target_index: int, target_compute: str):
        return WhisperModel(
            model_dir,
            device=target_device,
            device_index=target_index or 0,
            compute_type=target_compute,
        )

    log(f"loading {model_dir} on {device} ({compute})")
    try:
        model = load(device, index, compute)
    except Exception as exc:
        if device != "cuda":
            print(json.dumps({"error": f"could not load the model: {exc}"}))
            return 1
        # The wheel can enumerate the card and still fail to load cuBLAS or
        # cuDNN. That is a missing library, not a reason to lose the
        # transcription: CPU int8 is slower but it works, and the device line
        # below will say which one actually ran.
        log(f"CUDA could not load the model: {exc}")
        log("retrying on CPU (int8)")
        device, index, compute = "cpu", 0, "int8"
        try:
            model = load(device, index, compute)
        except Exception as cpu_exc:
            print(json.dumps({"error": f"could not load the model: {cpu_exc}"}))
            return 1

    # The server reads this out of the running job's output; the request alone
    # cannot say where a job ended up. Printed after the load so a CUDA failure
    # that fell back to the CPU reports the truth.
    print(json.dumps({"device": device}), flush=True)

    samples = decode_audio(audio_path)
    if samples is None:
        try:
            samples = decode_with_pyav(audio_path)
        except Exception as exc:
            print(json.dumps({"error": str(exc)}))
            return 1

    def run_transcription():
        segments, info = model.transcribe(samples, beam_size=1)
        return list(segments), info

    log(f"transcribing {len(samples) / TARGET_RATE:.1f}s")
    try:
        segments, info = run_transcription()
    except Exception as exc:
        if device != "cuda":
            print(json.dumps({"error": f"transcription failed: {exc}"}))
            return 1
        # Some CTranslate2 builds defer the CUDA library load until the first
        # inference. A model that loaded is not proof the backend can run.
        log(f"CUDA could not transcribe: {exc}")
        log("retrying on CPU (int8)")
        device, index, compute = "cpu", 0, "int8"
        try:
            model = load(device, index, compute)
            print(json.dumps({"device": device}), flush=True)
            segments, info = run_transcription()
        except Exception as cpu_exc:
            print(json.dumps({"error": f"transcription failed: {cpu_exc}"}))
            return 1

    text = "".join(s.text for s in segments).strip()
    print(
        json.dumps(
            {
                "text": text,
                "language": getattr(info, "language", None),
                "languageProbability": getattr(info, "language_probability", None),
                "duration": getattr(info, "duration", None),
                "segments": [
                    {"start": s.start, "end": s.end, "text": s.text} for s in segments
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())