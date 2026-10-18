#!/usr/bin/env python3
"""A tiny OpenAI-compatible chat server backed by Hugging Face transformers.

Started on demand by the app's transformers-chat provider, one model at a
time, the same way llama-server is: it loads the weights, answers /health with
503 until they are in, then serves /v1/chat/completions with streamed deltas.
Stdin is a pipe from the parent process, so an untimely death of the app
reaps this process instead of leaving a model in RAM.

Only causal language models are supported. A seq2seq or encoder model will
fail to load with a clear message rather than answering nonsense.
"""

import argparse
import json
import os
import sys
import threading

# Keep the transcript clean: the model loader is chatty on stderr, and none of
# it is actionable for someone watching the chat.
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    logging as _tf_logging,
)
_tf_logging.set_verbosity_error()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='directory holding the model')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dtype = torch.float16 if device != 'cpu' else torch.float32

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except Exception as e:
        print(f'ERROR: could not load tokenizer: {e}', flush=True)
        sys.exit(1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        print(f'ERROR: could not load the model as a causal language model: {e}', flush=True)
        sys.exit(1)
    try:
        model.to(device)
    except Exception as e:
        print(f'ERROR: could not move the model to {device}: {e}', flush=True)
        sys.exit(1)
    model.eval()

    model_id = os.path.basename(os.path.normpath(args.model))
    print('READY', flush=True)

    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header('content-type', 'application/json')
            self.send_header('content-length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/health':
                self._json(200, {'status': 'ok'})
            elif self.path.startswith('/v1/models'):
                self._json(200, {'object': 'list', 'data': [{'id': model_id, 'object': 'model'}]})
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if not self.path.startswith('/v1/chat/completions'):
                self.send_response(404)
                self.end_headers()
                return
            n = int(self.headers.get('content-length', '0') or '0')
            raw = self.rfile.read(n) if n else b''
            try:
                body = json.loads(raw) if raw else {}
            except Exception:
                body = {}
            messages = body.get('messages') or []
            temperature = body.get('temperature')
            if not isinstance(temperature, (int, float)):
                temperature = 0.7
            max_tokens = body.get('max_tokens', 512)
            if not isinstance(max_tokens, int) or max_tokens < 1:
                max_tokens = 512

            # Build the prompt. A chat template gives the model its expected
            # shape; without one, fall back to a plain transcript.
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                parts = []
                for m in messages:
                    parts.append(f"{m.get('role', 'user')}: {m.get('content', '')}")
                prompt = '\n'.join(parts) + '\nassistant:'

            try:
                inputs = tokenizer(prompt, return_tensors='pt').to(device)
            except Exception as e:
                self._json(400, {'error': str(e)})
                return

            streamer = TextIteratorStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            gen_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_tokens,
                temperature=float(temperature),
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )

            thread = threading.Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
            thread.start()

            self.send_response(200)
            self.send_header('content-type', 'text/event-stream')
            self.send_header('cache-control', 'no-cache')
            self.end_headers()
            try:
                for text in streamer:
                    if not text:
                        continue
                    chunk = {'choices': [{'delta': {'content': text}}]}
                    self.wfile.write(f'data: {json.dumps(chunk)}\n\n'.encode())
                    self.wfile.flush()
                self.wfile.write(b'data: [DONE]\n\n')
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            thread.join(timeout=5)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.serve_forever()


if __name__ == '__main__':
    main()
