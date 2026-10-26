"""Qwen-VL multimodal client for the multi-region VQA pipeline.

Calls a Qwen vision-language model through an OpenAI-compatible
chat/completions endpoint with an inline base64 data-URL image. Downscaling is
intentionally avoided: the image is only normalized to RGB and re-encoded as
high-quality JPEG so the payload is stable. Explicit connect/read timeouts and
a retry-with-backoff loop handle transient gateway failures. Unlike a naive
chat path, this never raises KeyError on a malformed body — it raises a
descriptive RuntimeError after retries are exhausted.
"""
import base64
import io
import os
import time
from pathlib import Path

import httpx
from openai import OpenAI
from PIL import Image


DEFAULT_BASE_URL = "https://your-llm-gateway.example.com/v1"
DEFAULT_MODEL = "Qwen3.5-VL-72B"

# Preserve full image information: no downscaling. Only normalize pixel
# format to RGB and re-encode as JPEG at high quality so the data-URL is a
# stable, gateway-acceptable payload.
_JPEG_QUALITY = 95


def _encode_image_data_url(image_path):
    """Read `image_path` and return a full-resolution JPEG data-URL."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"image does not exist: {path}")
    try:
        img = Image.open(path)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=_JPEG_QUALITY)
        payload = buf.getvalue()
    except Exception:
        # If PIL cannot decode (rare), fall back to raw bytes — let the
        # gateway reject it rather than failing prematurely.
        payload = path.read_bytes()
    b64 = base64.b64encode(payload).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def call_qwen_vl(prompt, image_path, api_key=None, model=DEFAULT_MODEL,
                 base_url=DEFAULT_BASE_URL, temperature=0.0,
                 max_tokens=1024, timeout=180, retries=4):
    """Call a Qwen-VL model through an OpenAI-compatible chat/completions endpoint.

    Returns the assistant text. Raises RuntimeError if all retries fail.
    """
    key = api_key or os.environ.get("MATRIX_API_KEY")
    if not key:
        raise RuntimeError("MATRIX_API_KEY is not configured")
    data_url = _encode_image_data_url(image_path)

    if isinstance(timeout, httpx.Timeout):
        timeouts = timeout
    else:
        timeouts = httpx.Timeout(connect=15.0, read=float(timeout),
                                 write=60.0, pool=15.0)

    client = OpenAI(
        base_url=base_url,
        api_key=key,
        http_client=httpx.Client(timeout=timeouts),
        max_retries=0,
    )
    last_error = None
    try:
        for attempt in range(max(1, retries)):
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {"type": "text", "text": prompt},
                        ],
                    }],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                choices = getattr(completion, "choices", None) or []
                if not choices:
                    last_error = RuntimeError("model returned no choices")
                    time.sleep(min(4.0, 0.8 * (attempt + 1)))
                    continue
                message = getattr(choices[0], "message", None)
                content = getattr(message, "content", None) if message else None
                text = (content or "").strip()
                if text:
                    return text
                last_error = RuntimeError("model returned an empty content body")
            except Exception as exc:
                last_error = exc
            time.sleep(min(4.0, 0.8 * (attempt + 1)))
    finally:
        client.close()
    raise RuntimeError(f"Qwen-VL request failed after {max(1, retries)} attempts: {last_error}")
