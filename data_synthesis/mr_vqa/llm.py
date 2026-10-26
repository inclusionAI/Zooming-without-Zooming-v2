import json
import os
import random
import re
import sys
import threading
import time

import requests
from openai import OpenAI

# --- Rate-limit-aware backoff helpers ---
#
# Burst failures were concentrated at the peak of each concurrent wave: dozens
# of in-flight requests hit the gateway's per-key limit at once, then retried
# with identical fixed sleeps (`time.sleep(0.5*(attempt+1))`, cap 2s) and no
# jitter, so they re-fired in a synchronized herd, hit the limit again, and
# burned the (small) retry budget within the limit window. Result: every burst
# collapsed into a cluster of `call_chat exhausted` failures, with near-zero
# model traffic between bursts.
#
# Fixes below:
#   - exponential backoff capped at RETRY_BACKOFF_CAP, with full-jitter so a
#     synchronized herd spreads out across the retry window instead of
#     re-colliding.
#   - honor `Retry-After` (seconds or HTTP-date) when the gateway advertises
#     one — that is the authoritative wait before the key recovers.
#   - a separate, larger retry budget on the 429/5xx path so transient rate
#     limiting can ride out the window instead of giving up after ~6s.
RETRY_BACKOFF_CAP = 20.0
RATE_LIMIT_RETRIES = 8
_rng = random.Random()  # process-local; seeded lazily by the interpreter


def _parse_retry_after(value):
    """Parse a Retry-After header value into a float number of seconds.

    Accepts either a delta-seconds integer/float or an HTTP-date. Returns None
    if it can't be parsed (caller falls back to computed backoff). Caps the
    returned wait at RETRY_BACKOFF_CAP so a misbehaving gateway can't stall a
    worker for minutes.
    """
    if not value:
        return None
    try:
        return min(RETRY_BACKOFF_CAP, max(0.0, float(value)))
    except (TypeError, ValueError):
        pass
    # HTTP-date form (RFC 7231). Use email.utils.parsedate_to_datetime for
    # robustness; fall through to None on any malformed date.
    try:
        from email.utils import parsedate_to_datetime
        from datetime import datetime, timezone
        dt = parsedate_to_datetime(value)
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = (dt - datetime.now(timezone.utc)).total_seconds()
        return min(RETRY_BACKOFF_CAP, max(0.0, delta))
    except Exception:
        return None


def _jittered_backoff(attempt, cap=RETRY_BACKOFF_CAP, base=0.4):
    """Full-jitter exponential backoff: uniform in [0, min(cap, base*2**attempt)).

    Full jitter (not decorrelated/equal) spreads a synchronized herd widest for
    the smallest worst-case wait, which is exactly what a burst of 429s needs.
    """
    upper = min(cap, base * (2 ** attempt))
    return _rng.uniform(0.0, upper)


def extract_json(text, expected_type=list):
    if not text:
        return expected_type()
    block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if block:
        text = block.group(1).strip()
    else:
        start = text.find("[") if expected_type is list else text.find("{")
        end = text.rfind("]") if expected_type is list else text.rfind("}")
        if start >= 0 and end >= 0:
            text = text[start:end + 1]
    try:
        data = json.loads(text)
        return data if isinstance(data, expected_type) else expected_type()
    except Exception:
        return expected_type()


class APIKeyPool:
    def __init__(self, keys, max_concurrent_per_key=8,
                 min_request_interval=None):
        # Preserve configured order while preventing an accidentally repeated
        # key from masquerading as additional capacity.
        self.keys = list(dict.fromkeys(k.strip() for k in keys if k.strip()))
        self.max_concurrent_per_key = max(1, int(max_concurrent_per_key))
        if min_request_interval is None:
            min_request_interval = os.environ.get(
                "API_KEY_MIN_REQUEST_INTERVAL_SECONDS", "0.5",
            )
        self.min_request_interval = max(0.0, float(min_request_interval))
        self.active = {key: 0 for key in self.keys}
        self.errors = {key: 0 for key in self.keys}
        self.next_allowed = {key: 0.0 for key in self.keys}
        self.disabled = set()
        self.condition = threading.Condition()

    def acquire(self, timeout=None):
        """Lease one key, applying capacity and per-key dispatch pacing.

        ``timeout=None`` is intentional backpressure: callers without a
        higher-level deadline wait until a key is released instead of turning
        a busy local queue into a model-call failure.  A finite timeout is
        reserved for callers that already have an end-to-end deadline.
        """
        deadline = None
        if timeout is not None:
            timeout = max(0.0, float(timeout))
            deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                now = time.monotonic()
                enabled = [
                    key for key in self.keys if key not in self.disabled
                ]
                if not enabled:
                    raise RuntimeError("No enabled API key available")
                capacity = [
                    key for key in enabled
                    if self.active[key] < self.max_concurrent_per_key
                ]
                ready = [
                    key for key in capacity
                    if self.next_allowed[key] <= now
                ]
                if ready:
                    key = min(ready, key=lambda value: (
                        self.errors[value] * 2 + self.active[value],
                        self.next_allowed[value],
                        value,
                    ))
                    self.active[key] += 1
                    # The interval is measured between request dispatch leases,
                    # not completions. All episode threads share this clock.
                    self.next_allowed[key] = now + self.min_request_interval
                    return key
                remaining = None if deadline is None else deadline - now
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("No API key available before deadline")
                wait_seconds = remaining
                if capacity:
                    pacing_wait = max(
                        0.001,
                        min(self.next_allowed[key] for key in capacity) - now,
                    )
                    wait_seconds = (
                        pacing_wait if remaining is None
                        else min(remaining, pacing_wait)
                    )
                self.condition.wait(timeout=wait_seconds)

    def release(self, key, has_error=False):
        with self.condition:
            if key not in self.active:
                return
            self.active[key] = max(0, self.active[key] - 1)
            if has_error:
                self.errors[key] += 1
            self.condition.notify_all()

    def disable(self, key):
        """Permanently remove an unauthorized key for this process."""
        with self.condition:
            if key not in self.active:
                return
            self.disabled.add(key)
            self.active[key] = max(0, self.active[key] - 1)
            self.errors[key] += 1000
            self.condition.notify_all()


_KEY_POOLS = {}
_KEY_POOLS_LOCK = threading.Lock()


def _get_key_pool(api_key, max_concurrent_per_key):
    keys = list(dict.fromkeys(
        k.strip() for k in str(api_key).split(",") if k.strip()
    ))
    if not keys:
        return None
    min_interval = max(0.0, float(os.environ.get(
        "API_KEY_MIN_REQUEST_INTERVAL_SECONDS", "0.5",
    )))
    cache_key = (tuple(keys), int(max_concurrent_per_key), min_interval)
    with _KEY_POOLS_LOCK:
        if cache_key not in _KEY_POOLS:
            _KEY_POOLS[cache_key] = APIKeyPool(
                keys,
                max_concurrent_per_key=max_concurrent_per_key,
                min_request_interval=min_interval,
            )
        return _KEY_POOLS[cache_key]


def call_chat(api_url, api_key, model, messages, temperature=0.3, top_p=None, max_tokens=2048,
              retries=4, request_timeout=180, max_concurrent_per_key=12,
              enable_thinking=False, rate_limit_retries=RATE_LIMIT_RETRIES,
              deadline_monotonic=0.0, network_retries=None):
    # Accept OpenAI-style base URLs (e.g. https://host/v1) and auto-append the
    # chat completions path so callers can pass either form.
    url = api_url
    if not url.rstrip("/").endswith("/chat/completions"):
        url = url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
    }
    if top_p is not None:
        payload["top_p"] = top_p
    key_pool = _get_key_pool(api_key, max_concurrent_per_key)
    consecutive_empty = 0
    last_error = "unknown_error"
    # A rate-limit attempt is retryable up to rate_limit_retries times; bad-key
    # and non-retryable attempts use the caller's `retries` budget. We drive the
    # loop with rl_attempt so a long 429 tail can survive the limit window
    # without forcing `retries` up on unrelated failures.
    rl_attempt = 0
    network_attempt = 0
    other_attempt = 0
    # Total tries is bounded by the sum of both budgets; in practice the first
    # non-rate-limit error that isn't a 5xx drains `other_attempt` quickly.
    max_other = max(1, retries)
    max_rl = max(1, rate_limit_retries)
    max_network = (
        max_rl if network_retries is None
        else max(0, int(network_retries))
    )
    deadline_monotonic = float(deadline_monotonic or 0.0)
    deadline_exhausted = False

    def remaining_seconds():
        if deadline_monotonic <= 0:
            return None
        return deadline_monotonic - time.monotonic()

    def bounded_sleep(seconds):
        nonlocal deadline_exhausted
        remaining = remaining_seconds()
        if remaining is not None:
            if remaining <= 0:
                deadline_exhausted = True
                return False
            seconds = min(float(seconds), remaining)
        time.sleep(max(0.0, float(seconds)))
        return True

    while True:
        remaining = remaining_seconds()
        if remaining is not None and remaining <= 0:
            deadline_exhausted = True
            break
        current_key = None
        try:
            current_key = (
                key_pool.acquire(timeout=max(0.0, remaining))
                if key_pool and remaining is not None
                else key_pool.acquire(timeout=None) if key_pool
                else str(api_key).split(",")[0].strip()
            )
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {current_key}"}
            request_remaining = remaining_seconds()
            if request_remaining is not None and request_remaining <= 0:
                deadline_exhausted = True
                if key_pool and current_key:
                    key_pool.release(current_key, has_error=False)
                    current_key = None
                break
            bounded_request_timeout = min(
                float(request_timeout), request_remaining,
            ) if request_remaining is not None else float(request_timeout)
            resp = requests.post(
                url, headers=headers, json=payload,
                timeout=(
                    min(15.0, max(0.05, bounded_request_timeout)),
                    max(0.05, bounded_request_timeout),
                ),
            )
            if resp.status_code in (401, 403):
                last_error = f"http_{resp.status_code}"
                if key_pool:
                    key_pool.disable(current_key)
                # Authentication failures are permanent for the current key.
                # A pooled request can retry immediately on another key.
                if not key_pool:
                    break
                continue
            if resp.status_code in (429, 500, 502, 503, 504):
                last_error = f"http_{resp.status_code}"
                if key_pool:
                    key_pool.release(current_key, has_error=True)
                rl_attempt += 1
                if rl_attempt > max_rl:
                    break
                # Prefer the gateway's declared wait; otherwise jittered exp backoff.
                # Full jitter spreads a synchronized herd so retries don't re-collide.
                ra = _parse_retry_after(resp.headers.get("Retry-After") or resp.headers.get("retry-after"))
                sleep_s = ra if ra is not None else _jittered_backoff(rl_attempt - 1)
                if not bounded_sleep(sleep_s):
                    break
                continue
            if resp.status_code != 200:
                last_error = f"http_{resp.status_code}"
                if key_pool:
                    key_pool.release(current_key, has_error=True)
                other_attempt += 1
                if other_attempt > max_other:
                    break
                if not bounded_sleep(
                    _jittered_backoff(other_attempt - 1, cap=2.0)
                ):
                    break
                continue
            body = resp.json()
            content = body["choices"][0]["message"].get("content")
            if not content:
                last_error = "empty_completion"
                if key_pool:
                    key_pool.release(current_key, has_error=True)
                consecutive_empty += 1
                # HTTP 200 with zero completion tokens usually means a broken
                # model route, not transient rate limiting. Avoid multiplying
                # one bad request into an hour-long 80-retry tail.
                if consecutive_empty >= 3:
                    return ""
                if not bounded_sleep(0.3):
                    break
                continue
            consecutive_empty = 0
            if key_pool:
                key_pool.release(current_key, has_error=False)
            return content
        except Exception as exc:
            last_error = type(exc).__name__
            if key_pool and current_key:
                key_pool.release(current_key, has_error=True)
            # A pool wait can only time out when the caller supplied an
            # end-to-end candidate deadline.  It is queue backpressure, not a
            # failed API request, so do not burn the model retry budget.
            if (
                isinstance(exc, TimeoutError)
                and key_pool
                and current_key is None
                and deadline_monotonic > 0
            ):
                deadline_exhausted = True
                last_error = "api_key_wait_deadline"
                break
            # Network timeouts have an independent budget. Bulk jobs can set
            # it to zero and move to another image without weakening the retry
            # policy for explicit HTTP 429/5xx responses.
            name = type(exc).__name__
            is_transient = ("Timeout" in name or "Connection" in name
                            or "ReadTimeout" in name or "TimeoutError" in name)
            if is_transient:
                network_attempt += 1
                if network_attempt > max_network:
                    break
                if not bounded_sleep(
                    _jittered_backoff(network_attempt - 1)
                ):
                    break
            else:
                other_attempt += 1
                if other_attempt > max_other:
                    break
                if not bounded_sleep(
                    _jittered_backoff(other_attempt - 1, cap=2.0)
                ):
                    break
    if deadline_exhausted:
        raise TimeoutError(
            f"model call exceeded candidate deadline: model={model}"
        )
    print(
        f"call_chat exhausted: model={model} "
        f"rl_attempts={rl_attempt}/{max_rl} "
        f"network_attempts={network_attempt}/{max_network} "
        f"other_attempts={other_attempt}/{max_other} "
        f"error={last_error}",
        file=sys.stderr,
    )
    return ""


def image_message(image_b64, mime="image/jpeg"):
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}}


class JudgeClient:
    def __init__(self, api_key, api_url, model):
        self.client = OpenAI(api_key=api_key, base_url=api_url, timeout=180)
        self.model = model

    def ask(self, prompt, temperature=0.0, max_tokens=1024, retries=4):
        last_exc = None
        for attempt in range(retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                return completion.choices[0].message.content
            except Exception as exc:
                last_exc = exc
                # backoff for 429 / rate limit / transient errors
                msg = str(exc).lower()
                if "429" in msg or "rate" in msg or "timeout" in msg or "502" in msg or "503" in msg or "504" in msg:
                    time.sleep(min(8.0, 0.5 * (attempt + 1) * 2))
                    continue
                # other errors: brief sleep and retry a few times
                if attempt < 3:
                    time.sleep(0.5)
                    continue
                raise
        raise last_exc if last_exc else RuntimeError("JudgeClient retries exhausted")
