"""Small explicit OpenAI-compatible client. Credentials never enter persisted payloads."""
from __future__ import annotations

import itertools
import json
import os
import threading
import time
import urllib.error
import urllib.request


class APIError(RuntimeError):
    pass


class CompatibleAPI:
    def __init__(self, endpoint, timeout=600, retries=3):
        self.endpoint = endpoint
        self.timeout, self.retries = timeout, retries
        self.counter, self.lock = itertools.count(), threading.Lock()

    def _url(self):
        with self.lock:
            return self.endpoint['base_urls'][next(self.counter) % len(self.endpoint['base_urls'])].rstrip('/')

    def _request(self, url, payload=None):
        key = os.environ.get(self.endpoint.get('api_key_env', ''), 'EMPTY')
        body = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(url, body, {'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            # Avoid printing arbitrary server bodies (which may echo credentials).
            error = APIError(f'HTTP {exc.code} from model service')
            error.retryable = exc.code in {408, 409, 429} or exc.code >= 500
            exc.close()
            raise error from None
        except (OSError, ValueError) as exc:
            error = APIError(f'{type(exc).__name__} during model service request')
            error.retryable = True
            raise error from None

    def health(self):
        for base in self.endpoint['base_urls']:
            data = self._request(base.rstrip('/') + '/models')
            entries = data.get('data') if isinstance(data, dict) else None
            if not isinstance(entries, list) or any(not isinstance(row, dict) or not isinstance(row.get('id'), str) for row in entries):
                raise APIError('Malformed model-list response')
            names = {row['id'] for row in entries}
            if self.endpoint['model'] not in names:
                raise APIError(f"Requested served model is absent from /v1/models: {self.endpoint['model']}")
        return True

    def complete(self, messages, *, thinking, max_tokens, temperature=0, **extra):
        payload = {'model': self.endpoint['model'], 'messages': messages,
                   'temperature': temperature, 'max_tokens': max_tokens, **extra}
        if thinking is not None:
            payload['chat_template_kwargs'] = {'enable_thinking': thinking}
        last = None
        for attempt in range(self.retries):
            try:
                response = self._request(self._url() + '/chat/completions', payload)
                choice = first_choice(response)
                message = choice.get('message')
                if not isinstance(message, dict) or 'content' not in message:
                    raise APIError('Malformed completion response (missing message/content)')
                if not isinstance(message['content'], (str, type(None))):
                    raise APIError('Malformed completion response (content is not text)')
                return response
            except APIError as exc:
                last = exc
                if not getattr(exc, 'retryable', False) or attempt+1 == self.retries:
                    break
                time.sleep(min(2 ** attempt, 8))
        raise last

    def complete_prompt(self, prompt, *, max_tokens, temperature=0):
        """Raw completions for the preserved native-tokenizer judge prefix.

        Never replace this with chat completions: the reference explicitly
        renders the judge template and appends an empty think block.
        """
        if not isinstance(prompt, str):
            raise ValueError('Rendered completion prompt must be text')
        payload = {'model': self.endpoint['model'], 'prompt': prompt,
                   'temperature': temperature, 'max_tokens': max_tokens}
        last = None
        for attempt in range(self.retries):
            try:
                response = self._request(self._url() + '/completions', payload)
                choice = first_choice(response)
                if not isinstance(choice.get('text'), str):
                    raise APIError('Malformed raw completion response')
                return response
            except APIError as exc:
                last = exc
                if not getattr(exc, 'retryable', False) or attempt+1 == self.retries:
                    break
                time.sleep(min(2 ** attempt, 8))
        raise last


def first_choice(response):
    choices = response.get('choices') if isinstance(response, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise APIError('Malformed completion response (invalid choices)')
    return choices[0]
