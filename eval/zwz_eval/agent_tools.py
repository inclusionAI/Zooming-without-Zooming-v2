"""Explicit live checks for the preserved AgentVista free tools; no model calls."""
from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def response_works(name, response):
    if not isinstance(response, str) or not response.strip():
        return False
    if response.startswith(('Error:', 'No results', 'No image results')):
        return False
    if name == 'web_search':
        return 'URL: ' in response
    if name == 'image_search':
        return 'Image URL: ' in response
    # A challenge/error HTML page is not a successful page fetch.
    return 'Example Domain' in response


def probe():
    vendor = Path(__file__).parent / 'resources/vendor/agentvista'
    sys.path.insert(0, str(vendor))
    os.environ['AGENTVISTA_FREE_TOOLS'] = '1'
    from tools import get_tool
    requests = [
        ('web_search', {'query': 'Eiffel Tower official website', 'max_results': 1}),
        ('image_search', {'query': 'Eiffel Tower', 'max_results': 1}),
        ('visit', {'url': 'https://example.org', 'goal': 'Read the example page'}),
    ]

    def call(item):
        name, parameters = item
        start = time.monotonic()
        result = {'tool': name}
        try:
            response = get_tool(name)().call(parameters)
            result.update(working=response_works(name, response),
                          response_preview=str(response)[:350])
        except Exception as exc:
            # Do not print endpoint credentials/proxy URLs from exception text.
            result.update(working=False, error_type=type(exc).__name__)
        result['seconds'] = round(time.monotonic() - start, 3)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        checks = list(pool.map(call, requests))
    return {'checks_passed': all(item['working'] for item in checks),
            'model_calls': 0, 'dataset_downloads': 0, 'checks': checks}


def check_live_tools(timeout=90):
    """Use a subprocess so the vendored top-level ``tools`` module stays isolated."""
    try:
        process = subprocess.run([sys.executable, str(Path(__file__).resolve())],
                                 capture_output=True, text=True, timeout=timeout)
        # Native imports may print warnings before the single final JSON line.
        report = json.loads(process.stdout.splitlines()[-1])
        if process.returncode != 0 or not isinstance(report, dict) or 'checks_passed' not in report:
            raise ValueError('Probe did not return a valid report')
        return report
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as exc:
        return {'checks_passed': False, 'model_calls': 0, 'dataset_downloads': 0,
                'error_type': type(exc).__name__, 'checks': []}


if __name__ == '__main__':
    print(json.dumps(probe(), ensure_ascii=False))
