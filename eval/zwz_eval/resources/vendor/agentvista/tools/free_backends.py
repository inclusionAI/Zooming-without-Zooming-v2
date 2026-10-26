"""Key-free, non-official AgentVista tool backends.

Activated only when ``AGENTVISTA_FREE_TOOLS=1``.  These classes intentionally
reuse the public tool names so the original agent prompt and function schemas
stay unchanged.  Their data sources are DuckDuckGo's public HTML/image index
and direct HTTP page fetches, not Serper/Jina; do not mix their scores with an
official AgentVista run.
"""

import html
import json
import os
import re
from urllib.parse import quote_plus, unquote

import requests

from tools.base import BaseTool
from tools.tool_registry import register_tool


_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 Chrome/124 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.8",
}


def _text(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value or "")
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def _params(value):
    if isinstance(value, str):
        return json.loads(value)
    return value or {}


@register_tool("web_search")
class FreeWebSearch(BaseTool):
    name = "web_search"
    description = "Search the public web and return titles, URLs, and snippets."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "default": 10},
        },
        "required": ["query"],
    }

    def call(self, params, **kwargs):
        params = _params(params)
        query = params.get("query", "").strip()
        count = max(1, min(int(params.get("max_results", 10)), 10))
        if not query:
            return "Error: No search query provided"
        try:
            page = requests.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query}, headers=_HEADERS, timeout=25,
            )
            page.raise_for_status()
        except requests.RequestException as exc:
            return f"Error: free web search failed: {type(exc).__name__}"
        pattern = re.compile(
            r'<a[^>]*class="result__a"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>'
            r'.*?(?:<a[^>]*class="result__snippet"[^>]*>|<div[^>]*class="result__snippet"[^>]*>)(?P<snippet>.*?)</(?:a|div)>',
            re.S,
        )
        rows = []
        for match in pattern.finditer(page.text):
            url = html.unescape(match.group("url"))
            if "uddg=" in url:
                url = unquote(url.split("uddg=", 1)[1].split("&", 1)[0])
            rows.append((url, _text(match.group("title")), _text(match.group("snippet"))))
            if len(rows) >= count:
                break
        if not rows:
            return f"No results found for query: {query!r}"
        return "\n\n".join(
            f"{i}. {title}\nURL: {url}\nSnippet: {snippet}"
            for i, (url, title, snippet) in enumerate(rows, 1)
        )


@register_tool("image_search")
class FreeImageSearch(BaseTool):
    name = "image_search"
    description = (
        "Search public image results from a text query. For reverse-image tasks, "
        "first visually describe the image and then make a distinctive text query."
    )
    parameters = {
        "type": "object",
        "properties": {
            "search_type": {"type": "string", "enum": ["text", "reverse"], "default": "text"},
            "query": {"type": "string"},
            "image_url": {"type": "string"},
            "max_results": {"type": "integer", "default": 5},
        },
        "required": [],
    }

    def call(self, params, **kwargs):
        params = _params(params)
        query = params.get("query", "").strip()
        if params.get("search_type") == "reverse" and not query:
            return (
                "Free backend does not upload private/local images to third-party reverse-search "
                "services. Inspect the supplied image and retry image_search with search_type='text' "
                "and a descriptive query."
            )
        if not query:
            return "Error: image_search requires a text query in free mode"
        count = max(1, min(int(params.get("max_results", 5)), 10))
        try:
            landing = requests.get(
                "https://duckduckgo.com/", params={"q": query, "iax": "images", "ia": "images"},
                headers=_HEADERS, timeout=25,
            )
            landing.raise_for_status()
            token = re.search(r"vqd=['\"]?([\d-]+)", landing.text)
            if not token:
                return "Error: free image search did not return a query token"
            result = requests.get(
                "https://duckduckgo.com/i.js",
                params={"l": "us-en", "o": "json", "q": query, "vqd": token.group(1), "f": ",,,,,", "p": "1"},
                headers={**_HEADERS, "Referer": "https://duckduckgo.com/"}, timeout=25,
            )
            result.raise_for_status()
            items = result.json().get("results", [])[:count]
        except (requests.RequestException, ValueError) as exc:
            # DuckDuckGo's image JSON is intermittently blocked.  Bing's public
            # result page supplies the same essential fields without a key.
            try:
                fallback = requests.get(
                    "https://www.bing.com/images/search",
                    params={"q": query, "form": "HDRSC2"}, headers=_HEADERS, timeout=25,
                )
                fallback.raise_for_status()
                raw_items = re.findall(r'murl(?:&quot;|\\")\s*:\s*(?:&quot;|\\")([^&\\"]+)', fallback.text)
                items = [{"image": html.unescape(url), "title": "", "url": ""} for url in raw_items[:count]]
            except requests.RequestException:
                return f"Error: free image search failed: {type(exc).__name__}"
        if not items:
            return f"No image results found for query: {query!r}"
        return "\n\n".join(
            f"{i}. {item.get('title', '')}\nImage URL: {item.get('image', '')}\nSource: {item.get('url', '')}"
            for i, item in enumerate(items, 1)
        )


@register_tool("visit")
class FreeVisit(BaseTool):
    name = "visit"
    description = "Fetch a webpage and return its readable textual content."
    parameters = {
        "type": "object",
        "properties": {"url": {"type": "string"}, "goal": {"type": "string"}},
        "required": ["url", "goal"],
    }

    def call(self, params, **kwargs):
        params = _params(params)
        url = params.get("url", "")
        if not url.startswith(("http://", "https://")):
            return "Error: URL must begin with http:// or https://"
        try:
            response = requests.get(url, headers=_HEADERS, timeout=30)
            response.raise_for_status()
            content = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", response.text, flags=re.I | re.S)
            content = _text(content)
        except requests.RequestException:
            # Public Jina Reader endpoint: no account/key is needed for modest
            # interactive use.  It is only a fallback for sites that block
            # direct fetches; this differs from the authenticated official mode.
            try:
                reader = requests.get("https://r.jina.ai/http://" + url.split("://", 1)[1], headers=_HEADERS, timeout=45)
                reader.raise_for_status()
                content = reader.text.strip()
            except requests.RequestException as exc:
                return f"Error: free visit failed: {type(exc).__name__}"
        if not content:
            return "Error: no readable text extracted from page"
        # Keep free-mode page extraction compact enough to leave room for the
        # image tokens and subsequent tool turns in a 128k local vLLM context.
        # The official authenticated backend is unaffected.
        limit = int(os.environ.get(
            "AGENTVISTA_FREE_VISIT_CHARS",
            (self.config or {}).get("max_content_length", 8000),
        ))
        return content[:limit]
