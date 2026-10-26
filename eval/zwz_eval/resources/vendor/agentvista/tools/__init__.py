"""Tools package: web_search, image_search, visit, code_interpreter."""

import os

from .base import BaseTool
from .tool_registry import TOOL_REGISTRY, register_tool, get_tool, list_tools, get_tool_info

try:
    from .code_interpreter import CodeInterpreter
except ImportError:
    CodeInterpreter = None

try:
    from .web_search import WebSearch
except ImportError:
    WebSearch = None

try:
    from .visit import Visit
except ImportError:
    Visit = None

try:
    from .image_search import ImageSearch
except ImportError:
    ImageSearch = None

# Opt-in key-free backends. They intentionally overwrite only the registry
# entries, keeping the official classes untouched when the flag is absent.
if os.environ.get("AGENTVISTA_FREE_TOOLS", "").lower() in {"1", "true", "yes"}:
    from .free_backends import FreeWebSearch, FreeImageSearch, FreeVisit

__all__ = [
    "BaseTool",
    "TOOL_REGISTRY",
    "register_tool",
    "get_tool",
    "list_tools",
    "get_tool_info",
    "CodeInterpreter",
    "WebSearch",
    "Visit",
    "ImageSearch",
]
