"""Build OpenAI-compatible tools schema from tool registry."""

from typing import List, Dict, Any

try:
    from tools import get_tool_info, TOOL_REGISTRY
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False
    TOOL_REGISTRY = {}


def build_openai_tools_schema(tool_names: List[str] = None) -> List[Dict[str, Any]]:
    """Build OpenAI tools format: [{"type": "function", "function": {...}}, ...]."""
    if not TOOLS_AVAILABLE:
        return []
    if tool_names is None:
        tool_names = list(TOOL_REGISTRY.keys())
    out = []
    for name in tool_names:
        try:
            info = get_tool_info(name)
            if not info:
                continue
            d = {"type": "function", "function": {"name": info["name"], "description": info["description"]}}
            if info.get("parameters"):
                d["function"]["parameters"] = info["parameters"]
            out.append(d)
        except Exception:
            continue
    return out
