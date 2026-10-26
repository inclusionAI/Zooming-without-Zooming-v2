"""Parse API responses for function/tool calls and answer tags."""

import re
import json
from typing import Tuple, Any, Dict, Union


def parse_function_call_response(response: Union[str, Dict], text_content: str = None) -> Tuple[str, Any]:
    """
    Parse API response: OpenAI/Gemini tool_calls or text with <answer> tag.
    Returns (action_type, data). action_type: "function_call", "tool_call", "answer", "reflect", "text", "error".
    """
    if isinstance(response, dict):
        if "tool_calls" in response and response["tool_calls"]:
            tc = response["tool_calls"]
            if isinstance(tc, list) and tc:
                first = tc[0]
                fn = first.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        params = json.loads(args)
                    except json.JSONDecodeError:
                        return "error", f"Invalid JSON: {args}"
                else:
                    params = args
                if not name:
                    return "error", "Missing tool name"
                return "function_call", {"tool_name": name, "parameters": params, "tool_call_id": first.get("id")}
        if "function_call" in response:
            fc = response["function_call"]
            if isinstance(fc, list):
                if not fc:
                    fc = None
                else:
                    fc = fc[0]
            # OpenAI-compatible vLLM responses commonly include
            # `function_call: null` for a normal text completion.  That is not
            # a malformed tool call: fall through to text_content below.
            if fc is not None:
                if not isinstance(fc, dict):
                    return "error", f"Unexpected function_call type: {type(fc)}"
                name = fc.get("name", "")
                params = fc.get("args", fc.get("arguments", {}))
                if not name:
                    return "error", "Missing function name"
                return "function_call", {"tool_name": name, "parameters": params}
        if text_content:
            response = text_content
        elif "content" in response:
            response = response["content"]
        elif "text" in response:
            response = response["text"]
        else:
            return "text", ""

    if isinstance(response, str):
        m = re.search(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(1).strip())
                if "tool_name" in parsed and "parameters" in parsed:
                    return "tool_call", parsed
                return "error", "tool_call must have tool_name and parameters"
            except json.JSONDecodeError as e:
                return "error", str(e)
        m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if m:
            return "answer", m.group(1).strip()
        m = re.search(r"<reflect>(.*?)</reflect>", response, re.DOTALL)
        if m:
            return "reflect", m.group(1).strip()
        return "text", response

    return "error", f"Unknown format: {type(response)}"
