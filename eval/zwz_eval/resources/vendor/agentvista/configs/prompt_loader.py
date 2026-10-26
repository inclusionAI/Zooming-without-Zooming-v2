"""
Load inference prompts from YAML. No evaluator prompts (greedy-only).
"""

import os
import re
import yaml

TOOL_DESCRIPTIONS = {
    "web_search": "Search the web for information, facts, or current events",
    "image_search": "Search for related images using text query or reverse image search",
    "visit": "Visit a webpage and extract its main content",
    "code_interpreter": "Execute Python code for image processing, analysis, and calculations",
}


def _rewrite_system_prompt_tools(system_prompt: str, enabled_tools: list) -> str:
    """Replace the numbered tool list in the system prompt with only the enabled tools."""
    if not enabled_tools or "You have access to the following tools:" not in system_prompt:
        return system_prompt
    lines = []
    for i, name in enumerate(enabled_tools, 1):
        desc = TOOL_DESCRIPTIONS.get(name, name)
        lines.append(f"    {i}. **{name}**: {desc}")
    replacement = "\n".join(lines) + "\n"
    pattern = r"(# AVAILABLE TOOLS:\s*\n\s*You have access to the following tools:\s*\n)(.*?)(\s*\n\s*# INSTRUCTIONS:)"
    match = re.search(pattern, system_prompt, re.DOTALL)
    if not match:
        return system_prompt
    return system_prompt[: match.start(2)] + replacement + system_prompt[match.end(2) :]


def load_inference_prompts(args):
    """Load inference prompts from YAML. Returns dict with SYSTEM_PROMPT, etc."""
    with open(args.inference_prompts_path, "r", encoding="utf-8") as f:
        PROMPTS = yaml.safe_load(f)
    system_prompt = PROMPTS["system_prompts"][args.system_prompt_key]
    enabled_env = os.environ.get("ENABLED_TOOLS", "").strip()
    if enabled_env:
        enabled_list = [t.strip() for t in enabled_env.split(",") if t.strip()]
        system_prompt = _rewrite_system_prompt_tools(system_prompt, enabled_list)
    prompts = {
        "SYSTEM_PROMPT": system_prompt,
        "FEEDBACK_PROMPT_TOO_SMALL": PROMPTS["feedback"]["too_small_bbox"],
        "TOOL_CALL_CROP_MULTI_TRUN_PROMPT": PROMPTS["turn_templates"]["local_greedy"]["tool_call_crop_multi_turn"],
        "INITIAL_USER_PROMPT": PROMPTS["turn_templates"]["local_greedy"]["initial_user_prompt"],
        "TTS_TOOL_CALL_CROP_MULTI_TRUN_PROMPT": PROMPTS["turn_templates"]["placeholder_based"]["tool_call_crop_multi_turn"],
    }
    return prompts


def setup_global_prompts(prompts_dict):
    """Return tuple (SYSTEM_PROMPT, TOOL_CALL_CROP_MULTI_TRUN_PROMPT, INITIAL_USER_PROMPT, FEEDBACK_PROMPT_TOO_SMALL, TTS_TOOL_CALL_CROP_MULTI_TRUN_PROMPT)."""
    return (
        prompts_dict["SYSTEM_PROMPT"],
        prompts_dict["TOOL_CALL_CROP_MULTI_TRUN_PROMPT"],
        prompts_dict["INITIAL_USER_PROMPT"],
        prompts_dict["FEEDBACK_PROMPT_TOO_SMALL"],
        prompts_dict["TTS_TOOL_CALL_CROP_MULTI_TRUN_PROMPT"],
    )
