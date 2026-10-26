# Selected definitions from the project-adapted reference source.
# Not a relicense of the original. See MANIFEST.json and THIRD_PARTY_NOTICES.md.
from __future__ import annotations
import re
import json
from decimal import Decimal, ROUND_HALF_UP
def build_messages(
    item: Dict,
    default_system: str = "",
    message_policy: str = "legacy",
) -> List[Dict]:
    '\n    \u4ece\u6807\u51c6\u683c\u5f0f\u6784\u5efa vLLM messages\n\n    \u5904\u7406 <image>/<video> \u5360\u4f4d\u7b26\uff0c\u8f6c\u6362\u4e3a multimodal content list\u3002\n\n    Args:\n        item: \u6807\u51c6\u683c\u5f0f\u6570\u636e\n        default_system: \u9ed8\u8ba4 system prompt\n        message_policy: legacy \u4fdd\u6301\u539f\u884c\u4e3a\uff1buser_only \u4e0d\u63d2\u5165 system \u6d88\u606f\n\n    Returns:\n        messages: [{"role": "system/user", "content": [...]}]\n    '
    messages = []
    img_list = item.get('images',[])
    if isinstance(img_list,str):
        img_list = [img_list]
    vid_list = item.get('videos',[])
    if isinstance(vid_list,str):
        vid_list = [vid_list]
    warning_str = ""

    # System prompt
    system_text = item.get("system") or default_system
    if message_policy != "user_only" and (system_text or system_text == ''):
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_text}]
        })

    # Preserved upstream logic; see the corresponding protocol documentation.
    queries = item.get("query", [])
    if isinstance(queries, str):
        queries = [queries]

    img_cnt, vid_cnt = 0, 0
    for query_text in queries:
        # Preserved upstream logic; see the corresponding protocol documentation.
        segments = re.split(r"(<image>|<video>)", query_text)
        content_list = []
        for segment in segments:
            if segment == "<image>":
                if img_cnt < len(img_list):
                    img_cnt += 1
                    content_list.append({"type": "image"})
                else:
                    warning_str += f"Warning: \u56fe\u7247\u6570\u91cf\u4e0d\u5339\u914d，\u5df2\u622a\u65ad，images\u56fe\u7247 {len(img_list)} \u5f20，\u5b9e\u9645\u5360\u4f4d\u7b26 {img_cnt} \u4e2a\n"
            elif segment == "<video>":
                if vid_cnt < len(vid_list):
                    vid_cnt += 1
                    content_list.append({"type": "video"})
                else:
                    warning_str += f"Warning: \u89c6\u9891\u6570\u91cf\u4e0d\u5339\u914d，\u5df2\u622a\u65ad，videos\u89c6\u9891 {len(vid_list)} \u4e2a，\u5b9e\u9645\u5360\u4f4d\u7b26 {vid_cnt} \u4e2a\n"
            elif segment:
                content_list.append({"type": "text", "text": segment})

        if content_list:
            messages.append({"role": "user", "content": content_list})
    
    if len(messages) > 0:
        if img_cnt < len(img_list):
            messages[-1]['content'].extend([{"type": "image"}]*(len(img_list)-img_cnt))
            warning_str += f"Warning: \u56fe\u7247\u6570\u91cf\u4e0d\u5339\u914d，\u5df2\u8865\u9f50，images\u56fe\u7247 {len(img_list)} \u5f20，\u5b9e\u9645\u5360\u4f4d\u7b26 {img_cnt} \u4e2a\n"
        if vid_cnt < len(vid_list):
            messages[-1]['content'].extend([{"type": "video"}]*(len(vid_list)-vid_cnt))
            warning_str += f"Warning: \u89c6\u9891\u6570\u91cf\u4e0d\u5339\u914d，\u5df2\u8865\u9f50，videos\u89c6\u9891 {len(vid_list)} \u4e2a，\u5b9e\u9645\u5360\u4f4d\u7b26 {vid_cnt} \u4e2a\n"

    if warning_str != "":
        item['vl_num_check'] = warning_str
    return messages
