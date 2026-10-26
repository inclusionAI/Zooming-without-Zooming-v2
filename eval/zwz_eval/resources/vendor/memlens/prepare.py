#!/usr/bin/env python3
"""Convert the official MEMLENS release into role-preserving API records."""

import argparse
import json
import os
from pathlib import Path


SYSTEM_PROMPT = (
    "Provide answers based on the given conversation history. "
    "If the question cannot be answered based on the given conversation, "
    'respond with "Insufficient information".'
)
DEFAULT_INSTRUCTION = "Directly output the answer with no extra output."


def image_path(info, image_root: Path) -> str | None:
    if not isinstance(info, dict) or not info.get("file"):
        return None
    value = Path(str(info["file"]))
    return str(value if value.is_absolute() else image_root / value)


def append_turn_blocks(text, image_infos, image_root, image_offset):
    paths = [p for p in (image_path(x, image_root) for x in image_infos or []) if p]
    blocks = []
    image_index = 0
    parts = str(text or "").split("<image>")
    if len(parts) > 1:
        for index, part in enumerate(parts):
            if part.strip():
                blocks.append({"type": "text", "text": part})
            if index < len(parts) - 1 and image_index < len(paths):
                blocks.append({"type": "image_path", "path": paths[image_index]})
                image_index += 1
    else:
        for path in paths:
            blocks.append({"type": "image_path", "path": path})
        if str(text or "").strip():
            blocks.append({"type": "text", "text": str(text).strip()})
        image_index = len(paths)
    for path in paths[image_index:]:
        blocks.append({"type": "image_path", "path": path})
    return blocks, paths, image_offset + len(paths)


def convert_record(row, split, image_root):
    messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    all_images = []
    image_offset = 0
    sessions = row.get("haystack_sessions") or []
    dates = row.get("haystack_dates") or []
    for session_index, session in enumerate(sessions, 1):
        if isinstance(session, dict):
            date = session.get("date", "unknown")
            turns = session.get("session") or []
        else:
            date = dates[session_index - 1] if session_index - 1 < len(dates) else "unknown"
            turns = session or []
        header_pending = True
        for turn in turns:
            blocks, paths, image_offset = append_turn_blocks(
                turn.get("content", ""), turn.get("images") or [], image_root, image_offset
            )
            all_images.extend(paths)
            if header_pending:
                header = f"=== Session {session_index} (Date: {date}) ==="
                if blocks and blocks[0].get("type") == "text":
                    blocks[0]["text"] = header + "\n" + blocks[0]["text"]
                else:
                    blocks.insert(0, {"type": "text", "text": header})
                header_pending = False
            if blocks:
                role = "assistant" if turn.get("role") == "assistant" else "user"
                messages.append({"role": role, "content": blocks})
    question_text = (
        f"{DEFAULT_INSTRUCTION}\n"
        f"Question Date: {row.get('question_date', 'unknown')}\n"
        f"Question: {row.get('question', '')}\n"
    )
    messages.append({"role": "user", "content": [{"type": "text", "text": question_text}]})
    qid = str(row.get("question_id", ""))
    return {
        "id": f"memlens-{split}-{qid}",
        "sample_id": qid,
        "query": row.get("question", ""),
        "images": all_images,
        "response": str(row.get("answer", "")),
        "category": row.get("question_type", ""),
        "task": f"memlens-{split}",
        "split": split,
        "system_prompt": SYSTEM_PROMPT,
        "messages": messages,
        "meta": {
            "question_subtype": row.get("question_subtype"),
            "question_date": row.get("question_date"),
            "haystack_session_ids": row.get("haystack_session_ids"),
            "answer_session_ids": row.get("answer_session_ids"),
            "prompt_protocol": "memlens_official_role_preserving_v1",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    source_dir = Path(args.source_dir)
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Scan the image tree once. Calling stat() for every repeated image path is
    # prohibitively slow on NAS/FUSE and can appear to hang for several minutes.
    existing_images = set()
    for subdir in ("haystack_images", "needle_images"):
        directory = image_root / subdir
        existing_images.update(str(directory / name) for name in os.listdir(directory))
    print(f"indexed {len(existing_images)} image files under {image_root}")
    for split in ("32k", "64k", "128k", "256k"):
        rows = json.loads((source_dir / f"dataset_{split}.json").read_text())
        converted = [convert_record(row, split, image_root) for row in rows]
        target = output_dir / f"memlens_{split}.json"
        target.write_text(json.dumps(converted, ensure_ascii=False), encoding="utf-8")
        missing = [p for row in converted for p in row["images"] if p not in existing_images]
        if missing:
            raise FileNotFoundError(f"{split}: {len(missing)} missing images; first={missing[0]}")
        print(f"{split}: {len(converted)} samples, {sum(len(x['images']) for x in converted)} images -> {target}")


if __name__ == "__main__":
    main()
