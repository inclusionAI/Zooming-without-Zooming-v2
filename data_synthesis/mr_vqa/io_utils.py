import hashlib
import json
import os
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def stable_id(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(records, path, append=False):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(record, path):
    write_jsonl([record], path, append=True)


def load_completed_ids(path, key="image_id"):
    if not path or not os.path.exists(path):
        return set()
    done = set()
    for item in iter_jsonl(path):
        value = item.get(key)
        if value:
            done.add(value)
    return done


def discover_images(folder):
    paths = []
    for root, _, files in os.walk(folder):
        for name in files:
            path = Path(root) / name
            if path.suffix.lower() in IMAGE_EXTS:
                paths.append(str(path))
    return sorted(paths)


def image_record_from_obj(obj, root=None):
    if isinstance(obj, str):
        path = obj
        caption = ""
    elif isinstance(obj, dict):
        path = ""
        for key in ("image_path", "image", "img_path", "path", "file_path", "filepath"):
            if isinstance(obj.get(key), str):
                path = obj[key]
                break
        caption = obj.get("caption") or obj.get("text") or obj.get("description") or ""
    else:
        return None

    if root and path and not os.path.isabs(path):
        path = os.path.join(root, path)
    if not path:
        return None
    return {
        "image_id": stable_id(os.path.abspath(path)),
        "image_path": os.path.abspath(path),
        "caption": caption,
    }

