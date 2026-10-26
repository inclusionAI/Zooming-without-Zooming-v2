'\nAPI-based multimodal inference.\n\n\u901a\u8fc7 OpenAI \u517c\u5bb9\u63a5\u53e3\uff08vLLM serve\uff09\u5bf9\u6307\u5b9a benchmark \u8fdb\u884c\u63a8\u7406\uff0c\n\u652f\u6301\u5355\u56fe/\u591a\u56fe\u3001\u76f8\u5bf9/\u7edd\u5bf9\u56fe\u50cf\u8def\u5f84\u3001\u65ad\u70b9\u7eed\u4f20\u4e0e\u5931\u8d25\u91cd\u8bd5\u3002\n\n\u8f93\u5165 benchmark JSON \u7edf\u4e00 schema\uff1a\u6bcf\u6761\u6837\u672c\u542b\n  - query:   \u95ee\u9898\u6587\u672c\uff0c\u53ef\u80fd\u542b <image> \u5360\u4f4d\u7b26\n  - images:  \u56fe\u50cf\u8def\u5f84\u5217\u8868\n  - response: \u6807\u51c6\u7b54\u6848\uff08ground truth\uff09\n\n\u8f93\u51fa\uff1a{out_dir}/{benchmark}/{model_name}_answer.jsonl\uff0c\u6bcf\u884c\u4e00\u6761\u8bb0\u5f55\uff0c\n      \u8ffd\u52a0 model_answer\uff08\u63d0\u53d6\u540e\u7684\u7b54\u6848\uff09\u53ca\u53ef\u9009 model_answer_raw\u3002\n'

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

from openai import OpenAI
from PIL import Image
from tqdm import tqdm

# Preserved upstream logic; see the corresponding protocol documentation.
VSTAR_BENCHMARKS = {"vstar"}
VSTAR_MAX_IMAGE_SIZE = 20 * 1024 * 1024

# Preserved upstream logic; see the corresponding protocol documentation.
DEFAULT_IMAGE_ROOTS = []


def pick_image_path(item):
    '\u8fd4\u56de\u7b2c\u4e00\u6761\u56fe\u50cf\u8def\u5f84\uff08\u517c\u5bb9\u7f3a\u5931\u60c5\u51b5\uff09\u3002'
    return (item.get("images") or [""])[0]


def resolve_image_path(path_str, image_roots):
    '\u76f8\u5bf9\u8def\u5f84\u5219\u4f9d\u6b21\u5728 image_roots \u4e2d\u67e5\u627e\uff1b\u7edd\u5bf9\u8def\u5f84\u6216\u5df2\u5b58\u5728\u5219\u76f4\u63a5\u7528\u3002'
    if not path_str:
        return path_str
    if os.path.isabs(path_str) and os.path.exists(path_str):
        return path_str
    if not os.path.isabs(path_str):
        for root in image_roots:
            cand = os.path.join(root, path_str)
            if os.path.exists(cand):
                return cand
    return path_str  # Preserved upstream logic; see the corresponding protocol documentation.


def make_sample_uid(item, benchmark):
    '\u751f\u6210\u7a33\u5b9a\u552f\u4e00\u6807\u8bc6\uff0c\u7528\u4e8e\u65ad\u70b9\u7eed\u4f20\u53bb\u91cd\u3002'
    if benchmark == "mimic-counting":
        stable_obj = {"benchmark": benchmark, "id": item.get("id"), "query": item.get("query", ""), "images": item.get("images") or []}
        raw = json.dumps(stable_obj, ensure_ascii=False, sort_keys=True)
        return "sha1:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()
    for key in ("sample_uid", "uid", "question_id", "index", "id"):
        value = item.get(key)
        if value is not None and str(value) != "":
            return f"{key}:{value}"
    stable_obj = {
        "benchmark": benchmark,
        "images": item.get("images") or [],
        "query": item.get("query", ""),
    }
    raw = json.dumps(stable_obj, ensure_ascii=False, sort_keys=True)
    return "sha1:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()


def should_retry_existing_record(item):
    '\u5df2\u5b58\u5728\u8bb0\u5f55\u662f\u5426\u9700\u8981\u91cd\u8dd1\uff08\u7a7a / \u62a5\u9519\uff09\u3002'
    model_answer = item.get("model_answer")
    if not isinstance(model_answer, str):
        return True
    model_answer = model_answer.strip()
    if not model_answer:
        return True
    return model_answer.startswith("[API_ERROR]") or model_answer.startswith("[FUTURE_ERROR]")


def compact_existing_output(path, benchmark):
    '\u5408\u5e76\u65ad\u70b9\u6587\u4ef6\u4e2d\u7684\u91cd\u590d\u6837\u672c\uff0c\u4fdd\u7559\u6700\u4f18\u8bb0\u5f55\uff08\u6b63\u5e38\u4f18\u4e8e\u9519\u8bef\uff09\u3002'
    if not path.exists():
        return [], {}, False
    ordered_uids = []
    best_records = {}
    changed = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                changed = True
                continue
            try:
                record = json.loads(line)
            except Exception:
                changed = True
                continue
            sample_uid = record.get("sample_uid") or make_sample_uid(record, benchmark)
            record["sample_uid"] = sample_uid
            if sample_uid not in best_records:
                ordered_uids.append(sample_uid)
                best_records[sample_uid] = record
                continue
            changed = True
            prev = best_records[sample_uid]
            prev_is_error = should_retry_existing_record(prev)
            curr_is_error = should_retry_existing_record(record)
            if prev_is_error and not curr_is_error:
                best_records[sample_uid] = record
            elif prev_is_error == curr_is_error:
                best_records[sample_uid] = record
    return ordered_uids, best_records, changed


def rewrite_output(path, ordered_uids, record_map):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for sample_uid in ordered_uids:
            record = record_map.get(sample_uid)
            if record is None:
                continue
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def normalize_model_answer(model_answer_raw):
    '\u4ece\u6a21\u578b\u539f\u59cb\u8f93\u51fa\u4e2d\u63d0\u53d6\u6700\u7ec8\u7b54\u6848\uff1a<answer> \u6807\u7b7e > Answer: > \u539f\u6587\u3002'
    if not isinstance(model_answer_raw, str):
        return model_answer_raw
    # A few checkpoints repeat complete answer blocks until max_tokens and end
    # with </think>. Preserve the last non-empty answer from the full response
    # before stripping the reasoning suffix.
    answer_matches = re.findall(
        r"<answer>\s*(.*?)\s*</answer>", model_answer_raw, flags=re.I | re.S
    )
    for answer in reversed(answer_matches):
        if answer.strip():
            return answer.strip()
    full_response = model_answer_raw
    think_end = model_answer_raw.rfind("</think>")
    if think_end != -1:
        final_response = model_answer_raw[think_end + len("</think>"):].strip()
        # A truncated/repetitive response can end exactly at </think>. Keep the
        # generated text for the benchmark judge instead of turning it into an
        # API failure that blocks the whole resume pipeline.
        if final_response:
            model_answer_raw = final_response
        else:
            # Degenerate generations sometimes repeat `answer </think>` many
            # times. The final segment between the last two closing tags is the
            # actual answer; do not pass the whole repetition to the judge.
            segments = re.split(r"\\?</think\s*>", full_response, flags=re.I)
            repeated_tail = segments[-2].strip() if len(segments) >= 3 else ""
            repeated_tail = re.sub(
                r"^.*?<think\s*>", "", repeated_tail, flags=re.I | re.S
            ).strip()
            model_answer_raw = repeated_tail or full_response.strip()
    start = model_answer_raw.rfind("<answer>")
    end = model_answer_raw.find("</answer>", start + len("<answer>")) if start != -1 else -1
    if start != -1 and end != -1 and end > start:
        return model_answer_raw[start + len("<answer>"):end].strip()
    if "Answer:" in model_answer_raw:
        return model_answer_raw[model_answer_raw.find("Answer:"):].strip()
    return model_answer_raw.strip()


def image_to_data_uri(path_str, benchmark):
    '\u5c06\u56fe\u50cf\u8f6c\u4e3a data URI\uff1bvstar benchmark \u5bf9\u8d85\u5927\u56fe\u8fdb\u884c\u538b\u7f29\u3002'
    if benchmark in VSTAR_BENCHMARKS:
        img = Image.open(path_str).convert("RGB")
        output = BytesIO()
        img.save(output, format="PNG")
        byte_data = output.getvalue()
        while len(byte_data) > VSTAR_MAX_IMAGE_SIZE and img.size[0] > 100 and img.size[1] > 100:
            new_size = (int(img.size[0] * 0.75), int(img.size[1] * 0.75))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            output = BytesIO()
            img.save(output, format="PNG")
            byte_data = output.getvalue()
        b64 = base64.b64encode(byte_data).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    p = Path(path_str)
    with open(p, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    mime = mimetypes.guess_type(str(p))[0] or "image/jpeg"
    return f"data:{mime};base64,{b64}"



def build_user_content(query, images, benchmark, image_roots):

    content = []
    placeholder_count = query.count("<image>")

    resolved_images = [resolve_image_path(img, image_roots) for img in images]

    if placeholder_count >= 2:
        parts = query.split("<image>")
        img_idx = 0
        for i, part in enumerate(parts):
            if part.strip():
                content.append({"type": "text", "text": part.strip()})
            if i < len(parts) - 1 and img_idx < len(resolved_images):
                data_uri = image_to_data_uri(resolved_images[img_idx], benchmark)
                content.append({"type": "image_url", "image_url": {"url": data_uri}})
                img_idx += 1
        # Preserved upstream logic; see the corresponding protocol documentation.
        while img_idx < len(resolved_images):
            data_uri = image_to_data_uri(resolved_images[img_idx], benchmark)
            content.append({"type": "image_url", "image_url": {"url": data_uri}})
            img_idx += 1
    else:
        for img_path in resolved_images:
            data_uri = image_to_data_uri(img_path, benchmark)
            content.append({"type": "image_url", "image_url": {"url": data_uri}})
        text = query.replace("<image>", "").strip()
        content.append({"type": "text", "text": text})

    return content


def build_record_messages(item, benchmark, image_roots):
    """Materialize a benchmark-provided OpenAI message list without flattening roles.

    Text blocks are preserved verbatim. Converter-side ``image_path`` blocks are
    replaced with API ``image_url`` blocks at request time so JSON stays small.
    """
    source_messages = item.get("messages")
    if not isinstance(source_messages, list) or not source_messages:
        return None
    messages = []
    for source in source_messages:
        role = source.get("role")
        content = source.get("content")
        if not role or not isinstance(content, list):
            raise ValueError("messages must contain role + list content")
        materialized = []
        for block in content:
            block_type = block.get("type")
            if block_type == "text":
                materialized.append({"type": "text", "text": str(block.get("text", ""))})
            elif block_type == "image_path":
                path = resolve_image_path(str(block.get("path", "")), image_roots)
                if not path or not os.path.isfile(path):
                    raise FileNotFoundError(f"message image not found: {path}")
                materialized.append({
                    "type": "image_url",
                    "image_url": {"url": image_to_data_uri(path, benchmark)},
                })
            elif block_type == "image_url":
                materialized.append(block)
            else:
                raise ValueError(f"unsupported message block type: {block_type}")
        messages.append({"role": role, "content": materialized})
    return messages


def main():
    parser = argparse.ArgumentParser(description="Run inference via OpenAI-compatible API")
    parser.add_argument("--benchmark", required=True, type=str)
    parser.add_argument("--benchmark_json", required=True, type=str, help="Path to benchmark JSON file")
    parser.add_argument("--out_dir", default="model_answer", type=str)
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--api_base", required=True, type=str,
                        help="OpenAI-compatible API base URL(s); comma-separated for multi-replica load balancing")
    parser.add_argument("--api_key", default="EMPTY", type=str)
    parser.add_argument("--model_id", required=True, type=str, help="OpenAI model ID")
    parser.add_argument("--max_tokens", default=4096, type=int)
    parser.add_argument("--max_retries", default=3, type=int)
    parser.add_argument("--parallel_workers", default=32, type=int)
    parser.add_argument("--max_samples", default=0, type=int,
                        help="Process at most this many pending samples (0 = all); useful for smoke tests")
    parser.add_argument("--abort_after_failures", default=8, type=int,
                        help="Abort after this many consecutive failed requests without writing failures")
    parser.add_argument("--request_timeout", default=600, type=int,
                        help="Per-request OpenAI client timeout in seconds")
    parser.add_argument("--enable_thinking", type=str, default="False", choices=["True", "False"],
                        help="Set enable_thinking via chat_template_kwargs (True=on, False=off). Default False.")
    parser.add_argument("--system_prompt", type=str, default=None,
                        help="Optional system prompt; omitted by default")
    parser.add_argument("--image_roots", type=str, default=",".join(DEFAULT_IMAGE_ROOTS),
                        help="Comma-separated roots for resolving relative image paths")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    benchmark = args.benchmark
    image_roots = [r for r in args.image_roots.split(",") if r]

    data_path = Path(args.benchmark_json)
    with open(data_path, "r", encoding="utf-8") as f:
        total_data = json.load(f)

    out_dir = Path(args.out_dir) / benchmark
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model_name}_answer.jsonl"

    done_ids = set()
    retry_ids = set()
    if out_path.exists():
        ordered_uids, existing_records, was_compacted = compact_existing_output(out_path, benchmark)
        if was_compacted:
            rewrite_output(out_path, ordered_uids, existing_records)
            print(f"Compacted checkpoint file to {len(existing_records)} unique samples.")
        for sample_uid in ordered_uids:
            x = existing_records[sample_uid]
            if should_retry_existing_record(x):
                retry_ids.add(sample_uid)
                continue
            done_ids.add(sample_uid)
        if retry_ids:
            print(f"Retrying {len(retry_ids)} samples with failed/empty existing records.")
        print(f"Checkpoint detected, {len(done_ids)} samples already completed.")

    todo_data = []
    for item in total_data:
        sample_uid = make_sample_uid(item, benchmark)
        if sample_uid in done_ids:
            continue
        record = dict(item)
        record["sample_uid"] = sample_uid
        todo_data.append(record)

    if args.max_samples > 0:
        todo_data = todo_data[:args.max_samples]

    print(f"Remaining samples to process: {len(todo_data)}")
    if not todo_data:
        print("All samples have been processed.")
        return

    # Preserved upstream logic; see the corresponding protocol documentation.
    api_bases = [b.strip() for b in args.api_base.split(",") if b.strip()]
    if not api_bases:
        raise ValueError('--api_base \u89e3\u6790\u540e\u4e3a\u7a7a')
    print(f"Using {len(api_bases)} API endpoint(s) for inference load balancing.")

    thread_local = threading.local()
    _client_counter = [0]
    _client_lock = threading.Lock()

    def get_client():
        c = getattr(thread_local, "client", None)
        if c is None:
            with _client_lock:
                idx = _client_counter[0] % len(api_bases)
                _client_counter[0] += 1
            thread_local.client = OpenAI(
                api_key=args.api_key,
                base_url=api_bases[idx],
                timeout=args.request_timeout,
            )
            thread_local.endpoint_idx = idx
            c = thread_local.client
        return c

    def run_one(item):
        sample_uid = item.get("sample_uid") or make_sample_uid(item, benchmark)
        images = item.get("images") or [""]
        query = item.get("query", "")
        messages = None
        for attempt in range(1, args.max_retries + 1):
            try:
                messages = build_record_messages(item, benchmark, image_roots)
                if messages is None:
                    dataset_system_prompt = item.get("system_prompt") or item.get("meta", {}).get("system_prompt")
                    # The official MileBench generator prepends the task instruction
                    # to the user prompt; it does not create a system-role message.
                    if benchmark.startswith("milebench-") and dataset_system_prompt:
                        query = str(dataset_system_prompt).rstrip() + "\n\n" + str(query).lstrip()
                        dataset_system_prompt = None
                    content = build_user_content(query, images, benchmark, image_roots)
                    # Preserved upstream logic; see the corresponding protocol documentation.
                    system_prompt = dataset_system_prompt or args.system_prompt
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": content})
                break
            except Exception as e:
                print(
                    f"[PREPROCESS_RETRY] sample_uid={sample_uid} "
                    f"attempt={attempt}/{args.max_retries} "
                    f"error={type(e).__name__}: {e}",
                    flush=True,
                )
                if attempt == args.max_retries:
                    raise
                time.sleep(1.0)

        model_answer = ""
        raw_model_answer = ""
        for attempt in range(1, args.max_retries + 1):
            client = get_client()
            try:
                extra_kwargs = {}
                if args.enable_thinking is not None:
                    extra_kwargs["extra_body"] = {
                        "chat_template_kwargs": {"enable_thinking": args.enable_thinking == "True"}
                    }
                resp = client.chat.completions.create(
                    model=args.model_id,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    **extra_kwargs,
                )
                choice = resp.choices[0]
                raw_model_answer = (choice.message.content or "").strip()
                model_answer = normalize_model_answer(raw_model_answer)
                # A normal stop with non-empty raw text is a genuine model
                # prediction even when normalization leaves only empty tags.
                # Preserve it so evaluation scores it instead of retrying a
                # deterministic output forever.
                if not isinstance(model_answer, str) or not model_answer.strip():
                    reasoning = getattr(choice.message, "reasoning_content", None)
                    finish_reason = getattr(choice, "finish_reason", None)
                    if finish_reason == "stop" and raw_model_answer:
                        print(
                            f"[MODEL_EMPTY_FINAL] sample_uid={sample_uid} "
                            f"finish_reason={finish_reason!r} "
                            f"raw={raw_model_answer!r} "
                            f"reasoning_len={len(reasoning or '')}",
                            flush=True,
                        )
                        model_answer = raw_model_answer
                    else:
                        raise RuntimeError(
                            "empty model response "
                            f"(finish_reason={finish_reason!r}, "
                            f"raw={raw_model_answer!r}, "
                            f"reasoning_len={len(reasoning or '')})"
                        )
                break
            except Exception as e:
                endpoint_idx = getattr(thread_local, "endpoint_idx", -1)
                endpoint = api_bases[endpoint_idx] if 0 <= endpoint_idx < len(api_bases) else "unknown"
                print(f"[API_RETRY] sample_uid={sample_uid} attempt={attempt}/{args.max_retries} endpoint={endpoint} error={type(e).__name__}: {e}", flush=True)
                if attempt == args.max_retries:
                    raw_model_answer = f"[API_ERROR] {e}"
                    model_answer = raw_model_answer
                else:
                    # Preserved upstream logic; see the corresponding protocol documentation.
                    if hasattr(thread_local, "client"):
                        del thread_local.client
                    if hasattr(thread_local, "endpoint_idx"):
                        del thread_local.endpoint_idx
                    time.sleep(1.0)
        record = dict(item)
        record["sample_uid"] = sample_uid
        record["request_message_roles"] = [message["role"] for message in messages]
        record["model_answer"] = model_answer
        if raw_model_answer and raw_model_answer != model_answer:
            record["model_answer_raw"] = raw_model_answer
        return record

    start = time.time()
    total_failures = 0
    consecutive_failures = 0
    successful_writes = 0
    aborted = False
    executor = ThreadPoolExecutor(max_workers=args.parallel_workers)
    future_to_item = {executor.submit(run_one, item): item for item in todo_data}
    try:
      with open(out_path, "a", encoding="utf-8") as f_out:
        with tqdm(total=len(todo_data), desc="Inference", unit="case", dynamic_ncols=True) as pbar:
            for future in as_completed(future_to_item):
                try:
                    record = future.result()
                except Exception as e:
                    item = future_to_item[future]
                    record = dict(item)
                    record["sample_uid"] = item.get("sample_uid") or make_sample_uid(item, benchmark)
                    record["model_answer"] = f"[FUTURE_ERROR] {e}"
                    print(
                        f"[FUTURE_ERROR] sample_uid={record['sample_uid']} "
                        f"error={type(e).__name__}: {e}",
                        flush=True,
                    )
                if should_retry_existing_record(record):
                    total_failures += 1
                    consecutive_failures += 1
                    # Keep the existing checkpoint record. Failed retries are deliberately
                    # not appended, so a dead API cannot pollute the result file.
                else:
                    consecutive_failures = 0
                    successful_writes += 1
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()
                pbar.update(1)
                if args.abort_after_failures > 0 and consecutive_failures >= args.abort_after_failures:
                    aborted = True
                    for pending in future_to_item:
                        pending.cancel()
                    break
    finally:
        executor.shutdown(wait=True, cancel_futures=aborted)

    elapsed = time.time() - start
    ordered_uids, final_records, was_compacted = compact_existing_output(out_path, benchmark)
    if was_compacted:
        rewrite_output(out_path, ordered_uids, final_records)
    print(f"Compacted final output to {len(final_records)} unique samples.")
    print(f"Inference done in {elapsed:.1f}s")
    print(f"Successful answers written: {successful_writes}; failed requests not written: {total_failures}")
    print(f"Saved answers to: {out_path}")
    if aborted:
        raise RuntimeError(
            f"Aborted after {consecutive_failures} consecutive failed requests; API service is unhealthy"
        )
    if total_failures:
        raise RuntimeError(f"{total_failures} requests failed; rerun is required")


if __name__ == "__main__":
    main()
