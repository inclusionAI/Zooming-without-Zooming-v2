import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from mr_vqa.image_utils import load_image
from mr_vqa.io_utils import append_jsonl, iter_jsonl, load_completed_ids
from mr_vqa.regions import propose_grounded_regions


_file_lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser(description="Extract grounded regions with one MLLM call per image")
    parser.add_argument("--manifest_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--api_url", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen3.5-397B-A17B")
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_regions", type=int, default=18)
    parser.add_argument("--min_regions", type=int, default=4)
    parser.add_argument("--min_area_ratio", type=float, default=0.0004)
    parser.add_argument("--max_area_ratio", type=float, default=0.18)
    parser.add_argument("--max_iou", type=float, default=0.65)
    parser.add_argument("--max_workers", type=int, default=4,
                        help="Parallel image workers")
    parser.add_argument(
        "--max_concurrent_per_key", type=int, default=12,
        help="Maximum in-flight requests for each comma-separated API key.",
    )
    return parser.parse_args()


def process_one(rec, args):
    try:
        img = load_image(rec["image_path"])
        rec["image_width"], rec["image_height"] = img.size
        proposal = propose_grounded_regions(
            rec,
            api_url=args.api_url,
            api_key=args.api_key,
            model=args.model,
            max_regions=args.max_regions,
            min_area_ratio=args.min_area_ratio,
            max_area_ratio=args.max_area_ratio,
            max_iou=args.max_iou,
            return_metadata=True,
            max_concurrent_per_key=args.max_concurrent_per_key,
        )
        regions = proposal["regions"]
        if not proposal["suitable_for_multi_region_vqa"]:
            status = "skipped_low_semantic_complexity"
        elif len(regions) < args.min_regions:
            status = "skipped_too_few_regions"
        else:
            status = "ok"
        return {
            **rec,
            "status": status,
            "regions": regions,
            "semantic_complexity": {
                "suitable": proposal["suitable_for_multi_region_vqa"],
                "reason": proposal["richness_reason"],
                "coherent_sets": proposal["coherent_sets"],
            },
        }
    except Exception as exc:
        return {**rec, "status": "error", "error": str(exc), "regions": []}


def main():
    args = parse_args()
    records = list(iter_jsonl(args.manifest_jsonl))
    if args.max_items:
        records = records[:args.max_items]
    if args.resume:
        done = load_completed_ids(args.output_jsonl, key="image_id")
        records = [r for r in records if r["image_id"] not in done]

    if not records:
        print("nothing to do")
        return

    workers = max(1, min(args.max_workers, len(records)))
    print(f"processing {len(records)} images with {workers} workers")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_one, rec, args): rec for rec in records}
        with tqdm(total=len(futures), desc="extract grounded regions") as pbar:
            for fut in as_completed(futures):
                rec = futures[fut]
                result = fut.result()
                with _file_lock:
                    append_jsonl(result, args.output_jsonl)
                pbar.update(1)

    print(f"wrote region records to {args.output_jsonl}")


if __name__ == "__main__":
    main()
