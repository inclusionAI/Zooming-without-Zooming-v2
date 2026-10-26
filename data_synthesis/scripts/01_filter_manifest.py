"""Stage 01 — build a manifest and filter it by image content-richness.

Takes either a folder of images (recursively scanned) or an existing
manifest JSONL with ``{image_id, image_path, caption}`` rows, and writes
a filtered manifest JSONL for ``run_pipeline_continuous.sh``.

Each image is scored with ``mr_vqa.image_filter`` on five metrics
(long-side resolution, mean RGB Shannon entropy, Canny edge density,
luminance std-dev, long/short-side ratio). Only images passing all
thresholds are written out, so stage 02 does not spend region-extraction
budget on flat or low-content scenes.

Usage:

  python scripts/01_filter_manifest.py \
    --image_folder /path/to/images \
    --output_jsonl data/filtered_manifest.jsonl \
    --workers 32

  # or, from an existing manifest:
  python scripts/01_filter_manifest.py \
    --input_jsonl data/manifest.jsonl \
    --output_jsonl data/filtered_manifest.jsonl
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mr_vqa import image_filter
from mr_vqa.io_utils import (
    discover_images,
    image_record_from_obj,
    iter_jsonl,
    load_completed_ids,
    write_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build + content-richness-filter an image manifest")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image_folder", type=str,
                        help="Folder of images to scan recursively")
    source.add_argument("--input_jsonl", type=str,
                        help="Existing manifest JSONL to filter")
    parser.add_argument("--output_jsonl", type=str, required=True,
                        help="Filtered manifest JSONL to write")
    parser.add_argument("--workers", type=int, default=32,
                        help="Concurrent process workers")
    parser.add_argument("--max_items", type=int, default=None,
                        help="Cap input records after dedup")
    # Filter thresholds; defaults come from mr_vqa.image_filter.
    parser.add_argument("--min_resolution", type=int,
                        default=image_filter.MIN_RESOLUTION)
    parser.add_argument("--min_entropy", type=float,
                        default=image_filter.MIN_ENTROPY)
    parser.add_argument("--min_edge_density", type=float,
                        default=image_filter.MIN_EDGE_DENSITY)
    parser.add_argument("--min_std_dev", type=float,
                        default=image_filter.MIN_STD_DEV)
    parser.add_argument("--min_long_short_ratio", type=float,
                        default=image_filter.MIN_LONG_SHORT_RATIO)
    return parser.parse_args()


def _load_records(args):
    records = []
    if args.input_jsonl:
        for obj in iter_jsonl(args.input_jsonl):
            rec = image_record_from_obj(obj)
            if rec and os.path.exists(rec["image_path"]):
                records.append(rec)
    else:
        for path in discover_images(args.image_folder):
            rec = image_record_from_obj(path)
            if rec:
                records.append(rec)

    deduped, seen = [], set()
    for rec in records:
        if rec["image_id"] in seen:
            continue
        seen.add(rec["image_id"])
        deduped.append(rec)
    if args.max_items:
        deduped = deduped[:args.max_items]
    return deduped


def _process_one(rec, thresholds):
    """Filter a single record. Returns (rec | None, reason | None)."""
    path = rec.get("image_path")
    if not path or not os.path.exists(path):
        return None, "file_missing"
    try:
        with open(path, "rb") as f:
            metrics = image_filter.analyze(f.read())
    except Exception:
        return None, "analyze_failed"
    ok, reason, metrics = image_filter.evaluate(metrics, **thresholds)
    if not ok:
        return None, reason
    out = {
        "image_id": rec.get("image_id"),
        "image_path": path,
        "caption": rec.get("caption", ""),
        "width": int(metrics["width"]),
        "height": int(metrics["height"]),
        "entropy": round(float(metrics["entropy"]), 4),
        "edge_density": round(float(metrics["edge_density"]), 6),
        "std_dev": round(float(metrics["std_dev"]), 2),
        "long_short_ratio": round(
            float(metrics["long_side"] / max(1, metrics["short_side"])), 4),
    }
    return out, None


def main():
    args = parse_args()
    thresholds = {
        "min_resolution": args.min_resolution,
        "min_entropy": args.min_entropy,
        "min_edge_density": args.min_edge_density,
        "min_std_dev": args.min_std_dev,
        "min_long_short_ratio": args.min_long_short_ratio,
    }

    records = _load_records(args)
    print(f"[input] {len(records)} records")

    done = load_completed_ids(args.output_jsonl)
    pending = [r for r in records if r.get("image_id") not in done]
    print(f"[resume] {len(done)} already written; {len(pending)} pending")
    if not pending:
        print("[done] nothing to do")
        return

    kept, filtered, errors = [], 0, 0
    fail_reasons = {}
    worker = partial(_process_one, thresholds=thresholds)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(worker, r) for r in pending]
        for i, fut in enumerate(as_completed(futures), start=1):
            try:
                rec, reason = fut.result()
            except Exception:
                rec, reason = None, "exception"
            if rec is not None:
                kept.append(rec)
            elif reason in ("file_missing", "analyze_failed", "exception"):
                errors += 1
            else:
                filtered += 1
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
            if i % 500 == 0:
                print(f"[progress] {i}/{len(pending)} kept={len(kept)} "
                      f"filtered={filtered} err={errors}", flush=True)

    # Sort by input order for a deterministic manifest, then append so the
    # output can be resumed across runs with load_completed_ids().
    order = {r["image_id"]: n for n, r in enumerate(records)}
    kept.sort(key=lambda r: order.get(r["image_id"], len(order)))
    write_jsonl(kept, args.output_jsonl, append=True)

    print(f"\n[done] kept={len(kept)} filtered={filtered} err={errors} "
          f"-> {args.output_jsonl}")
    print(f"[reasons] {fail_reasons}")


if __name__ == "__main__":
    main()
