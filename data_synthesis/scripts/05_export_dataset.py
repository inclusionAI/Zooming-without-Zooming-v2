import argparse
import json
import os
import sys

from datasets import Dataset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from mr_vqa.io_utils import iter_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Export validated VQA to parquet")
    parser.add_argument("--validated_jsonl", type=str, required=True)
    parser.add_argument("--output_parquet", type=str, required=True)
    parser.add_argument("--format", choices=["open", "mcq"], default="open")
    parser.add_argument("--include_invalid", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for rec in tqdm(iter_jsonl(args.validated_jsonl), desc="export"):
        for item in rec.get("vqa_results", []):
            if not args.include_invalid and not str(item.get("status", "")).startswith("valid"):
                continue
            if args.format == "mcq":
                if not item.get("mcq"):
                    continue
                problem = item["mcq"]["mcq_question"]
                answer = item["mcq"]["correct_option"]
            else:
                problem = (item.get("final_question")
                           or (item.get("mcq") or {}).get("neutral_question")
                           or item.get("question"))
                answer = item.get("answer")
            if not problem or not answer:
                continue
            rows.append({
                "images": [item.get("final_image_path")
                           or item.get("marked_image_path")
                           or rec.get("image_path")],
                "problem": problem,
                "answer": answer,
                "image_id": rec.get("image_id"),
                "image_path": rec.get("image_path"),
                "question_type": item.get("question_type"),
                "template_id": item.get("template_id"),
                "final_answer_kind": item.get("final_answer_kind"),
                "observed_answer_kind": item.get("observed_answer_kind"),
                "answer_space": item.get("answer_space"),
                "certificate_mode": (
                    (item.get("reasoning_certificate") or {}).get("mode")
                ),
                "reasoning_certificate": json.dumps(
                    item.get("reasoning_certificate") or {}, ensure_ascii=False
                ),
                "difficulty": item.get("difficulty"),
                "privileged_layout_path": item.get("privileged_layout_path"),
                "marked_image_path": item.get("marked_image_path"),
                "final_image_path": item.get("final_image_path"),
                "reference_mode": item.get("reference_mode", "marked"),
                "pipeline_version": item.get("pipeline_version"),
                "evidence_scope": item.get("evidence_scope", "single_image"),
                "source_image_ids": json.dumps(
                    item.get("source_image_ids", []), ensure_ascii=False
                ),
                "source_images": json.dumps(
                    item.get("source_images", []), ensure_ascii=False
                ),
                "coordinate_frame": item.get("coordinate_frame", "not_applicable"),
                "relation_basis": item.get("relation_basis", "not_applicable"),
                "spatial_relation": item.get("spatial_relation", ""),
                "evidence_regions": json.dumps(item.get("evidence_regions", []), ensure_ascii=False),
                "status": item.get("status"),
            })

    os.makedirs(os.path.dirname(args.output_parquet) or ".", exist_ok=True)
    Dataset.from_list(rows).to_parquet(args.output_parquet)
    print(f"exported {len(rows)} rows to {args.output_parquet}")


if __name__ == "__main__":
    main()
