import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, Future

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from mr_vqa.io_utils import append_jsonl, iter_jsonl, load_completed_ids
from mr_vqa.mcq import generate_mcq, generate_unmarked_mcq, run_mcq_rejection_sampling
from mr_vqa.premium_audit import audit_candidate as run_premium_audit
from mr_vqa.questions import validate_question_answer_contract
from mr_vqa.validation import build_validation_images

# JudgeClient no longer needed: stage-04 only does MCQ generation + rejection
# sampling. The semantic-equivalence / drop-one-necessity tests have been
# merged into stage-03 selfcheck(see mr_vqa/selfcheck.py).


_file_lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser(description="Validate multi-region necessity")
    parser.add_argument("--candidates_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--evidence_dir", type=str, required=True)
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--api_url", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen3.5-397B-A17B",
                        help="Primary strong multimodal model")
    parser.add_argument("--secondary_api_key", type=str, default=None,
                        help="Optional API key for the secondary anchor model")
    parser.add_argument("--secondary_api_url", type=str, default=None,
                        help="Optional API URL for the secondary anchor model")
    parser.add_argument("--secondary_model", type=str, default="Kimi-K2.6",
                        help="Secondary strong multimodal model for unmarked reference auditing")
    parser.add_argument("--reject_api_key", type=str, default=None,
                        help="Optional API key for the smaller rejection model")
    parser.add_argument("--reject_api_url", type=str, default=None,
                        help="Optional API URL for the smaller rejection model")
    parser.add_argument("--reject_model", type=str, default=None,
                        help="Small model used on the final marked or unmarked image for MCQ rejection sampling")
    parser.add_argument("--premium_audit", action="store_true",
                        help="Run Qwen-VL cross-audit AFTER rejection sampling keeps a candidate. "
                             "Moved here from stage-03 so too-easy candidates are rejected (cheap, Qwen3.6) "
                             "before spending the Qwen-VL audit budget on them.")
    parser.add_argument("--premium_api_key", type=str, default=None,
                        help="Matrix API key for the optional premium audit.")
    parser.add_argument("--premium_qwen_vl_model", type=str, default="Qwen3.5-VL-72B")
    parser.add_argument("--premium_timeout", type=int, default=180)
    parser.add_argument("--premium_retries", type=int, default=2)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reject_samples", type=int, default=4,
                        help="Number of small-model requests on the final image")
    parser.add_argument("--unmarked_audit_samples", type=int, default=2,
                        help="Independent reference-audit samples per strong model")
    parser.add_argument("--layout_canvas_size", type=int, default=1024)
    parser.add_argument("--skip_mcq", action="store_true",
                        help="Skip MCQ generation and final rejection sampling")
    parser.add_argument("--max_valid_per_image", type=int, default=4)
    parser.add_argument("--image_workers", type=int, default=2,
                        help="Parallel images")
    parser.add_argument("--candidate_workers", type=int, default=4,
                        help="Parallel candidates within one image")
    return parser.parse_args()


def process_candidate(rec, cand, args):
    try:
        paths = build_validation_images(
            rec, cand, args.evidence_dir, canvas_size=args.layout_canvas_size
        )
        # GT is the VOTED answer produced by stage-03 selfcheck (most common
        # normalized answer among the 4 privileged-view votes, requiring
        # >=3 votes). The designer's hypothetical_answer is logged only.
        sc = cand.get("self_check") or {}
        answer = (sc.get("voted_gt") or "").strip()
        if not answer:
            return {**cand, "status": "invalid_missing_voted_gt",
                    "answer": answer,
                    "privileged_layout_path": paths["privileged_layout"],
                    "marked_image_path": paths["marked_original"],
                    "drop_one_paths": paths["drop"]}
        contract_ok, contract_reasons = validate_question_answer_contract(
            cand.get("question"), answer, cand.get("final_answer_kind"),
            cand.get("anchor_colors"),
            answer_semantics=cand.get("answer_semantics", "scalar"),
        )
        if not contract_ok:
            return {
                **cand,
                "status": "invalid_question_answer_scope",
                "answer": answer,
                "contract_reasons": contract_reasons,
                "privileged_layout_path": paths["privileged_layout"],
                "marked_image_path": paths["marked_original"],
                "drop_one_paths": paths["drop"],
            }
        status = "valid vqa"
        mcq = None
        rejection = None
        reference_mode = "marked"
        reference_audit = None
        final_image_path = paths["marked_original"]
        if not args.skip_mcq:
            # Anchor-independent aggregate answers can safely attempt a natural
            # unmarked form. Selection/pair answers often ARE outline colors,
            # so they retain the marked image unless separately re-anchored.
            if cand.get("question_type") in {
                "aggregate_count", "aggregate_attribute_count", "aggregate_numeric",
            }:
                try:
                    mcq, reference_audit = generate_unmarked_mcq(
                    paths["original"], cand["question"], answer,
                    cand.get("evidence_regions") or [],
                    rec.get("image_width"), rec.get("image_height"),
                    args.api_url, args.api_key, args.model,
                    args.secondary_api_url or args.api_url,
                    args.secondary_api_key or args.api_key,
                    args.secondary_model,
                    audit_samples=args.unmarked_audit_samples,
                    )
                except Exception as exc:
                    mcq = None
                    reference_audit = {
                        "pass": False,
                        "reason": "unmarked_audit_error",
                        "error": str(exc),
                    }
                if mcq:
                    reference_mode = "unmarked"
                    final_image_path = paths["original"]
            if not mcq:
                mcq = generate_mcq(
                    paths["marked_original"], cand["question"], answer,
                    args.api_url, args.api_key, args.model,
                    answer_vocabulary=cand.get("answer_vocabulary"),
                )
            if not mcq:
                status = "invalid_mcq_generation"
            elif args.reject_model:
                rejection = run_mcq_rejection_sampling(
                    final_image_path,
                    mcq["mcq_question"],
                    mcq["correct_option"],
                    args.reject_api_url or args.api_url,
                    args.reject_api_key or args.api_key,
                    args.reject_model,
                    samples=args.reject_samples,
                    marked=reference_mode == "marked",
                )
                if not rejection.get("sampling_valid", True):
                    status = "invalid_rejection_sampling"
                elif not rejection["keep"]:
                    status = "rejected_too_easy"
                else:
                    status = f"valid_{rejection['difficulty']}"
        # Premium cross-audit runs AFTER rejection sampling so that too-easy
        # candidates (cheaply rejected by the small Qwen3.6 model) never reach
        # the Qwen-VL audit. This is the main cost saver. The audit's acceptance
        # rule is unchanged (mr_vqa.premium_audit); only its position in the
        # pipeline moved here from stage-03. A failed audit flips the status so
        # the candidate is not counted toward max_valid_per_image and is not
        # written to all_valid.
        premium = None
        if status.startswith("valid") and args.premium_audit:
            if not args.premium_api_key:
                premium = {"pass": False, "reason": "premium_api_key_missing"}
            else:
                premium = run_premium_audit(
                    cand, answer, paths["privileged_layout"],
                    args.premium_api_key,
                    qwen_vl_model=args.premium_qwen_vl_model,
                    timeout=args.premium_timeout,
                    retries=args.premium_retries,
                )
            if not premium.get("pass"):
                status = "rejected_premium_audit"
        # Strip designer-written `hypothetical_answer` / `reasoning` when the
        # designer's answer diverges from the voted GT. The designer's
        # hypothetical_answer is unreliable (~63% wrong in smoke testing);
        # we keep `voted_gt` as the canonical GT and retain `reasoning_hops`
        # (process description, no answer) for diagnostics.
        result = {
            **cand,
            "status": status,
            "answer": answer,
            "difficulty": rejection.get("difficulty") if rejection else None,
            "privileged_layout_path": paths["privileged_layout"],
            "marked_image_path": paths["marked_original"],
            "final_image_path": final_image_path,
            "reference_mode": reference_mode,
            "reference_audit": reference_audit,
            "premium_audit": premium,
            "source_question": cand.get("question"),
            "final_question": ((mcq or {}).get("neutral_question")
                               or cand.get("question")),
            "pipeline_version": "mr_vqa_v4_reasoning_certificate",
            "drop_one_paths": paths["drop"],
            "mcq": mcq,
            "rejection_sampling": rejection,
        }
        if not sc.get("designer_hint_matches_voted_gt"):
            result.pop("hypothetical_answer", None)
            result.pop("reasoning", None)
            result["designer_hint_divergent"] = True
        else:
            result["designer_hint_divergent"] = False
        return result
    except Exception as exc:
        return {**cand, "status": "error", "error": str(exc)}


def process_image(rec, args):
    results = []
    valid_count = 0
    valid_lock = threading.Lock()
    stop_event = threading.Event()
    cands = [c for c in rec.get("candidates", []) if not c.get("self_check_filtered")]

    def work(cand):
        if stop_event.is_set():
            return None
        return process_candidate(rec, cand, args)

    with ThreadPoolExecutor(max_workers=max(1, min(args.candidate_workers, len(cands)))) as ex:
        futures = [ex.submit(work, c) for c in cands]
        for fut in as_completed(futures):
            if stop_event.is_set():
                for f in futures:
                    if not f.done():
                        f.cancel()
                break
            res = fut.result()
            if res is None:
                continue
            results.append(res)
            if res.get("status", "").startswith("valid"):
                with valid_lock:
                    valid_count += 1
                    if valid_count >= args.max_valid_per_image:
                        stop_event.set()
    return {**rec, "vqa_results": results}


def main():
    args = parse_args()
    records = list(iter_jsonl(args.candidates_jsonl))
    if args.max_items:
        records = records[:args.max_items]
    if args.resume:
        done = load_completed_ids(args.output_jsonl, key="image_id")
        records = [r for r in records if r["image_id"] not in done]

    if not records:
        print("nothing to do")
        return

    judge = None  # judge no longer used in stage-04 after merge with stage-03 selfcheck

    image_workers = max(1, min(args.image_workers, len(records)))
    print(f"processing {len(records)} images with {image_workers} image workers x {args.candidate_workers} candidate workers")

    with ThreadPoolExecutor(max_workers=image_workers) as ex:
        futures = {ex.submit(process_image, rec, args): rec for rec in records}
        with tqdm(total=len(futures), desc="validate") as pbar:
            for fut in as_completed(futures):
                rec = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    result = {**rec, "vqa_results": [], "error": str(exc)}
                with _file_lock:
                    append_jsonl(result, args.output_jsonl)
                pbar.update(1)

    print(f"wrote validated records to {args.output_jsonl}")


if __name__ == "__main__":
    main()
