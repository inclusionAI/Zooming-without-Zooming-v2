import argparse
import hashlib
from collections import Counter
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from mr_vqa.io_utils import append_jsonl, iter_jsonl, load_completed_ids
from mr_vqa.call_qwen_vl import call_qwen_vl
from mr_vqa.premium_audit import audit_candidate as run_premium_audit
from mr_vqa.questions import (
    answer_kind_matches_declared,
    choose_region_groups,
    classify_final_answer_kind,
    generate_questions,
    has_anchor_attribute_leak,
    judge_question_quality,
    propose_region_groups,
    validate_question_answer_contract,
)
from mr_vqa.selfcheck import self_check_candidate


_file_lock = threading.Lock()


def _candidate_priority_score(item):
    values = []
    for key in ("naturalness_score", "visual_confidence", "multi_region_score"):
        try:
            values.append(float(item.get(key, 3)))
        except (TypeError, ValueError):
            values.append(3.0)
    score = sum(values)
    qtype = item.get("question_type")
    if qtype in {
        "arithmetic_composition", "conditional_composition",
        "boolean_composition", "relational_composition",
        "multi_hop_reasoning", "spatial_chain", "dependent_chain",
        "ocr_visual_verification", "cross_region_pairing",
        "viewpoint_transform", "agent_relative_direction",
        "geometric_line_relation", "topological_relation",
        "depth_ordering", "spatial_transitivity",
        "cross_image_composition",
    }:
        score += 2.0
    elif qtype == "aggregate_count":
        score -= 0.5
    answer_kind = item.get("observed_answer_kind") or item.get("final_answer_kind")
    if item.get("answer_space") == "open":
        score += 2.0
    elif answer_kind in {"semantic", "semantic_set", "semantic_pair", "text"}:
        score += 1.0
    elif answer_kind == "boolean":
        score += 0.5
    elif answer_kind == "numeric":
        score -= 0.25
    elif answer_kind == "anchor":
        score -= 0.5
    score -= 0.35 * len(item.get("local_gate_soft_flags") or [])
    score -= 0.05 * len(item.get("local_gate_repairs") or [])
    return score


def _select_image_candidates(candidates, max_candidates):
    """Keep small sets; above the soft cap, cover types before filling by quality."""
    if max_candidates <= 0:
        return []
    ranked_candidates = sorted(candidates, key=_candidate_priority_score, reverse=True)
    if len(ranked_candidates) <= max_candidates:
        return ranked_candidates

    selected = []
    selected_ids = set()
    selected_types = set()

    # Once the soft cap is exceeded, retain only the best candidate from each
    # reasoning family. Distinct-type coverage may itself exceed the cap.
    for cand in ranked_candidates:
        qtype = cand.get("question_type")
        if qtype in selected_types:
            continue
        selected.append(cand)
        selected_ids.add(cand["candidate_id"])
        selected_types.add(qtype)

    # When fewer than max_candidates distinct types exist, fill the remaining
    # slots with the highest-quality candidates regardless of repeated type.
    if len(selected) < max_candidates:
        for cand in ranked_candidates:
            if cand["candidate_id"] in selected_ids:
                continue
            selected.append(cand)
            selected_ids.add(cand["candidate_id"])
            if len(selected) >= max_candidates:
                break
    return selected


def parse_args():
    parser = argparse.ArgumentParser(description="Generate multi-region candidate questions")
    parser.add_argument("--regions_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--api_url", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen3.5-397B-A17B")
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--min_evidence_regions", type=int, default=3)
    parser.add_argument("--max_evidence_regions", type=int, default=7)
    parser.add_argument("--max_regions_for_groups", type=int, default=10)
    parser.add_argument("--max_groups_per_image", type=int, default=4)
    parser.add_argument("--min_group_spread", type=float, default=0.08)
    parser.add_argument("--questions_per_group", type=int, default=3)
    parser.add_argument("--max_candidates_per_group", type=int, default=4,
                        help="Keep only the highest self-rated candidates per group before costly verification.")
    parser.add_argument("--max_candidates_per_image", type=int, default=8,
                        help="Soft image-level cap. Keep all candidates at or below it; above it, cover every type, then fill to the cap by quality. Type coverage may exceed the cap.")
    parser.add_argument("--selfcheck_candidate_workers", type=int, default=4,
                        help="Parallel candidate workers for self-check within one image.")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="Parallel image workers")
    parser.add_argument("--parallel_groups", action="store_true",
                        help="Also parallelize MLLM calls across region groups within an image")
    parser.add_argument("--use_proposal", action="store_true",
                        help="Use STRONG model to propose region combinations before generating questions.")
    parser.add_argument("--reuse_proposed_groups", action="store_true",
                        help="Reuse proposed_groups embedded in the input records instead of calling the proposal model.")
    parser.add_argument("--proposal_fallback", action="store_true",
                        help="When --use_proposal and the model proposes nothing, fall back to heuristic choose_region_groups.")
    parser.add_argument("--selfcheck", action="store_true",
                        help="Run cross-model privileged answer anchoring and adaptive drop-one necessity checks.")
    parser.add_argument("--secondary_api_key", type=str, default=None,
                        help="API key for the secondary anchor model (Kimi-K2.6).")
    parser.add_argument("--secondary_api_url", type=str, default=None,
                        help="API URL for the secondary anchor model (Kimi-K2.6).")
    parser.add_argument("--secondary_model", type=str, default="Kimi-K2.6",
                        help="Secondary strong multimodal model used as second voter in self-check.")
    parser.add_argument("--selfcheck_primary_samples", type=int, default=2,
                        help="Number of primary-model samples on the privileged view during self-check voting.")
    parser.add_argument("--selfcheck_secondary_samples", type=int, default=2,
                        help="Number of secondary-model samples on the privileged view during self-check voting.")
    parser.add_argument("--selfcheck_min_votes", type=int, default=3,
                        help="Minimum privileged votes (out of primary+secondary) that must match hypothetical_answer.")
    parser.add_argument("--selfcheck_drop_samples", type=int, default=1,
                        help="Initial drop-one samples per region (primary model only).")
    parser.add_argument("--selfcheck_drop_confirm_samples", type=int, default=3,
                        help="Total samples for a suspicious drop-one region after adaptive confirmation.")
    parser.add_argument("--selfcheck_drop_min_matches", type=int, default=2,
                        help="GT-matching ablation answers required to reject a region as unnecessary.")
    parser.add_argument("--selfcheck_drop_fail_threshold", type=int, default=2,
                        help="Number of individually-sufficient regions required to reject a "
                             "candidate. 1 = any single redundant region fails (strict). "
                             "2 = tolerate one redundant region (B-option).")
    parser.add_argument("--selfcheck_workdir", type=str, default=None,
                        help="Optional base directory for self-check evidence images. Defaults to a temp dir.")
    parser.add_argument("--qgen_workdir", type=str, default=None,
                        help="Optional base directory for question-generation privileged layouts. Defaults to a temp dir.")
    parser.add_argument("--reuse_qgen_dir", type=str, default=None,
                        help="Reuse saved <image>/<region-ids>/generation_response.txt files instead of calling the question-generation model.")
    parser.add_argument("--judge_api_url", type=str, default=None,
                        help="Optional API URL for the LLM judge that clusters ambiguous vote answers by semantic equivalence.")
    parser.add_argument("--judge_api_key", type=str, default=None,
                        help="Optional API key for the LLM judge.")
    parser.add_argument("--judge_model", type=str, default="GLM-5.2",
                        help="LLM judge model used to cluster semantically equivalent vote answers when fast _norm equality does not reach the min vote threshold.")
    parser.add_argument("--quality_judge", action="store_true",
                        help="Run a text-only naturalness / multi-region necessity judge after question generation.")
    parser.add_argument("--quality_judge_api_url", type=str, default=None,
                        help="Optional API URL for quality judge. Defaults to --judge_api_url or --api_url.")
    parser.add_argument("--quality_judge_api_key", type=str, default=None,
                        help="Optional API key for quality judge. Defaults to --judge_api_key or --api_key.")
    parser.add_argument("--quality_judge_model", type=str, default=None,
                        help="Quality judge model. Defaults to --judge_model.")
    parser.add_argument("--premium_audit", action="store_true",
                        help="Run Qwen-VL screening after self-check passes.")
    parser.add_argument("--premium_api_key", type=str, default=None,
                        help="Matrix API key for the optional premium audit.")
    parser.add_argument("--premium_qwen_vl_model", type=str, default="Qwen3.5-VL-72B")
    parser.add_argument("--premium_timeout", type=int, default=180)
    parser.add_argument("--premium_retries", type=int, default=2)
    parser.add_argument("--multi_source_generation", action="store_true",
                        help="Generate candidates concurrently with Qwen plus budgeted Qwen-VL calls.")
    parser.add_argument("--generation_api_key", type=str, default=None,
                        help="Matrix key for Qwen-VL question generation.")
    parser.add_argument("--generation_qwen_vl_model", type=str,
                        default="Qwen3.5-VL-72B")
    parser.add_argument("--premium_generation_groups", type=int, default=1,
                        help="Top region groups per image expanded by Qwen-VL.")
    parser.add_argument("--qwen_generation_questions", type=int, default=4)
    parser.add_argument("--premium_generation_timeout", type=int, default=60)
    return parser.parse_args()


def process_one(rec, args):
    def build_fallback_groups():
        groups = choose_region_groups(
            rec["regions"], rec["image_width"], rec["image_height"],
            min_regions=args.min_evidence_regions,
            max_regions=args.max_evidence_regions,
            max_regions_for_groups=args.max_regions_for_groups,
            max_groups=args.max_groups_per_image * 3,
            min_spread=args.min_group_spread,
        )
        if rec.get("evidence_scope") != "cross_image":
            return groups[:args.max_groups_per_image]
        groups = [
            group for group in groups
            if len({r.get("source_image_id") for r in group
                    if r.get("source_image_id")}) >= 2
        ]
        if groups:
            return groups[:args.max_groups_per_image]
        # Deterministic round-robin fallback: at least one region per source
        # image, then fill remaining slots by confidence.
        by_source = {}
        for region in rec["regions"]:
            by_source.setdefault(region.get("source_image_id"), []).append(region)
        ordered_sources = [sid for sid in by_source if sid]
        if len(ordered_sources) < 2:
            return []
        for regions in by_source.values():
            regions.sort(key=lambda r: r.get("grounding_confidence") or 0, reverse=True)
        base = [by_source[sid][0] for sid in ordered_sources[:5]]
        remaining = sorted(
            [r for sid in ordered_sources for r in by_source[sid][1:]],
            key=lambda r: r.get("grounding_confidence") or 0,
            reverse=True,
        )
        group = (base + remaining)[:args.max_evidence_regions]
        return [group] if len(group) >= args.min_evidence_regions else []

    if args.reuse_proposed_groups:
        regions_by_id = {region["region_id"]: region for region in rec["regions"]}
        group_specs = []
        for saved in rec.get("proposed_groups", []):
            region_ids = [str(rid) for rid in saved.get("region_ids", [])]
            if not region_ids or any(rid not in regions_by_id for rid in region_ids):
                continue
            group_specs.append({
                "regions": [regions_by_id[rid] for rid in region_ids],
                "task_family": saved.get("task_family"),
                "necessity_mode": saved.get("necessity_mode"),
                "reason": saved.get("reason", "reused_proposal"),
            })
    elif args.use_proposal:
        group_specs = propose_region_groups(
            rec, args.api_url, args.api_key, args.model,
            min_regions=args.min_evidence_regions,
            max_regions=args.max_evidence_regions,
            max_groups=args.max_groups_per_image,
            max_candidate_regions=args.max_regions_for_groups,
            return_specs=True,
        )
        if not group_specs and args.proposal_fallback:
            fallback_groups = build_fallback_groups()
            group_specs = [{"regions": g, "task_family": None,
                            "necessity_mode": None, "reason": "heuristic_fallback"}
                           for g in fallback_groups]
    else:
        fallback_groups = build_fallback_groups()
        group_specs = [{"regions": g, "task_family": None,
                        "necessity_mode": None, "reason": "heuristic"}
                       for g in fallback_groups]
    rec["proposed_groups"] = [{
        "region_ids": [r["region_id"] for r in spec["regions"]],
        "task_family": spec.get("task_family"),
        "necessity_mode": spec.get("necessity_mode"),
        "reason": spec.get("reason", ""),
    } for spec in group_specs]
    candidates = []
    group_diagnostics = []

    def gen_for_group(group_idx_spec):
        group_idx, spec = group_idx_spec
        group = spec["regions"]
        generation_response_text = None
        if args.reuse_qgen_dir:
            response_path = os.path.join(
                args.reuse_qgen_dir,
                rec["image_id"],
                "|".join(r["region_id"] for r in group),
                "generation_response.txt",
            )
            if not os.path.exists(response_path):
                raise FileNotFoundError(
                    f"missing reusable question response: {response_path}"
                )
            with open(response_path, "r", encoding="utf-8") as f:
                generation_response_text = f.read()
        def run_source(source):
            question_count = args.questions_per_group
            generation_callable = None
            response_text = generation_response_text if source == "qwen" else None
            if source == "qwen_vl":
                question_count = args.qwen_generation_questions

                def generation_callable(prompt, layout_path):
                    return call_qwen_vl(
                        prompt, layout_path,
                        api_key=args.generation_api_key,
                        model=args.generation_qwen_vl_model,
                        temperature=0.6,
                        max_tokens=8192,
                        timeout=args.premium_generation_timeout,
                        retries=4,
                    )
            return generate_questions(
                rec, group, args.api_url, args.api_key, args.model,
                question_count,
                layout_workdir=args.qgen_workdir,
                preferred_task_family=spec.get("task_family"),
                preferred_necessity_mode=spec.get("necessity_mode"),
                return_diagnostics=True,
                generation_response_text=response_text,
                generation_callable=generation_callable,
                generation_source=source,
            )

        sources = ["qwen"]
        premium_group = (
            args.multi_source_generation
            and bool(args.generation_api_key)
            and group_idx < max(0, args.premium_generation_groups)
        )
        if premium_group:
            sources.append("qwen_vl")

        source_results = {}
        source_errors = {}
        if len(sources) == 1:
            source_results["qwen"] = run_source("qwen")
        else:
            # The slowest source overlaps the existing Qwen request instead of
            # extending the group with serial closed-model calls.
            with ThreadPoolExecutor(max_workers=len(sources)) as source_executor:
                futures = {
                    source_executor.submit(run_source, source): source
                    for source in sources
                }
                for future in as_completed(futures):
                    source = futures[future]
                    try:
                        source_results[source] = future.result()
                    except Exception as exc:
                        source_errors[source] = str(exc)

        all_qs = []
        source_diagnostics = {}
        for source in sources:
            result = source_results.get(source)
            if not result:
                continue
            source_qs, diagnostics = result
            all_qs.extend(source_qs)
            source_diagnostics[source] = diagnostics

        # Merge near-identical wording before costly validation. When two
        # sources produce the same question, retain the higher-scored program.
        deduped = {}
        for item in all_qs:
            key = "".join(
                ch for ch in str(item.get("question") or "").lower()
                if ch.isalnum()
            )
            if not key:
                continue
            previous = deduped.get(key)
            if previous is None or _candidate_priority_score(item) > _candidate_priority_score(previous):
                deduped[key] = item
        qs = list(deduped.values())

        def merge_counter(field):
            merged = Counter()
            for diagnostics in source_diagnostics.values():
                merged.update(diagnostics.get(field, {}))
            return dict(merged)

        gate_diagnostics = {
            "raw_count": sum(
                d.get("raw_count", 0) for d in source_diagnostics.values()
            ),
            "accepted_before_limit": len(qs),
            "rejection_counts": merge_counter("rejection_counts"),
            "repair_counts": merge_counter("repair_counts"),
            "soft_flag_counts": merge_counter("soft_flag_counts"),
            "rejection_examples": {},
            "generation_sources_requested": sources,
            "generation_source_counts": dict(Counter(
                item.get("generation_source") or "qwen" for item in qs
            )),
            "generation_source_errors": source_errors,
            "source_details": source_diagnostics,
        }
        for diagnostics in source_diagnostics.values():
            for reason, examples in diagnostics.get("rejection_examples", {}).items():
                bucket = gate_diagnostics["rejection_examples"].setdefault(reason, [])
                bucket.extend(examples[:max(0, 3 - len(bucket))])
        if args.quality_judge and qs:
            qs = judge_question_quality(
                rec, group, qs,
                args.quality_judge_api_url or args.judge_api_url or args.api_url,
                args.quality_judge_api_key or args.judge_api_key or args.api_key,
                args.quality_judge_model or args.judge_model,
            )
        ranked = sorted(qs, key=_candidate_priority_score, reverse=True)
        # Preserve one high-quality candidate per type before filling remaining
        # slots. This prevents aggregate_count from occupying every expensive
        # self-check slot merely because it is easier for the designer model.
        qs = []
        seen_types = set()
        selected_answer_kinds = Counter()
        for item in ranked:
            qtype = item.get("question_type")
            if qtype in seen_types:
                continue
            answer_kind = item.get("observed_answer_kind") or item.get("final_answer_kind")
            if answer_kind == "anchor" and selected_answer_kinds["anchor"] >= 2:
                continue
            if answer_kind == "numeric" and selected_answer_kinds["numeric"] >= 1:
                continue
            qs.append(item)
            seen_types.add(qtype)
            selected_answer_kinds[answer_kind] += 1
            if len(qs) >= args.max_candidates_per_group:
                break
        # Repeated paraphrases consume expensive 4+4 validation calls without
        # improving dataset diversity. Only keep one same-type backup when the
        # group genuinely produced no alternative reasoning family.
        if len(qs) < args.max_candidates_per_group and len(seen_types) == 1:
            selected_ids = {id(x) for x in qs}
            selected_type_counts = Counter(x.get("question_type") for x in qs)
            for item in ranked:
                if id(item) in selected_ids:
                    continue
                qtype = item.get("question_type")
                if selected_type_counts[qtype] >= 2:
                    continue
                answer_kind = item.get("observed_answer_kind") or item.get("final_answer_kind")
                if answer_kind == "anchor" and selected_answer_kinds["anchor"] >= 2:
                    continue
                if answer_kind == "numeric" and selected_answer_kinds["numeric"] >= 1:
                    continue
                qs.append(item)
                selected_ids.add(id(item))
                selected_type_counts[qtype] += 1
                selected_answer_kinds[answer_kind] += 1
                if len(qs) >= min(args.max_candidates_per_group, 2):
                    break
            qs = qs[:args.max_candidates_per_group]
        out = []
        for q_idx, q in enumerate(qs):
            cid_src = (
                f"{rec['image_id']}_{group_idx}_{q_idx}_"
                f"{q.get('generation_source', 'qwen')}_{q.get('question', '')}"
            )
            cand = {
                **q,
                "candidate_id": hashlib.md5(cid_src.encode("utf-8")).hexdigest()[:16],
                "group_index": group_idx,
                "group_task_family": spec.get("task_family"),
                "group_necessity_mode": spec.get("necessity_mode"),
                "evidence_regions": group,
                "evidence_scope": rec.get("evidence_scope", "single_image"),
                "source_images": rec.get("source_images", []),
            }
            out.append(cand)
        gate_diagnostics.update({
            "group_index": group_idx,
            "region_ids": [r["region_id"] for r in group],
            "selected_after_group_limit": len(out),
        })
        return out, gate_diagnostics

    if args.parallel_groups and len(group_specs) > 1:
        with ThreadPoolExecutor(max_workers=min(len(group_specs), 4)) as ex:
            for parts, diagnostics in ex.map(gen_for_group, enumerate(group_specs)):
                candidates.extend(parts)
                group_diagnostics.append(diagnostics)
    else:
        for group_idx, spec in enumerate(group_specs):
            parts, diagnostics = gen_for_group((group_idx, spec))
            candidates.extend(parts)
            group_diagnostics.append(diagnostics)

    generated_before_image_limit = len(candidates)
    candidates = _select_image_candidates(
        candidates, args.max_candidates_per_image
    )

    def run_selfcheck(cand):
        sc = self_check_candidate(
            rec, cand,
            args.api_url, args.api_key, args.model,
            args.secondary_api_url or args.api_url,
            args.secondary_api_key or args.api_key,
            args.secondary_model,
            primary_samples=args.selfcheck_primary_samples,
            secondary_samples=args.selfcheck_secondary_samples,
            min_priv_votes=args.selfcheck_min_votes,
            drop_samples=args.selfcheck_drop_samples,
            drop_confirm_samples=args.selfcheck_drop_confirm_samples,
            drop_min_matches=args.selfcheck_drop_min_matches,
            drop_fail_threshold=args.selfcheck_drop_fail_threshold,
            work_dir=args.selfcheck_workdir,
            judge_api_url=args.judge_api_url,
            judge_api_key=args.judge_api_key,
            judge_model=args.judge_model,
        )
        cand["self_check"] = sc
        if sc.get("pass"):
            voted_gt = sc.get("voted_gt") or ""
            voted_kind = classify_final_answer_kind(
                voted_gt, cand.get("anchor_colors")
            )
            cand["observed_answer_kind"] = voted_kind
            contract_ok, contract_reasons = validate_question_answer_contract(
                cand.get("question"), voted_gt,
                cand.get("final_answer_kind"), cand.get("anchor_colors"),
                answer_semantics=cand.get("answer_semantics", "scalar"),
            )
            if not contract_ok:
                sc["pass"] = False
                sc["reason"] = "voted_question_answer_contract:" + "+".join(contract_reasons)
            elif not answer_kind_matches_declared(
                cand.get("final_answer_kind"), voted_kind
            ):
                sc["pass"] = False
                sc["reason"] = "voted_answer_kind_mismatch"
            elif has_anchor_attribute_leak(
                cand.get("question"), voted_gt, cand.get("anchor_colors")
            ):
                sc["pass"] = False
                sc["reason"] = "voted_anchor_attribute_leak"
            if sc.get("pass") and args.premium_audit:
                if not args.premium_api_key:
                    premium = {
                        "pass": False,
                        "reason": "premium_api_key_missing",
                    }
                else:
                    premium = run_premium_audit(
                        cand,
                        voted_gt,
                        sc["privileged_layout_path"],
                        args.premium_api_key,
                        qwen_vl_model=args.premium_qwen_vl_model,
                        timeout=args.premium_timeout,
                        retries=args.premium_retries,
                    )
                cand["premium_audit"] = premium
                if not premium.get("pass"):
                    sc["pass"] = False
                    sc["reason"] = premium.get(
                        "reason", "premium_cross_audit_failed"
                    )
        if not sc.get("pass"):
            cand["self_check_filtered"] = True
        return cand

    if args.selfcheck and candidates:
        candidate_workers = max(1, min(
            args.selfcheck_candidate_workers, len(candidates)
        ))
        if candidate_workers == 1:
            candidates = [run_selfcheck(cand) for cand in candidates]
        else:
            with ThreadPoolExecutor(max_workers=candidate_workers) as ex:
                candidates = list(ex.map(run_selfcheck, candidates))

    rejection_examples = {}
    for diagnostics in group_diagnostics:
        for reason, examples in diagnostics.get("rejection_examples", {}).items():
            bucket = rejection_examples.setdefault(reason, [])
            bucket.extend(examples[:max(0, 3 - len(bucket))])

    aggregate_gate_diagnostics = {
        "groups": len(group_diagnostics),
        "raw_count": sum(d.get("raw_count", 0) for d in group_diagnostics),
        "accepted_before_group_limit": sum(
            d.get("accepted_before_limit", 0) for d in group_diagnostics
        ),
        "selected_after_group_limit": generated_before_image_limit,
        "selected_after_image_limit": len(candidates),
        "rejected_by_image_limit": max(0, generated_before_image_limit - len(candidates)),
        "rejection_counts": dict(sum(
            (Counter(d.get("rejection_counts", {})) for d in group_diagnostics),
            Counter(),
        )),
        "repair_counts": dict(sum(
            (Counter(d.get("repair_counts", {})) for d in group_diagnostics),
            Counter(),
        )),
        "soft_flag_counts": dict(sum(
            (Counter(d.get("soft_flag_counts", {})) for d in group_diagnostics),
            Counter(),
        )),
        "rejection_examples": rejection_examples,
        "group_details": sorted((
            {key: value for key, value in diagnostics.items()
             if key != "rejection_examples"}
            for diagnostics in group_diagnostics
        ), key=lambda d: d.get("group_index", -1)),
    }

    return {
        **rec,
        "candidate_status": "ok",
        "local_gate_diagnostics": aggregate_gate_diagnostics,
        "candidate_type_counts": dict(Counter(
            c.get("question_type") or "unknown" for c in candidates
        )),
        "candidate_template_counts": dict(Counter(
            c.get("template_id") or "freeform" for c in candidates
        )),
        "candidate_answer_kind_counts": dict(Counter(
            c.get("observed_answer_kind") or c.get("final_answer_kind") or "unknown"
            for c in candidates
        )),
        "candidate_generation_source_counts": dict(Counter(
            c.get("generation_source") or "qwen" for c in candidates
        )),
        "candidate_certificate_mode_counts": dict(Counter(
            ((c.get("reasoning_certificate") or {}).get("mode") or "unknown")
            for c in candidates
        )),
        "selfcheck_pass_type_counts": dict(Counter(
            c.get("question_type") or "unknown" for c in candidates
            if (c.get("self_check") or {}).get("pass")
        )),
        "selfcheck_pass_template_counts": dict(Counter(
            c.get("template_id") or "freeform" for c in candidates
            if (c.get("self_check") or {}).get("pass")
        )),
        "selfcheck_pass_answer_kind_counts": dict(Counter(
            c.get("observed_answer_kind") or c.get("final_answer_kind") or "unknown"
            for c in candidates if (c.get("self_check") or {}).get("pass")
        )),
        "selfcheck_pass_generation_source_counts": dict(Counter(
            c.get("generation_source") or "qwen" for c in candidates
            if (c.get("self_check") or {}).get("pass")
        )),
        "selfcheck_pass_certificate_mode_counts": dict(Counter(
            ((c.get("reasoning_certificate") or {}).get("mode") or "unknown")
            for c in candidates if (c.get("self_check") or {}).get("pass")
        )),
        "premium_audit_counts": dict(Counter(
            ("pass" if (c.get("premium_audit") or {}).get("pass") else "fail")
            for c in candidates if c.get("premium_audit")
        )),
        "candidates": candidates,
    }


def main():
    args = parse_args()
    records = [r for r in iter_jsonl(args.regions_jsonl) if r.get("status") == "ok" and len(r.get("regions", [])) >= 2]
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
        with tqdm(total=len(futures), desc="generate questions") as pbar:
            for fut in as_completed(futures):
                rec = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    result = {**rec, "candidate_status": "error", "error": str(exc), "candidates": []}
                with _file_lock:
                    append_jsonl(result, args.output_jsonl)
                pbar.update(1)

    print(f"wrote candidates to {args.output_jsonl}")


if __name__ == "__main__":
    main()
