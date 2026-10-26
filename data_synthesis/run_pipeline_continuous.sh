#!/usr/bin/env bash
# Continuous multi-region VQA pipeline runner.
#
# Reads a manifest (data/filtered_manifest.jsonl, produced by
# scripts/01_filter_manifest.py) sequentially; each batch takes the next
# BATCH_SIZE rows. State = next manifest row index.
#
# Stops when EITHER:
#   - data/all_valid.jsonl reaches TARGET_VALID records, OR
#   - MAX_BATCHES batches have been run (MAX_BATCHES=0 => no cap)
#   - manifest runs out of rows
#
# Resume:
#   - data/pipeline_state.txt: single integer = next manifest line index
#   - data/processed_image_ids.txt: every image_id that entered the pipeline
#     (one per line, appended as the batch starts). Useful for audit/dedup.
#   - Stage-level: each batch keeps manifest/regions/candidates/validated;
#     reruns use --resume to skip already-done images within the batch.
#   - Failed images: appended to data/failed_images.jsonl with batch id,
#     image_id, last stage attempted, and reason.
#
# Usage:
#   bash run_pipeline_continuous.sh [TARGET_VALID] [BATCH_SIZE] [MAX_BATCHES] [MANIFEST]
#   TARGET_VALID default 10000
#   BATCH_SIZE   default 10
#   MAX_BATCHES  default 0 (== unbounded; stop on TARGET_VALID only)
#   MANIFEST     default data/filtered_manifest.jsonl
set -uo pipefail

cd "$(dirname "$0")"
source configs/local_env.sh

TARGET_VALID="${1:-50000}"
BATCH_SIZE="${2:-10}"
MAX_BATCHES="${3:-0}"
MANIFEST="${4:-data/filtered_manifest.jsonl}"

STATE_FILE="data/pipeline_state.txt"
ALL_VALID="data/all_valid.jsonl"
FAILED_IMAGES="data/failed_images.jsonl"
PROCESSED_IDS="data/processed_image_ids.txt"
BATCH_ROOT="data/batch_runs"
mkdir -p "$BATCH_ROOT" "$(dirname "$ALL_VALID")" \
         "$(dirname "$STATE_FILE")" "$(dirname "$FAILED_IMAGES")" \
         "$(dirname "$PROCESSED_IDS")"

# Premium Qwen-VL models serve two distinct roles:
#   - GENERATION (stage 03): a budgeted Qwen-VL slice adds question
#     diversity on top of the Qwen primary source.
#   - AUDIT (stage 04): Qwen-VL cross-audit, now run AFTER rejection sampling
#     so too-easy candidates (cheaply rejected by the small Qwen3.6) never
#     spend the Qwen-VL audit budget.
# Splitting the arg groups lets stage 03 do generation-only and stage 04 do
# audit-only, matching the relocated audit step in 04_validate_questions.py.
PREMIUM_GENERATION_ARGS=()
PREMIUM_AUDIT_ARGS=()
if [[ -n "${MATRIX_API_KEY:-}" ]]; then
  PREMIUM_GENERATION_ARGS=(
    --multi_source_generation
    --generation_api_key "$MATRIX_API_KEY"
    --generation_qwen_vl_model "${PREMIUM_QWEN_VL_MODEL:-Qwen3.5-VL-72B}"
    --premium_generation_groups "${PREMIUM_GENERATION_GROUPS:-1}"
    --qwen_generation_questions "${QWEN_VL_GENERATION_QUESTIONS:-4}"
    --premium_generation_timeout "${PREMIUM_GENERATION_TIMEOUT:-60}"
  )
  PREMIUM_AUDIT_ARGS=(
    --premium_audit
    --premium_api_key "$MATRIX_API_KEY"
    --premium_qwen_vl_model "${PREMIUM_QWEN_VL_MODEL:-Qwen3.5-VL-72B}"
  )
fi

# Minimum free space in MB on the working filesystem; abort if below.
MIN_FREE_MB="${MIN_FREE_MB:-5120}"
free_space_mb() {
  df -Pk "$1" | awk 'NR==2 {printf "%d\n", $4 / 1024}'
}
WORK_DIR_FS="$(free_space_mb "$(dirname "$BATCH_ROOT")")"
if [[ "$WORK_DIR_FS" -lt "$MIN_FREE_MB" ]] 2>/dev/null; then
  echo "[pipeline] free disk ${WORK_DIR_FS}MB below MIN_FREE_MB=${MIN_FREE_MB}MB threshold; abort."
  exit 2
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "[pipeline] manifest not found: $MANIFEST"
  echo "[pipeline] generate it first with: python scripts/01_filter_manifest.py --image_folder <your_images> --output_jsonl $MANIFEST"
  exit 1
fi
MANIFEST_TOTAL=$(wc -l < "$MANIFEST" | tr -d ' ')

# Read state: next manifest index to consume
if [[ -f "$STATE_FILE" ]]; then
  read -r START_IDX < "$STATE_FILE"
  if [[ -z "$START_IDX" ]]; then START_IDX=0; fi
else
  START_IDX=0
fi

count_valid() {
  if [[ -f "$ALL_VALID" ]]; then
    wc -l < "$ALL_VALID" | tr -d ' '
  else
    echo 0
  fi
}

batch_done=0
while true; do
  # Per-batch disk check: bail out if free space dropped below threshold.
  CUR_FREE_MB="$(free_space_mb "$(dirname "$BATCH_ROOT")")"
  if [[ "$CUR_FREE_MB" -lt "$MIN_FREE_MB" ]] 2>/dev/null; then
    echo "[pipeline] free disk ${CUR_FREE_MB}MB < MIN_FREE_MB=${MIN_FREE_MB}MB before batch; aborting (state preserved at idx=$START_IDX)."
    exit 3
  fi

  CUR_VALID=$(count_valid)
  if [[ "$CUR_VALID" -ge "$TARGET_VALID" ]]; then
    echo "[pipeline] reached TARGET_VALID=$TARGET_VALID (current=$CUR_VALID). stopping."
    break
  fi
  if [[ "$MAX_BATCHES" -gt 0 && "$batch_done" -ge "$MAX_BATCHES" ]]; then
    echo "[pipeline] reached MAX_BATCHES=$MAX_BATCHES. stopping."
    break
  fi
  if [[ "$START_IDX" -ge "$MANIFEST_TOTAL" ]]; then
    echo "[pipeline] manifest exhausted (idx=$START_IDX / total=$MANIFEST_TOTAL). stopping."
    break
  fi

  END_IDX=$((START_IDX + BATCH_SIZE))
  if [[ "$END_IDX" -gt "$MANIFEST_TOTAL" ]]; then
    END_IDX=$MANIFEST_TOTAL
  fi
  THIS_BATCH=$((END_IDX - START_IDX))

  BATCH_ID=$(date +"%Y%m%d_%H%M%S")
  RUN_DIR="$BATCH_ROOT/run_$BATCH_ID"
  mkdir -p "$RUN_DIR"

  echo "================================================================"
  echo "[pipeline] batch $((batch_done + 1)) | run_dir=$RUN_DIR | manifest[$START_IDX:$END_IDX] ($THIS_BATCH rows) | valid_so_far=$CUR_VALID/$TARGET_VALID | total_manifest=$MANIFEST_TOTAL"
  echo "================================================================"

  # ----- Slice BATCH_SIZE rows from manifest -> RUN_DIR/manifest.jsonl -----
  python - "$MANIFEST" "$RUN_DIR/manifest.jsonl" "$START_IDX" "$END_IDX" <<'PY'
import json, sys
src, dst, lo, hi = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with open(src, "r", encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
    for i, line in enumerate(f):
        if i >= hi:
            break
        if i >= lo:
            obj = json.loads(line)
            # keep only fields downstream stages expect
            rec = {
                "image_id": obj.get("image_id"),
                "image_path": obj.get("image_path"),
                "caption": obj.get("caption", ""),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"slice [{lo}:{hi}] -> {dst}")
PY

  # Track which images entered this batch (failure accounting + audit)
  python - "$RUN_DIR/manifest.jsonl" "$RUN_DIR/image_ids.txt" "$PROCESSED_IDS" <<'PY'
import json, sys
src, ids_file, processed_file = sys.argv[1], sys.argv[2], sys.argv[3]
ids = []
with open(src, "r", encoding="utf-8") as f, \
     open(ids_file, "w", encoding="utf-8") as ids_out, \
     open(processed_file, "a", encoding="utf-8") as proc_out:
    for line in f:
        obj = json.loads(line)
        iid = obj.get("image_id")
        if iid:
            ids.append(iid)
            ids_out.write(iid + "\n")
            proc_out.write(iid + "\n")
print(f"wrote {len(ids)} image_ids")
PY

  # ----- Stage 02: extract regions -----
  python scripts/02_extract_regions.py \
    --manifest_jsonl "$RUN_DIR/manifest.jsonl" \
    --output_jsonl "$RUN_DIR/regions.jsonl" \
    --api_key "$MLLM_KEY" \
    --api_url "$MLLM_URL" \
    --model "$STRONG_MODEL" \
    --resume \
    > "$RUN_DIR/02.log" 2>&1
  stage02_rc=$?
  if [[ $stage02_rc -eq 0 && ! -s "$RUN_DIR/regions.jsonl" ]]; then
    CUR_FREE_MB="$(free_space_mb "$(dirname "$BATCH_ROOT")")"
    echo "[pipeline] stage02 rc=0 but regions.jsonl empty (likely disk full; free=${CUR_FREE_MB}MB). aborting at idx=$START_IDX."
    exit 4
  fi

  # ----- Stage 03: generate questions + merged selfcheck -----
  stage03_rc=1
  if [[ $stage02_rc -eq 0 ]]; then
    python scripts/03_generate_questions.py \
      --regions_jsonl "$RUN_DIR/regions.jsonl" \
      --output_jsonl "$RUN_DIR/candidates.jsonl" \
      --api_key "$MLLM_KEY" \
      --api_url "$MLLM_URL" \
      --model "$STRONG_MODEL" \
      --use_proposal \
      --max_evidence_regions 7 \
      --max_regions_for_groups 18 \
      --max_groups_per_image 6 \
      --questions_per_group 8 \
      --max_candidates_per_group 4 \
      --max_candidates_per_image 8 \
      --selfcheck_candidate_workers 6 \
      --max_workers 10 \
      --parallel_groups \
      --selfcheck \
      --secondary_api_url "$MLLM_URL" \
      --secondary_api_key "$MLLM_KEY" \
      --secondary_model "$SECONDARY_MODEL" \
      --selfcheck_primary_samples 4 \
      --selfcheck_secondary_samples 4 \
      --selfcheck_min_votes 6 \
      --selfcheck_drop_samples 1 \
      --selfcheck_drop_confirm_samples 3 \
      --selfcheck_drop_min_matches 2 \
      --selfcheck_drop_fail_threshold 2 \
      --selfcheck_workdir "$RUN_DIR/selfcheck_evidence" \
      --qgen_workdir "$RUN_DIR/qgen_evidence" \
      --judge_api_url "$MLLM_URL" \
      --judge_api_key "$MLLM_KEY" \
      --judge_model "$JUDGE_MODEL" \
      ${PREMIUM_GENERATION_ARGS[@]+"${PREMIUM_GENERATION_ARGS[@]}"} \
      --resume \
      > "$RUN_DIR/03.log" 2>&1
    stage03_rc=$?
    if [[ $stage03_rc -eq 0 && ! -s "$RUN_DIR/candidates.jsonl" ]]; then
      CUR_FREE_MB="$(free_space_mb "$(dirname "$BATCH_ROOT")")"
      echo "[pipeline] stage03 rc=0 but candidates.jsonl empty (likely disk full; free=${CUR_FREE_MB}MB). aborting at idx=$START_IDX."
      exit 5
    fi
  fi

  # ----- Stage 04: MCQ + rejection sampling -----
  stage04_rc=1
  if [[ $stage03_rc -eq 0 ]]; then
    python scripts/04_validate_questions.py \
      --candidates_jsonl "$RUN_DIR/candidates.jsonl" \
      --output_jsonl "$RUN_DIR/validated.jsonl" \
      --evidence_dir "$RUN_DIR/validation_evidence" \
      --api_key "$MLLM_KEY" \
      --api_url "$MLLM_URL" \
      --model "$STRONG_MODEL" \
      --secondary_api_url "$MLLM_URL" \
      --secondary_api_key "$MLLM_KEY" \
      --secondary_model "$SECONDARY_MODEL" \
      --reject_api_url "$MLLM_URL" \
      --reject_api_key "$MLLM_KEY" \
      --reject_model "$REJECT_MODEL" \
      --reject_samples 4 \
      --image_workers 4 \
      --candidate_workers 8 \
      ${PREMIUM_AUDIT_ARGS[@]+"${PREMIUM_AUDIT_ARGS[@]}"} \
      > "$RUN_DIR/04.log" 2>&1
    stage04_rc=$?
  fi

  # ----- Collect valid cases + detailed per-stage accounting -----
  python - "$RUN_DIR" "$ALL_VALID" "$FAILED_IMAGES" "$BATCH_ID" \
          "$stage02_rc" "$stage03_rc" "$stage04_rc" <<'PY'
import json, os, sys
from collections import defaultdict

run_dir, all_valid, failed_path, batch_id, r2, r3, r4 = sys.argv[1:8]

iid_file = os.path.join(run_dir, "image_ids.txt")
image_ids = []
if os.path.exists(iid_file):
    image_ids = [l.strip() for l in open(iid_file) if l.strip()]

# --- load stage outputs to build per-image accounting ---
# stage 02 -> regions per image
regions_path = os.path.join(run_dir, "regions.jsonl")
img_regions = {}  # iid -> {"status": ..., "count": int}
if os.path.exists(regions_path):
    for r in (json.loads(l) for l in open(regions_path)):
        iid = r.get("image_id")
        if iid is None:
            continue
        regs = r.get("regions", []) or []
        img_regions[iid] = {"status": r.get("status"),
                             "count": len(regs)}

# stage 03 -> candidates per image + selfcheck + drop-one
cands_path = os.path.join(run_dir, "candidates.jsonl")
img_cands = {}  # iid -> list of candidate dicts (with self_check, group_index, self_check_filtered)
if os.path.exists(cands_path):
    for r in (json.loads(l) for l in open(cands_path)):
        iid = r.get("image_id")
        if iid is None:
            continue
        img_cands[iid] = r.get("candidates", []) or []

# stage 04 -> validated vqa_results per image
val_path = os.path.join(run_dir, "validated.jsonl")
img_results = {}
if os.path.exists(val_path):
    for r in (json.loads(l) for l in open(val_path)):
        iid = r.get("image_id")
        results = r.get("vqa_results", [])
        img_results[iid] = results

# --- aggregate + append valid cases ---
added = 0
per_image = []
totals = {"input_images": len(image_ids),
          "regions_extracted": 0, "images_no_regions": 0,
          "groups_proposed": 0, "candidates_total": 0,
          "candidate_question_type": defaultdict(int),
          "candidate_template": defaultdict(int),
          "candidate_answer_kind": defaultdict(int),
          "candidate_generation_source": defaultdict(int),
          "selfcheck_pass_question_type": defaultdict(int),
          "selfcheck_pass_template": defaultdict(int),
          "selfcheck_pass_answer_kind": defaultdict(int),
          "selfcheck_passed": 0, "selfcheck_filtered": 0,
          "after_selfcheck": 0,
          "premium_audit_run": 0, "premium_audit_passed": 0,
          "premium_audit_failed": 0, "premium_audit_errors": 0,
          "premium_routes": defaultdict(int),
          "premium_wall_seconds": 0.0,
          "premium_qwen_vl_seconds": 0.0,
          "mcq_generated": 0, "mcq_failed": 0,
          "after_mcq": 0,
          "rejection_run": 0, "rejected_too_easy": 0,
          "valid_total": 0, "valid_difficulty": defaultdict(int),
          "valid_question_type": defaultdict(int),
          "valid_template": defaultdict(int),
          "valid_answer_kind": defaultdict(int),
          "valid_generation_source": defaultdict(int),
          "valid_reference_mode": defaultdict(int),
          "selfcheck_fail_reasons": defaultdict(int),
          "drop_one_fail_reasons": defaultdict(int)}

with open(all_valid, "a", encoding="utf-8") as out_f:
    for iid in image_ids:
        reg = img_regions.get(iid, {})
        reg_count = reg.get("count", 0)
        if reg_count == 0:
            totals["images_no_regions"] += 1
        totals["regions_extracted"] += reg_count

        cands = img_cands.get(iid, [])
        # group_index dedup to count distinct groups
        group_ids = {c.get("group_index") for c in cands}
        totals["groups_proposed"] += len(group_ids)
        totals["candidates_total"] += len(cands)

        sc_pass = 0
        sc_filtered = 0
        sc_fail_reasons = defaultdict(int)
        drop_fail_reasons = defaultdict(int)
        premium_run = 0
        premium_pass = 0
        for c in cands:
            qtype = c.get("question_type") or "unknown"
            template_id = c.get("template_id") or "freeform"
            answer_kind = c.get("observed_answer_kind") or c.get("final_answer_kind") or "unknown"
            generation_source = c.get("generation_source") or "qwen"
            totals["candidate_question_type"][qtype] += 1
            totals["candidate_template"][template_id] += 1
            totals["candidate_answer_kind"][answer_kind] += 1
            totals["candidate_generation_source"][generation_source] += 1
            # NOTE: premium_audit is no longer recorded on stage-03 candidates.
            # It now lives on stage-04 vqa_results (validated.jsonl) because the
            # audit runs after rejection sampling. Tallied in the results loop
            # below.
            sc = c.get("self_check") or {}
            if sc.get("pass"):
                sc_pass += 1
                totals["selfcheck_pass_question_type"][qtype] += 1
                totals["selfcheck_pass_template"][template_id] += 1
                totals["selfcheck_pass_answer_kind"][answer_kind] += 1
            else:
                sc_filtered += 1
                sc_fail_reasons[sc.get("reason", "unknown")] += 1
                if sc.get("reason", "").startswith("drop_one_"):
                    drop_fail_reasons[sc.get("reason")] += 1
        totals["selfcheck_passed"] += sc_pass
        totals["selfcheck_filtered"] += sc_filtered
        totals["after_selfcheck"] += sc_pass
        for k, v in sc_fail_reasons.items():
            totals["selfcheck_fail_reasons"][k] += v
        for k, v in drop_fail_reasons.items():
            totals["drop_one_fail_reasons"][k] += v

        results = img_results.get(iid, [])
        mcq_ok = 0
        mcq_fail = 0
        rej_run = 0
        rej_easy = 0
        valid = 0
        diff = defaultdict(int)
        for c in results:
            status = str(c.get("status", ""))
            if c.get("mcq"):
                mcq_ok += 1
            if status == "invalid_mcq_generation":
                mcq_fail += 1
            if c.get("rejection_sampling"):
                rej_run += 1
            if status == "rejected_too_easy":
                rej_easy += 1
            # Premium audit now runs in stage-04 (after rejection sampling),
            # so its result is on the vqa_result, not the stage-03 candidate.
            premium = c.get("premium_audit") or {}
            if premium:
                premium_run += 1
                totals["premium_audit_run"] += 1
                totals["premium_routes"][premium.get("route") or "unknown"] += 1
                totals["premium_wall_seconds"] += float(
                    premium.get("wall_latency_seconds") or 0
                )
                totals["premium_qwen_vl_seconds"] += float(
                    (premium.get("qwen_vl") or {}).get("latency_seconds") or 0
                )
                if premium.get("pass"):
                    premium_pass += 1
                    totals["premium_audit_passed"] += 1
                else:
                    totals["premium_audit_failed"] += 1
                qwen_vl_entry = premium.get("qwen_vl") or {}
                totals["premium_audit_errors"] += sum(
                    bool(qwen_vl_entry.get("error"))
                    or bool(qwen_vl_entry.get("parse_error"))
                )
            if status.startswith("valid"):
                valid += 1
                d = c.get("difficulty") or "unknown"
                diff[d] += 1
                totals["valid_question_type"][c.get("question_type") or "unknown"] += 1
                totals["valid_template"][c.get("template_id") or "freeform"] += 1
                totals["valid_answer_kind"][
                    c.get("observed_answer_kind") or c.get("final_answer_kind") or "unknown"
                ] += 1
                totals["valid_generation_source"][
                    c.get("generation_source") or "qwen"
                ] += 1
                totals["valid_reference_mode"][c.get("reference_mode") or "marked"] += 1
                # write slim record — keep rejection_sampling for audit HTML;
                # only drop heavy per-region path blobs.
                slim = {k: v for k, v in c.items()
                        if k not in ("drop_one_paths", "validation")}
                slim["image_id"] = iid
                slim["batch_run"] = os.path.basename(run_dir)
                # Consolidate the two final images into a stable shared dir
                # (data/valid_evidence/<image_id>/<candidate_id>/) so the batch
                # directory can be deleted wholesale afterward. Rewrite the
                # path fields so downstream consumers point at the consolidated
                # copies, not the transient per-batch evidence tree.
                import shutil as _shutil
                _vid = slim.get("image_id") or ""
                _cid = slim.get("candidate_id") or "cand"
                _cons = os.path.join("data", "valid_evidence", str(_vid), str(_cid))
                os.makedirs(_cons, exist_ok=True)
                _path_fields = ("marked_image_path", "privileged_layout_path", "final_image_path")
                for _pf in _path_fields:
                    _src = slim.get(_pf)
                    if not _src or not os.path.exists(_src):
                        continue
                    _dst = os.path.join(_cons, os.path.basename(_src))
                    try:
                        if os.path.abspath(_src) != os.path.abspath(_dst):
                            _shutil.copy2(_src, _dst)
                        slim[_pf] = _dst
                    except OSError as _e:
                        print(f"[consolidate] copy failed {_src} -> {_dst}: {_e}", flush=True)
                out_f.write(json.dumps(slim, ensure_ascii=False) + "\n")
                added += 1
        totals["mcq_generated"] += mcq_ok
        totals["mcq_failed"] += mcq_fail
        totals["after_mcq"] += mcq_ok
        totals["rejection_run"] += rej_run
        totals["rejected_too_easy"] += rej_easy
        totals["valid_total"] += valid
        for k, v in diff.items():
            totals["valid_difficulty"][k] += v

        # per-image row for the table
        any_valid = valid > 0
        any_sc_pass = sc_pass > 0
        if not results:
            if r2 != "0": terminal = "stopped_at_stage02"
            elif r3 != "0": terminal = "stopped_at_stage03"
            elif r4 != "0": terminal = "stopped_at_stage04"
            elif not cands: terminal = "no_candidates"
            elif not any_sc_pass: terminal = "all_filtered_at_stage03"
            else: terminal = "no_stage04_results"
        elif any_valid:
            terminal = "valid"
        elif any_sc_pass:
            terminal = "stage04_fail"
        else:
            terminal = "stage03_fail"
        per_image.append({
            "image_id": iid,
            "regions": reg_count,
            "candidates": len(cands),
            "selfcheck_pass": sc_pass,
            "premium_run": premium_run,
            "premium_pass": premium_pass,
            "mcq_ok": mcq_ok,
            "valid": valid,
            "terminal": terminal,
        })

# --- write batch_summary.json ---
summary = {
    "batch_id": batch_id,
    "stage_return_codes": {"stage02": r2, "stage03": r3, "stage04": r4},
    "totals": {k: (dict(v) if isinstance(v, defaultdict) else v)
               for k, v in totals.items()},
    "per_image": per_image,
}
summary_path = os.path.join(run_dir, "batch_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# --- mark failed images ---
last_stage = "stage00b"
if r2 == "0":
    last_stage = "stage02"
if r3 == "0":
    last_stage = "stage03"
if r4 == "0":
    last_stage = "stage04"
with open(failed_path, "a", encoding="utf-8") as fr:
    for row in per_image:
        iid = row["image_id"]
        if row["terminal"] == "valid":
            stage, reason = "valid", "ok"
        elif row["terminal"] == "stage04_fail":
            results = img_results.get(iid, [])
            reasons = sorted({c.get("status", "") for c in results})
            stage, reason = "stage04", ";".join(reasons)
        elif row["terminal"] == "stage03_fail":
            cands = img_cands.get(iid, [])
            reasons = sorted({(c.get("self_check") or {}).get("reason", "unknown")
                              for c in cands})
            stage, reason = "stage03", ";".join(reasons)
        else:
            stage, reason = last_stage, f"stopped_at_{last_stage}"
        fr.write(json.dumps({
            "image_id": iid,
            "batch_id": batch_id,
            "last_stage": stage,
            "reason": reason,
            "candidate_count": row["candidates"],
            "regions": row["regions"],
        }, ensure_ascii=False) + "\n")

# --- print compact per-image table + totals ---
print(f"\n[batch {batch_id}] per-image breakdown:")
print(f"  {'image_id':<18} {'reg':>4} {'cand':>5} {'sc_pass':>7} {'mcq':>4} {'valid':>5}  terminal")
print(f"  {'-'*18} {'-'*4} {'-'*5} {'-'*7} {'-'*4} {'-'*5}  {'-'*20}")
for row in per_image:
    print(f"  {row['image_id']:<18} {row['regions']:>4} {row['candidates']:>5} "
          f"{row['selfcheck_pass']:>7} {row['mcq_ok']:>4} {row['valid']:>5}  {row['terminal']}")

print(f"\n[batch {batch_id}] totals:")
t = totals
print(f"  input_images        = {t['input_images']}")
print(f"  regions_extracted   = {t['regions_extracted']}  (no_regions_img={t['images_no_regions']})")
print(f"  groups_proposed     = {t['groups_proposed']}")
print(f"  candidates_total    = {t['candidates_total']}")
print(f"    candidate_types   = {dict(t['candidate_question_type'])}")
print(f"    candidate_templates = {dict(t['candidate_template'])}")
print(f"    candidate_answer_kinds = {dict(t['candidate_answer_kind'])}")
print(f"    generation_sources = {dict(t['candidate_generation_source'])}")
print(f"  selfcheck_passed    = {t['selfcheck_passed']}  -> filtered={t['selfcheck_filtered']}")
print(f"    pass_types        = {dict(t['selfcheck_pass_question_type'])}")
print(f"    pass_templates    = {dict(t['selfcheck_pass_template'])}")
print(f"    pass_answer_kinds = {dict(t['selfcheck_pass_answer_kind'])}")
print(f"    fail_reasons      = {dict(t['selfcheck_fail_reasons'])}")
print(f"    drop_one_reasons  = {dict(t['drop_one_fail_reasons'])}")
print(f"  after_selfcheck     = {t['after_selfcheck']}")
if t['premium_audit_run']:
    avg_premium = t['premium_wall_seconds'] / t['premium_audit_run']
    print(f"  premium_audit       = {t['premium_audit_passed']}/{t['premium_audit_run']} passed "
          f"(failed={t['premium_audit_failed']}, errors={t['premium_audit_errors']}, "
          f"avg_wall={avg_premium:.1f}s)")
    print(f"    model_latency_sum = qwen_vl:{t['premium_qwen_vl_seconds']:.1f}s")
    print(f"    routes            = {dict(t['premium_routes'])}")
print(f"  mcq_generated       = {t['mcq_generated']}  (failed={t['mcq_failed']})")
print(f"  after_mcq           = {t['after_mcq']}")
print(f"  rejection_run       = {t['rejection_run']}  rejected_too_easy={t['rejected_too_easy']}")
print(f"  valid_total         = {t['valid_total']}  added_to_all_valid={added}")
if t['valid_difficulty']:
    print(f"  valid_difficulty    = {dict(t['valid_difficulty'])}")
if t['valid_question_type']:
    print(f"  valid_question_type = {dict(t['valid_question_type'])}")
if t['valid_template']:
    print(f"  valid_template      = {dict(t['valid_template'])}")
if t['valid_answer_kind']:
    print(f"  valid_answer_kind   = {dict(t['valid_answer_kind'])}")
if t['valid_generation_source']:
    print(f"  valid_gen_source    = {dict(t['valid_generation_source'])}")
if t['valid_reference_mode']:
    print(f"  reference_mode      = {dict(t['valid_reference_mode'])}")
print(f"  summary written to  = {summary_path}")

# --- Disk cleanup: valid final images have already been consolidated into
# data/valid_evidence/<image_id>/<candidate_id>/ by the append loop above. The
# whole batch directory (manifest/regions/candidates/validated + every
# evidence subdir: qgen/selfcheck/validation) is now dead weight — drop it
# entirely so long-term storage stays bounded. Only batch_summary.json is
# rescued to data/batch_runs/_summaries/ for audit.
import shutil
summary_rescue_dir = os.path.join("data", "batch_runs", "_summaries")
os.makedirs(summary_rescue_dir, exist_ok=True)
batch_id = os.path.basename(run_dir)
try:
    shutil.copy2(summary_path, os.path.join(summary_rescue_dir, f"{batch_id}.json"))
except OSError as exc:
    print(f"[cleanup] failed to rescue summary: {exc}")

removed_bytes = 0
for root, _dirs, files in os.walk(run_dir):
    for fn in files:
        try:
            removed_bytes += os.path.getsize(os.path.join(root, fn))
        except OSError:
            pass
try:
    shutil.rmtree(run_dir)
    print(f"\n[cleanup] deleted batch dir {batch_id} (~{removed_bytes/1024/1024:.0f} MB) "
          f"after consolidating valid images into data/valid_evidence/")
except OSError as exc:
    print(f"[cleanup] failed to remove {run_dir}: {exc}")
PY

  # ----- Advance state -----
  START_IDX=$END_IDX
  echo "$START_IDX" > "$STATE_FILE"
  CUR_VALID=$(count_valid)
  PROCESSED_COUNT=$(wc -l < "$PROCESSED_IDS" | tr -d ' ')
  echo "[pipeline] batch_done=$((batch_done + 1)) valid_total=$CUR_VALID/$TARGET_VALID next_idx=$START_IDX/$MANIFEST_TOTAL processed_total=$PROCESSED_COUNT"
  batch_done=$((batch_done + 1))
done

echo "[pipeline] finished. Total valid: $(count_valid) (target was $TARGET_VALID)"
echo "  Valid samples:    $ALL_VALID"
echo "  Failed images:    $FAILED_IMAGES"
echo "  Processed ids:    $PROCESSED_IDS"
echo "  State file:       $STATE_FILE (next idx=$START_IDX / total=$MANIFEST_TOTAL)"
