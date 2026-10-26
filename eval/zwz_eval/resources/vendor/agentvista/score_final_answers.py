# Portable copy of the actual final-answer postjudge; only its import root is changed.
import argparse
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser()
parser.add_argument("roots", nargs="+")
args = parser.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.general_qa_tool import GPT4VisionClient, SYSTEM_PROMPT, QUERY_PROMPT

def judge(path):
    destination = path.parent / "judge.json"
    if destination.exists():
        return json.loads(destination.read_text())
    result = json.loads(path.read_text())
    answer = result.get("final_answer", "")
    if not answer or answer.startswith("Error:"):
        return {"question_id": result["question_id"], "status": "infer_failed", "score": None}
    query = QUERY_PROMPT.format(question=result["prompt"], ground_truth=result["ground_truth"], prediction=answer)
    score, explanation = GPT4VisionClient().query(query, system_prompt=SYSTEM_PROMPT)
    if score not in ("0", "1"):
        raise RuntimeError(f"judge failed: {path}: {explanation}")
    row = {"question_id": result["question_id"], "status": "judged", "score": int(score), "explanation": explanation}
    destination.with_suffix(".tmp").write_text(json.dumps(row, ensure_ascii=False, indent=2))
    os.replace(destination.with_suffix(".tmp"), destination)
    return row

for root in map(Path, args.roots):
    files = sorted(root.rglob("result.json"))
    # A temporary API failure must not discard completed judgments or abort the
    # entire model. Failed items are left without judge.json and are retried by
    # the next invocation.
    rows = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(judge, path) for path in files]
        for future in futures:
            try:
                rows.append(future.result())
            except Exception as exc:
                print(f"[judge] deferred item after transient failure: {exc}", flush=True)
    root.joinpath("judgments.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    valid = [r for r in rows if r["score"] is not None]
    summary = {"samples": len(rows), "judged": len(valid), "infer_failed": len(rows)-len(valid), "correct": sum(r["score"] for r in valid), "accuracy_on_judged": sum(r["score"] for r in valid)/len(valid) if valid else None}
    summary["expected_samples"] = 209
    summary["missing_results"] = max(0, 209 - len(rows))
    summary["accuracy_full_dataset"] = summary["correct"] / 209
    root.joinpath("judge_summary.json").write_text(json.dumps(summary, indent=2))
    print(root, summary, flush=True)

