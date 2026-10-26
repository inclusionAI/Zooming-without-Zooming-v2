"""Result writing, trajectory saving, and summary statistics."""

import os
import json


def save_trajectory(save_directory, json_dict):
    """Append one JSON record to traj.jsonl in save_directory."""
    os.makedirs(save_directory, exist_ok=True)
    with open(os.path.join(save_directory, "traj.jsonl"), "a+") as f:
        f.write(json.dumps(json_dict, ensure_ascii=False) + "\n")


def save_results(results, output_dir):
    """Write list of result dicts to output_dir/results.jsonl."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "results.jsonl")
    with open(path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print(f"Results saved to {path}")


def calculate_summary_metrics(results):
    """Compute summary metrics from result list."""
    if not results:
        return {}
    n = len(results)
    acc = sum(r.get("accuracy_score", 0) for r in results) / n
    def turns(r):
        ch = r.get("conversation_history", [])
        return sum(1 for m in ch if m.get("role") == "assistant")
    avg_turns = sum(turns(r) for r in results) / n
    return {
        "total_samples": n,
        "overall_accuracy_score": round(acc, 4),
        "average_turns_per_sample": round(avg_turns, 2),
    }


def save_summary_metrics(summary_metrics, output_dir, print_message=False):
    """Write summary_metrics to output_dir/summary_metrics.json."""
    path = os.path.join(output_dir, "summary_metrics.json")
    with open(path, "w") as f:
        json.dump(summary_metrics, f, indent=2)
    if print_message:
        print(f"Summary saved to {path}")


def print_summary(results, output_dir):
    """Compute, print, and save summary. No-op if results is empty."""
    if not results:
        return
    m = calculate_summary_metrics(results)
    print("\n--- Evaluation Summary ---")
    print(f"Total samples: {m['total_samples']}")
    print(f"Overall accuracy: {m['overall_accuracy_score']:.4f}")
    print(f"Average turns: {m['average_turns_per_sample']:.2f}")
    print(f"Summary saved to {os.path.join(output_dir, 'summary_metrics.json')}")
    print("------------------------\n")
    save_summary_metrics(m, output_dir, print_message=False)
