"""
Greedy inference script for vision-language agent with tool calling.
Single-path (greedy) only; no experience, skill, reflection, or TTS search.
"""

import argparse
import os
import sys
import json
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from configs.prompt_loader import load_inference_prompts, setup_global_prompts
from engine.api.api_processors import process_single_sample, set_global_prompts as set_processors_prompts
from engine.api.api_model_caller import set_global_prompts as set_model_caller_prompts
from utils.result_utils import save_results, print_summary


def get_sample_metadata(sample, sample_idx, output_dir):
    """Return (question_id, sample_dir) for a sample."""
    question_id = sample.get("doc_id", sample.get("question_id", f"sample_{sample_idx}"))
    sample_dir = os.path.join(output_dir, str(question_id))
    return question_id, sample_dir


def check_sample_completed(sample_dir, args):
    """Return truthy if sample already has traj and metrics (for --skip-completed)."""
    if not getattr(args, "skip_completed", False):
        return None
    result_path = os.path.join(sample_dir, "result.json")
    if os.path.exists(result_path):
        try:
            with open(result_path, encoding="utf-8") as f:
                result = json.load(f)
            answer = result.get("final_answer", "")
            if answer and not answer.startswith("Error:"):
                return result
        except (OSError, ValueError):
            pass
    return None
    traj_path = os.path.join(sample_dir, "traj.jsonl")
    metrics_path = os.path.join(sample_dir, "metrics.json")
    if not os.path.exists(traj_path) or not os.path.exists(metrics_path):
        return None
    try:
        found_turn = False
        with open(traj_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("turn_idx") is not None:
                    found_turn = True
                    break
        if not found_turn:
            return None
    except Exception:
        return None
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return {"sample_rollout_results": [{"accuracy_score": metrics.get("accuracy_score")}]}


def main(args):
    args.model_name = os.environ.get("REASONING_MODEL_NAME")
    if not args.model_name:
        raise ValueError("REASONING_MODEL_NAME environment variable must be set")

    print(f"Starting greedy inference. Reasoning model: {args.model_name}")

    try:
        prompts_dict = load_inference_prompts(args)
        (
            SYSTEM_PROMPT,
            TOOL_CALL_CROP_MULTI_TRUN_PROMPT,
            _,
            FEEDBACK_PROMPT_TOO_SMALL,
            TTS_TOOL_CALL_CROP_MULTI_TRUN_PROMPT,
        ) = setup_global_prompts(prompts_dict)
        TOOL_CALL_CROP_MULTI_TRUN_PROMPT = TTS_TOOL_CALL_CROP_MULTI_TRUN_PROMPT
    except Exception as e:
        print(f"Error loading inference prompts: {e}")
        return

    set_processors_prompts(SYSTEM_PROMPT, TOOL_CALL_CROP_MULTI_TRUN_PROMPT, FEEDBACK_PROMPT_TOO_SMALL)
    set_model_caller_prompts(FEEDBACK_PROMPT_TOO_SMALL, TOOL_CALL_CROP_MULTI_TRUN_PROMPT)

    try:
        import yaml
        tool_config_path = getattr(args, "tool_config_path", os.path.join(SCRIPT_DIR, "configs", "tool_configs.yaml"))
        if os.path.exists(tool_config_path):
            with open(tool_config_path, "r", encoding="utf-8") as f:
                args.tool_configs = yaml.safe_load(f) or {}
        else:
            args.tool_configs = {}
    except Exception as e:
        print(f"Error loading tool configs: {e}")
        args.tool_configs = {}

    if args.input_file.endswith(".jsonl"):
        with open(args.input_file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    else:
        with open(args.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

    if getattr(args, "max_samples", None) and args.max_samples > 0:
        data = data[: args.max_samples]
    print(f"Loaded {len(data)} samples.")

    os.makedirs(args.output_dir, exist_ok=True)
    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": getattr(args, "max_completion_tokens", 8192),
    }

    all_results = []
    for sample_idx, sample in enumerate(tqdm(data, desc="Processing samples")):
        question_id, sample_dir = get_sample_metadata(sample, sample_idx, args.output_dir)
        os.makedirs(sample_dir, exist_ok=True)

        completed = check_sample_completed(sample_dir, args)
        if completed:
            all_results.append(completed)
            continue

        result = process_single_sample(sample, args, sampling_params, rollout_idx=None)
        if result:
            all_results.append(result)
            result_path = os.path.join(sample_dir, "result.json")
            with open(result_path + ".tmp", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            os.replace(result_path + ".tmp", result_path)
            metrics = {
                "accuracy_score": result.get("accuracy_score"),
                "trajectory_score": result.get("trajectory_score"),
                "trajectory_analysis": result.get("trajectory_analysis"),
            }
            with open(os.path.join(sample_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

    save_results(all_results, args.output_dir)
    print_summary(all_results, args.output_dir)
    print("Inference finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Greedy tool-calling agent inference.")
    parser.add_argument("--input-file", type=str, required=True, help="Input JSON/JSONL file.")
    parser.add_argument("--image-folder", type=str, required=True, help="Folder containing images.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--skip-completed", action="store_true", help="Skip samples with existing traj and metrics.")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to process.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-completion-tokens", type=int, default=8192)
    parser.add_argument("--max-turns", type=int, default=16)
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--max-total-tokens", type=int, default=65536)
    parser.add_argument("--max-pixels", type=int, default=2000000)
    parser.add_argument("--min-pixels", type=int, default=40000)
    parser.add_argument("--inference-prompts-path", type=str, default=None)
    parser.add_argument("--system-prompt-key", type=str, default="multi_tool_agent_search")
    parser.add_argument("--tool-config-path", type=str, default=None)
    parser.add_argument("--image-search-max-calls", type=int, default=100)
    parser.add_argument("--web-search-max-calls", type=int, default=100)
    parser.add_argument("--bbox-format", type=str, default="norm999", choices=["auto", "pixel", "norm999", "norm1"])

    args = parser.parse_args()
    if args.inference_prompts_path is None:
        args.inference_prompts_path = os.path.join(SCRIPT_DIR, "prompts", "inference_prompts.yaml")
    if args.tool_config_path is None:
        args.tool_config_path = os.path.join(SCRIPT_DIR, "configs", "tool_configs.yaml")
    main(args)
