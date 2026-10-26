# Selected definitions from the project-adapted reference source.
# Not a relicense of the original. See MANIFEST.json and THIRD_PARTY_NOTICES.md.
from __future__ import annotations
import re
import json
from decimal import Decimal, ROUND_HALF_UP
MCA = {
    "obj_spatial_relation_oo", "obj_spatial_relation_oc_mv",
    "obj_spatial_relation_oo_mv", "spatial_imagination_oc",
    "spatial_imagination_oo", "spatial_imagination_oc_mv",
    "spatial_imagination_oo_mv", "position_matching", "camera_motion_infer",
    "distance_infer_center_oo", "distance_infer_center_oo_mv",
}

NA = {
    "depth_prediction_oc", "depth_prediction_oo", "distance_prediction_oc",
    "distance_prediction_oo", "depth_prediction_oc_mv", "depth_prediction_oo_mv",
    "distance_prediction_oo_mv", "distance_prediction_oc_mv",
}

MULTIVIEW_NA = {
    "depth_prediction_oc_mv", "depth_prediction_oo_mv",
    "distance_prediction_oc_mv", "distance_prediction_oo_mv",
}

ANSWER_FORMAT_SYSTEM = ("A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The answer are enclosed within <answer> </answer> tags, respectively, i.e., reasoning process here <answer> answer here </answer>")

LOW = NA

MIDDLE = {"view_change_infer", "position_matching", "camera_motion_infer"}

HIGH = MCA - {"position_matching", "camera_motion_infer"}

def official_prompt(row: dict) -> str:
    question = row["question"]
    task = row["task"]
    if task in NA:
        return f"\n{question}\nPlease answer the question using a single word or phrase."
    if task in MCA:
        coords = ""
        if task in {"position_matching", "camera_motion_infer"}:
            coords = "The values represent the bounding box coordinates normalized to a 0-1000 scale, with the top-left corner as the origin of the image."
        return f"\n{question}\n{coords}\nAnswer with the option's letter from the given choices directly."
    if task == "view_change_infer":
        return f"\n{question}\n\n"
    raise ValueError(f"Unknown SPAR-Bench task: {task}")

def exact_match(pred: str, target: str) -> float:
    pred, target = pred.lower().strip(), target.lower().strip()
    if not pred:
        return 0.0
    return float(pred == target or pred in target or pred[0] == target)

def first_number(pred: str, task: str) -> float:
    values = re.findall(r"(?<!\^)\d+\.\d+|(?<!\^)\d+", pred)
    if not values:
        raise ValueError("no number")
    return float(values[-1] if task in MULTIVIEW_NA else values[0])

def mra(pred: float, target: float) -> float:
    distance = abs(pred - target) if target == 0 else abs((pred - target) / target)
    thresholds = [0.5 + i * 0.05 for i in range(11)]
    return sum(distance <= 1 - t for t in thresholds) / len(thresholds)

def parse_instruction(instruction: str) -> dict[str, float]:
    return {key: float(value) for key, value in (item.split(":") for item in instruction.split(","))}

def view_change_score(pred: str, answer: str) -> float:
    actions = [
        ("move_right", "move_left"), ("move_up", "move_down"),
        ("move_forward", "move_backward"), ("rotate_right", "rotate_left"),
        ("rotate_up", "rotate_down"),
    ]
    p, a = parse_instruction(pred), parse_instruction(answer)
    return sum(mra(p.get(x, 0) - p.get(y, 0), a.get(x, 0) - a.get(y, 0)) for x, y in actions) / len(actions)

def extract_final_answer(pred: str, task: str) -> str:
    """Recover a model's final answer without changing the official score rules.

    Some trained checkpoints render their native chat template with a mandatory
    ``<answer>...</answer>`` envelope; other Qwen checkpoints write a natural
    language conclusion.  SPAR's official evaluator expects a bare token only
    because its reference models follow that instruction.  We therefore remove
    only the surrounding completion prose before applying the unmodified
    official metric.
    """
    match = re.search(r"<answer\b[^>]*>\s*(.*?)\s*</answer>", pred, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    if task in MCA:
        # Prefer the final explicit conclusion, e.g. "The answer is option D".
        choices = list(re.finditer(
            r"(?:final\s+)?(?:answer|option|choice)\s*(?:is|:|=)?\s*"
            r"(?:option\s*)?[\(\[]?([A-D])[\)\].,;:!\s]*",
            pred,
            flags=re.IGNORECASE,
        ))
        if choices:
            return choices[-1].group(1)
        # A final standalone option letter is the next most conservative form.
        tail_choice = re.search(r"(?:^|\n|\s)[\(\[]?([A-D])[\)\].!\s]*$", pred, re.IGNORECASE)
        if tail_choice:
            return tail_choice.group(1)

    if task in NA:
        # With an explanatory completion, the final numeric expression is the
        # answer; bare official outputs still contain exactly one number.
        values = re.findall(r"(?<!\^)\d+\.\d+|(?<!\^)\d+", pred)
        if values:
            return values[-1]

    return pred

def official_score(row: dict, pred: str) -> float:
    pred = extract_final_answer(pred, row["task"])
    try:
        if row["task"] in MCA:
            return exact_match(pred, row["answer"])
        if row["task"] in NA:
            return mra(first_number(pred, row["task"]), float(row["answer"]))
        if row["task"] == "view_change_infer":
            return view_change_score(pred, row["answer"])
    except Exception:
        return 0.0
    return 0.0
