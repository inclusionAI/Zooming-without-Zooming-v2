# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pure helpers for verified-history multi-turn VQA rollouts.

This module deliberately has no torch/ray dependency.  Keeping schema validation,
answer extraction, and history selection here makes the online policy easy to test
without launching a rollout server.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

_ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL)
_OPTION_PATTERN = re.compile(r"^[\s\(\[]*([A-F])(?:[\s\.\)\]:-]|$)", flags=re.IGNORECASE)
_LEADING_ARTICLE_PATTERN = re.compile(r"^(?:a|an|the)\s+", flags=re.IGNORECASE)


@dataclass(frozen=True)
class AnswerScore:
    """Scoring result for one complete reasoning-plus-answer response."""

    reward: float
    accuracy: float
    format: float
    extracted_answer: str
    eligible_for_history: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_episode_spec(value: Any) -> dict[str, Any]:
    """Normalize a parquet scalar containing an episode JSON object."""

    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError(f"episode_json must decode to a dict, got {type(value).__name__}")
    return value


def validate_episode_spec(episode: dict[str, Any]) -> None:
    """Fail closed when the runtime episode violates the send-once contract."""

    episode_id = str(episode.get("episode_id") or "").strip()
    if not episode_id:
        raise ValueError("episode_id is required")
    if episode.get("image_delivery_mode") != "send_each_source_once":
        raise ValueError(
            f"episode {episode_id}: image_delivery_mode must be 'send_each_source_once', "
            f"got {episode.get('image_delivery_mode')!r}"
        )

    turns = episode.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"episode {episode_id}: turns must be a non-empty list")

    delivered: list[str] = []
    delivered_set: set[str] = set()
    previous_turn_id = 0
    for position, turn in enumerate(turns, start=1):
        if not isinstance(turn, dict):
            raise TypeError(f"episode {episode_id}: turn {position} must be a dict")
        turn_id = int(turn.get("turn_id", position))
        if turn_id <= previous_turn_id:
            raise ValueError(f"episode {episode_id}: turn ids must be strictly increasing")
        previous_turn_id = turn_id

        question = str(turn.get("question") or "").strip()
        ground_truth = str(turn.get("ground_truth") or "").strip()
        if not question:
            raise ValueError(f"episode {episode_id} turn {turn_id}: question is empty")
        if not ground_truth:
            raise ValueError(f"episode {episode_id} turn {turn_id}: ground_truth is empty")

        new_images = turn.get("new_images") or []
        if not isinstance(new_images, list):
            raise TypeError(f"episode {episode_id} turn {turn_id}: new_images must be a list")
        for image in new_images:
            if not isinstance(image, dict):
                raise TypeError(f"episode {episode_id} turn {turn_id}: each new image must be a dict")
            image_id = str(image.get("image_id") or "").strip()
            image_path = str(image.get("path") or "").strip()
            if not image_id or not image_path:
                raise ValueError(f"episode {episode_id} turn {turn_id}: image_id and path are required")
            if image_id in delivered_set:
                raise ValueError(f"episode {episode_id} turn {turn_id}: image {image_id!r} is delivered more than once")
            delivered.append(image_id)
            delivered_set.add(image_id)

        context_image_ids = turn.get("context_image_ids")
        if context_image_ids is not None and set(map(str, context_image_ids)) != delivered_set:
            raise ValueError(
                f"episode {episode_id} turn {turn_id}: context_image_ids do not match images delivered so far"
            )


def build_user_content(turn: dict[str, Any]) -> list[dict[str, Any]]:
    """Build one user message while introducing each source image exactly once."""

    content: list[dict[str, Any]] = []
    for image in turn.get("new_images") or []:
        content.append({"type": "image", "image": image["path"]})
    content.append({"type": "text", "text": str(turn["question"])})
    return content


def stable_random(*parts: Any) -> random.Random:
    """Return a process-independent RNG for reproducible history selection."""

    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big", signed=False)
    return random.Random(seed)


def choose_verified_candidate(scores: Iterable[AnswerScore], rng: random.Random) -> int | None:
    """Uniformly choose one history-eligible candidate without creating branches."""

    eligible = [index for index, score in enumerate(scores) if score.eligible_for_history]
    return rng.choice(eligible) if eligible else None


def extract_answer(response: str) -> tuple[str, bool]:
    """Extract the final ``<answer>`` span and validate its placement."""

    response = response or ""
    matches = list(_ANSWER_PATTERN.finditer(response))
    if not matches:
        return response.strip(), False
    final_match = matches[-1]
    format_valid = len(matches) == 1 and not response[final_match.end() :].strip()
    return final_match.group(1).strip(), format_valid


def normalize_answer(value: Any) -> str:
    """Normalize short VQA answers without inspecting the reasoning text."""

    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"\s+", " ", text)
    text = _LEADING_ARTICLE_PATTERN.sub("", text)
    # Keep letters and numbers from every script while treating punctuation as spacing.
    text = "".join(char if char.isalnum() or char in {".", "-"} else " " for char in text)
    return re.sub(r"\s+", " ", text).strip(" .")


def _as_decimal(value: str) -> Decimal | None:
    try:
        return Decimal(value.replace(",", ""))
    except (InvalidOperation, ValueError):
        return None


def _answers_equal(candidate: str, reference: str) -> bool:
    normalized_candidate = normalize_answer(candidate)
    normalized_reference = normalize_answer(reference)
    if normalized_candidate == normalized_reference:
        return True

    candidate_number = _as_decimal(normalized_candidate)
    reference_number = _as_decimal(normalized_reference)
    if candidate_number is not None and reference_number is not None:
        return candidate_number == reference_number

    if re.fullmatch(r"[a-f]", normalized_reference):
        option_match = _OPTION_PATTERN.match(candidate.strip())
        return option_match is not None and option_match.group(1).casefold() == normalized_reference
    return False


def collect_reference_answers(
    ground_truth: Any,
    acceptable_answers: Iterable[Any] | None = None,
) -> tuple[str, ...]:
    """Collect non-empty ground-truth aliases without splitting a string into characters."""

    values = [ground_truth]
    if acceptable_answers is not None:
        if isinstance(acceptable_answers, str):
            values.append(acceptable_answers)
        else:
            values.extend(acceptable_answers)
    return tuple(dict.fromkeys(str(value).strip() for value in values if str(value or "").strip()))


def score_response(
    response: str,
    ground_truth: Any,
    acceptable_answers: Iterable[Any] | None = None,
    *,
    require_answer_tag: bool = True,
) -> AnswerScore:
    """Score only the final answer while retaining the entire response for RL."""

    extracted_answer, format_valid = extract_answer(response)
    references = collect_reference_answers(ground_truth, acceptable_answers)
    accuracy = bool(extracted_answer) and any(_answers_equal(extracted_answer, reference) for reference in references)
    return score_response_with_accuracy(
        response,
        accuracy,
        require_answer_tag=require_answer_tag,
        extracted_answer=extracted_answer,
        format_valid=format_valid,
    )


def score_response_with_accuracy(
    response: str,
    accuracy: bool,
    *,
    require_answer_tag: bool = True,
    extracted_answer: str | None = None,
    format_valid: bool | None = None,
) -> AnswerScore:
    """Build an ``AnswerScore`` from an external semantic judgement."""

    if extracted_answer is None or format_valid is None:
        extracted_answer, format_valid = extract_answer(response)
    accuracy = bool(extracted_answer) and bool(accuracy)
    eligible = accuracy and (format_valid or not require_answer_tag)
    return AnswerScore(
        reward=float(eligible),
        accuracy=float(accuracy),
        format=float(format_valid),
        extracted_answer=extracted_answer,
        eligible_for_history=eligible,
    )


def format_gt_fallback(ground_truth: Any) -> str:
    """Create a minimal, non-trainable assistant history entry for an all-wrong turn."""

    return f"<answer>{str(ground_truth).strip()}</answer>"
