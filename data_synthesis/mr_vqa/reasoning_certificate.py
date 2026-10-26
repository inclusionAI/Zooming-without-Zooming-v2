"""Structural validation for model-generated multi-region reasoning certificates.

Certificates are proofs of evidence use, not question templates. Executable
certificates use a small auditable operator set; evidence graphs may use open
operation names so semantic and open-answer questions are not forced into a
closed ontology.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Set, Tuple


CERTIFICATE_MODES = {"executable", "evidence_graph"}

# This list constrains deterministic programs, not evidence graphs. Observation
# and semantic graph operations remain open-vocabulary under evidence_graph.
EXECUTABLE_OPS = {
    "observe", "count", "read_text", "read_number", "read_attribute",
    "classify", "detect", "select", "describe", "report",
    "count_objects", "classify_object", "detect_object", "detect_attribute",
    "get_attribute", "filter_by_attribute", "select_by_feature",
    "select_by_attribute", "compare_attribute", "aggregate_set",
    "add", "subtract", "multiply", "divide", "absolute_difference",
    "sum", "compare", "argmax", "argmin", "filter", "match",
    "select_unique", "lookup", "pair", "boolean_and", "boolean_or",
    "boolean_not", "return_value", "return_description",
}

# The certificate is a structural proof, while pixel-level correctness is
# established by privileged cross-model voting. Accept common deterministic
# spellings instead of discarding a sound graph over a cosmetic op-name choice.
EXECUTABLE_OP_ALIASES = {
    "compute_absolute_difference": "absolute_difference",
    "compute_difference": "subtract",
    "calculate_difference": "subtract",
    "calculate_sum": "sum",
    "compute_sum": "sum",
    "total": "sum",
    "retrieve_attribute": "get_attribute",
    "report_attribute": "return_description",
    "return": "return_value",
}

_SAFE_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _as_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _step_regions(step: Dict[str, Any]) -> List[str]:
    regions = _as_string_list(step.get("regions"))
    singular = str(step.get("region") or "").strip()
    if singular and singular not in regions:
        regions.append(singular)
    return regions


def _ancestors(output_step: str, dependencies: Dict[str, List[str]]) -> Set[str]:
    reached: Set[str] = set()
    stack = [output_step]
    while stack:
        current = stack.pop()
        if current in reached:
            continue
        reached.add(current)
        stack.extend(dependencies.get(current, []))
    return reached


def validate_reasoning_certificate(
    certificate: Any,
    evidence_region_ids: Iterable[str],
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Validate certificate shape and whether every region reaches the output."""
    reasons: List[str] = []
    details: Dict[str, Any] = {}
    expected_regions = {str(rid) for rid in evidence_region_ids}

    if not isinstance(certificate, dict):
        return False, ["missing_reasoning_certificate"], details

    mode = str(certificate.get("mode") or "").strip().lower()
    if mode not in CERTIFICATE_MODES:
        reasons.append("invalid_certificate_mode")

    steps = certificate.get("steps")
    if not isinstance(steps, list) or not (2 <= len(steps) <= 16):
        return False, reasons + ["invalid_certificate_steps"], details

    step_by_id: Dict[str, Dict[str, Any]] = {}
    dependencies: Dict[str, List[str]] = {}
    regions_by_step: Dict[str, Set[str]] = {}
    operations: List[str] = []
    ordered_ids: List[str] = []

    for index, raw_step in enumerate(steps):
        if not isinstance(raw_step, dict):
            reasons.append("invalid_certificate_step")
            continue
        step_id = str(raw_step.get("id") or "").strip()
        op = str(raw_step.get("op") or "").strip().lower()
        op = EXECUTABLE_OP_ALIASES.get(op, op)
        if not _SAFE_NAME.fullmatch(step_id) or step_id in step_by_id:
            reasons.append("invalid_or_duplicate_step_id")
            continue
        if not _SAFE_NAME.fullmatch(op):
            reasons.append("invalid_step_operation")
        elif mode == "executable" and op not in EXECUTABLE_OPS:
            reasons.append("unsupported_executable_operation")

        inputs = _as_string_list(raw_step.get("inputs"))
        if any(input_id not in step_by_id for input_id in inputs):
            reasons.append("dangling_or_forward_step_input")

        regions = set(_step_regions(raw_step))
        if regions - expected_regions:
            reasons.append("unknown_evidence_region")
        if not inputs and not regions:
            reasons.append("step_has_no_evidence_or_inputs")
        if not str(raw_step.get("query") or "").strip():
            reasons.append("step_missing_query")

        step_by_id[step_id] = raw_step
        dependencies[step_id] = inputs
        regions_by_step[step_id] = regions
        operations.append(op)
        ordered_ids.append(step_id)

    output_step = str(certificate.get("output_step") or "").strip()
    if output_step not in step_by_id:
        reasons.append("invalid_output_step")
        reachable_steps: Set[str] = set()
    else:
        reachable_steps = _ancestors(output_step, dependencies)

    reachable_regions: Set[str] = set()
    for step_id in reachable_steps:
        reachable_regions.update(regions_by_step.get(step_id, set()))

    if reachable_regions != expected_regions:
        reasons.append("not_all_regions_reach_output")
    if set(step_by_id) != reachable_steps:
        reasons.append("disconnected_or_decorative_steps")

    details.update({
        "mode": mode,
        "step_count": len(step_by_id),
        "output_step": output_step,
        "reachable_steps": sorted(reachable_steps),
        "reachable_regions": sorted(reachable_regions),
        "expected_regions": sorted(expected_regions),
        "operations": operations,
    })
    return not reasons, list(dict.fromkeys(reasons)), details
