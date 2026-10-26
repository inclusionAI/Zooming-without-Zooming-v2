import unittest

from mr_vqa.questions import _evaluate_region_expression, _expression_root_is_abs
from mr_vqa.reasoning_certificate import validate_reasoning_certificate


class ReasoningCertificateTest(unittest.TestCase):
    def test_open_evidence_graph_accepts_free_operation_names(self):
        certificate = {
            "mode": "evidence_graph",
            "steps": [
                {"id": "s1", "op": "inspect_hand_activity", "regions": ["r001"],
                 "inputs": [], "query": "what the person is doing"},
                {"id": "s2", "op": "match_social_role", "regions": ["r002"],
                 "inputs": ["s1"], "query": "which person matches the relation"},
                {"id": "s3", "op": "describe_final_action", "regions": ["r003"],
                 "inputs": ["s2"], "query": "the final visible action"},
            ],
            "output_step": "s3",
        }
        passed, reasons, details = validate_reasoning_certificate(
            certificate, {"r001", "r002", "r003"}
        )
        self.assertTrue(passed, reasons)
        self.assertEqual(details["reachable_regions"], ["r001", "r002", "r003"])

    def test_decorative_region_is_rejected(self):
        certificate = {
            "mode": "evidence_graph",
            "steps": [
                {"id": "s1", "op": "observe", "regions": ["r001"],
                 "inputs": [], "query": "first clue"},
                {"id": "s2", "op": "observe", "regions": ["r002"],
                 "inputs": [], "query": "unused clue"},
                {"id": "s3", "op": "describe", "regions": [],
                 "inputs": ["s1"], "query": "final answer"},
            ],
            "output_step": "s3",
        }
        passed, reasons, _ = validate_reasoning_certificate(
            certificate, {"r001", "r002"}
        )
        self.assertFalse(passed)
        self.assertIn("not_all_regions_reach_output", reasons)
        self.assertIn("disconnected_or_decorative_steps", reasons)

    def test_executable_mode_uses_audited_operations(self):
        certificate = {
            "mode": "executable",
            "steps": [
                {"id": "s1", "op": "invent_answer", "regions": ["r001"],
                 "inputs": [], "query": "invent something"},
                {"id": "s2", "op": "return_value", "regions": ["r002"],
                 "inputs": ["s1"], "query": "return result"},
            ],
            "output_step": "s2",
        }
        passed, reasons, _ = validate_reasoning_certificate(
            certificate, {"r001", "r002"}
        )
        self.assertFalse(passed)
        self.assertIn("unsupported_executable_operation", reasons)

    def test_absolute_difference_is_executable_and_explicit(self):
        expression = "abs(r009 - (r010 + r011))"
        self.assertEqual(
            _evaluate_region_expression(expression, {"r009": 2, "r010": 1, "r011": 1}),
            0,
        )
        self.assertTrue(_expression_root_is_abs(expression))
        self.assertFalse(_expression_root_is_abs("r009 - (r010 + r011)"))


if __name__ == "__main__":
    unittest.main()

