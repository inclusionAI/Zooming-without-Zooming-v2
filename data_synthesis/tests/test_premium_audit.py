import json
import unittest
from unittest.mock import patch

from mr_vqa.premium_audit import (
    _parse_audit_response,
    _reference_match,
    audit_candidate,
)


PASS_RESULT = {
    "verdict": "pass",
    "answer_complete": True,
    "question_unambiguous": True,
    "evidence_sufficient": True,
    "independent_answer": "oval",
    "issues": [],
}


class PremiumAuditTest(unittest.TestCase):
    def setUp(self):
        self.candidate = {
            "question": "What shape is the badge on the selected vehicle?",
            "answer_semantics": "scalar",
            "evidence_regions": [{"region_id": "r000"}],
        }

    @patch("mr_vqa.premium_audit.call_qwen_vl")
    def test_qwen_vl_accept_passes(self, call_qwen_vl):
        call_qwen_vl.return_value = json.dumps(PASS_RESULT)
        result = audit_candidate(
            self.candidate, "oval", "/tmp/layout.jpg", "key"
        )
        self.assertTrue(result["pass"])
        self.assertEqual(result["route"], "qwen_vl_pass")
        self.assertNotIn("gpt", result)
        self.assertTrue(result["qwen_vl"]["pass"])
        self.assertEqual(call_qwen_vl.call_count, 1)

    @patch("mr_vqa.premium_audit.call_qwen_vl")
    def test_definite_qwen_vl_rejection_rejects(self, call_qwen_vl):
        rejected = {
            **PASS_RESULT,
            "verdict": "fail",
            "answer_complete": False,
            "issues": ["answer omits a requested output"],
        }
        call_qwen_vl.return_value = json.dumps(rejected)
        result = audit_candidate(
            self.candidate, "oval", "/tmp/layout.jpg", "key"
        )
        self.assertFalse(result["pass"])
        self.assertEqual(result["route"], "qwen_vl_reject")
        self.assertNotIn("gpt", result)
        self.assertFalse(result["qwen_vl"]["pass"])

    @patch("mr_vqa.premium_audit.call_qwen_vl")
    def test_answer_mismatch_is_uncertain_and_rejected(self, call_qwen_vl):
        mismatch = {**PASS_RESULT, "independent_answer": "shield"}
        call_qwen_vl.return_value = json.dumps(mismatch)
        result = audit_candidate(
            self.candidate, "oval", "/tmp/layout.jpg", "key"
        )
        self.assertFalse(result["pass"])
        self.assertEqual(result["route"], "qwen_vl_uncertain")
        self.assertFalse(result["qwen_vl"]["reference_match"])

    @patch("mr_vqa.premium_audit.call_qwen_vl")
    def test_high_risk_still_uses_single_qwen_vl_call(self, call_qwen_vl):
        self.candidate["question_type"] = "multi_hop_reasoning"
        call_qwen_vl.return_value = json.dumps(PASS_RESULT)
        result = audit_candidate(
            self.candidate, "oval", "/tmp/layout.jpg", "key"
        )
        self.assertTrue(result["pass"])
        self.assertEqual(result["route"], "qwen_vl_pass")
        self.assertEqual(call_qwen_vl.call_count, 1)
        self.assertEqual(result["high_risk_reasons"], ["question_type:multi_hop_reasoning"])

    @patch("mr_vqa.premium_audit.call_qwen_vl")
    def test_model_error_fails_closed(self, call_qwen_vl):
        call_qwen_vl.side_effect = RuntimeError("timeout")
        result = audit_candidate(
            self.candidate, "oval", "/tmp/layout.jpg", "key"
        )
        self.assertFalse(result["pass"])
        self.assertEqual(result["route"], "qwen_vl_uncertain")
        self.assertIn("error", result["qwen_vl"])

    @patch("mr_vqa.premium_audit.call_qwen_vl")
    def test_qwen_vl_generated_question_audited_by_qwen_vl(self, call_qwen_vl):
        self.candidate["generation_source"] = "qwen_vl"
        call_qwen_vl.return_value = json.dumps(PASS_RESULT)
        result = audit_candidate(
            self.candidate, "oval", "/tmp/layout.jpg", "key"
        )
        self.assertTrue(result["pass"])
        self.assertEqual(result["route"], "qwen_vl_pass")
        self.assertIsNotNone(result["qwen_vl"])
        self.assertEqual(call_qwen_vl.call_count, 1)

    def test_parser_rejects_missing_or_wrongly_typed_fields(self):
        parsed, error = _parse_audit_response('{"verdict":"pass"}')
        self.assertIsNone(parsed)
        self.assertEqual(error, "missing_required_fields")

        invalid = {**PASS_RESULT, "answer_complete": "true"}
        parsed, error = _parse_audit_response(json.dumps(invalid))
        self.assertIsNone(parsed)
        self.assertEqual(error, "invalid_boolean:answer_complete")

        invalid = {**PASS_RESULT, "explanation": "extra field"}
        parsed, error = _parse_audit_response(json.dumps(invalid))
        self.assertIsNone(parsed)
        self.assertEqual(error, "unexpected_fields")

    def test_parser_accepts_fenced_valid_json(self):
        parsed, error = _parse_audit_response(
            "```json\n" + json.dumps(PASS_RESULT) + "\n```"
        )
        self.assertIsNone(error)
        self.assertEqual(parsed["independent_answer"], "oval")

    def test_ocr_identifier_ignores_separator_punctuation(self):
        candidate = {
            "question": "What registration code is printed on the boat?",
            "answer_semantics": "scalar",
            "final_answer_kind": "text",
        }
        self.assertEqual(
            _reference_match("TRF 718", "TRF.718", candidate),
            (True, "ocr_identifier"),
        )
        self.assertEqual(
            _reference_match("1.25", "125", candidate),
            (False, "none"),
        )

    def test_set_match_allows_order_but_not_missing_items(self):
        candidate = {
            "question": "Which marked regions contain fog?",
            "answer_semantics": "set_complete",
            "final_answer_kind": "anchor_set",
        }
        self.assertEqual(
            _reference_match(
                "pink, brown and orange", "brown + orange + pink", candidate
            ),
            (True, "unordered_set"),
        )
        self.assertEqual(
            _reference_match("brown and pink", "brown + orange + pink", candidate),
            (False, "none"),
        )

    def test_scalar_wrapper_is_conservative(self):
        candidate = {
            "question": "What brand is written on the selected sign?",
            "answer_semantics": "scalar",
            "final_answer_kind": "text",
        }
        self.assertEqual(
            _reference_match("SoCo creamery", "SoCo", candidate),
            (True, "scalar_wrapper"),
        )
        self.assertEqual(
            _reference_match("not SoCo", "SoCo", candidate),
            (False, "none"),
        )

        multi_output = {
            **candidate,
            "question": "Which region contains a person, and what is their role?",
        }
        self.assertEqual(
            _reference_match("orange region; driver", "driver", multi_output),
            (False, "none"),
        )


if __name__ == "__main__":
    unittest.main()
