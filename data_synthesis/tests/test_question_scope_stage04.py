import importlib.util
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "scripts", "04_validate_questions.py"
)
SPEC = importlib.util.spec_from_file_location("validate_questions_script", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class QuestionScopeStage04Test(unittest.TestCase):
    @patch.object(MODULE, "generate_mcq")
    @patch.object(MODULE, "build_validation_images")
    def test_incomplete_scalar_answer_stops_before_mcq(self, build_images, generate_mcq):
        build_images.return_value = {
            "privileged_layout": "/tmp/privileged.jpg",
            "marked_original": "/tmp/marked.jpg",
            "drop": {},
        }
        candidate = {
            "question": (
                "Arrange the three marked grilles from left to right, and state "
                "the shape of the badge on the middle vehicle."
            ),
            "answer_semantics": "scalar",
            "final_answer_kind": "semantic",
            "anchor_colors": {},
            "self_check": {"voted_gt": "oval"},
        }
        result = MODULE.process_candidate(
            {}, candidate, SimpleNamespace(evidence_dir="/tmp", layout_canvas_size=1024)
        )
        self.assertEqual(result["status"], "invalid_question_answer_scope")
        self.assertIn(
            "multiple_output_requests_for_scalar_answer",
            result["contract_reasons"],
        )
        generate_mcq.assert_not_called()


if __name__ == "__main__":
    unittest.main()
