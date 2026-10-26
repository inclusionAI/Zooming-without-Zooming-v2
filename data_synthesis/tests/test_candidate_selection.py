import importlib.util
import os
import unittest


SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "scripts", "03_generate_questions.py"
)
SPEC = importlib.util.spec_from_file_location("generate_questions_script", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CandidateSelectionTest(unittest.TestCase):
    def _candidate(self, cid, qtype, kind, score=4, soft_flags=None):
        return {
            "candidate_id": cid,
            "question_type": qtype,
            "final_answer_kind": kind,
            "observed_answer_kind": kind,
            "naturalness_score": score,
            "visual_confidence": score,
            "multi_region_score": score,
            "local_gate_soft_flags": soft_flags or [],
        }

    def test_soft_image_target_preserves_every_question_type(self):
        candidates = [
            self._candidate("a1", "anomaly_detection", "anchor", 5),
            self._candidate("a2", "extremum", "anchor", 5),
            self._candidate("a3", "candidate_matching", "anchor", 5),
            self._candidate("a4", "set_filtering", "anchor", 5),
            self._candidate("n1", "aggregate_count", "numeric", 5),
            self._candidate("n2", "arithmetic_composition", "numeric", 5),
            self._candidate("n3", "aggregate_numeric", "numeric", 5),
            self._candidate("s1", "spatial_chain", "semantic", 4),
            self._candidate("r1", "relational_composition", "semantic", 4),
        ]
        selected = MODULE._select_image_candidates(candidates, 8)
        self.assertEqual(len(selected), 9)
        self.assertEqual(
            {item["question_type"] for item in selected},
            {item["question_type"] for item in candidates},
        )

    def test_below_soft_cap_keeps_all_candidates_without_backups(self):
        candidates = [
            self._candidate("a1", "anomaly_detection", "anchor", 5),
            self._candidate("a2", "anomaly_detection", "anchor", 4),
            self._candidate("n1", "aggregate_count", "numeric", 5),
        ]
        selected = MODULE._select_image_candidates(candidates, 8)
        self.assertEqual(len(selected), 3)
        self.assertEqual(
            {item["candidate_id"] for item in selected},
            {item["candidate_id"] for item in candidates},
        )

    def test_above_soft_cap_covers_types_then_fills_by_quality(self):
        candidates = [
            self._candidate("a1", "anomaly_detection", "anchor", 5),
            self._candidate("a2", "anomaly_detection", "anchor", 4),
            self._candidate("a3", "anomaly_detection", "anchor", 3),
            self._candidate("a4", "anomaly_detection", "anchor", 2),
            self._candidate("n1", "aggregate_count", "numeric", 5),
            self._candidate("n2", "aggregate_count", "numeric", 4),
            self._candidate("n3", "aggregate_count", "numeric", 3),
            self._candidate("s1", "spatial_chain", "semantic", 5),
            self._candidate("s2", "spatial_chain", "semantic", 4),
        ]
        selected = MODULE._select_image_candidates(candidates, 8)
        self.assertEqual(len(selected), 8)
        self.assertEqual(
            {item["candidate_id"] for item in selected},
            {"a1", "a2", "a3", "n1", "n2", "n3", "s1", "s2"},
        )

    def test_soft_flags_reduce_priority_without_hard_rejection(self):
        clean = self._candidate("clean", "spatial_chain", "semantic", 4)
        flagged = self._candidate(
            "flagged", "spatial_chain", "semantic", 4,
            soft_flags=["low_naturalness", "freeform_quota_exceeded"],
        )
        selected = MODULE._select_image_candidates([flagged, clean], 1)
        self.assertEqual(selected[0]["candidate_id"], "clean")


if __name__ == "__main__":
    unittest.main()
