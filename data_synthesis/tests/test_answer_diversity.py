import unittest
from unittest.mock import patch

from mr_vqa.question_templates import (
    QUESTION_TEMPLATES,
    select_question_templates,
    validate_template_registry,
)
from mr_vqa.questions import (
    answer_kind_matches_declared,
    classify_final_answer_kind,
    has_anchor_attribute_leak,
    infer_answer_semantics,
    normalize_question_type,
    open_answer_word_count,
    uses_open_answer_space,
    validate_question_answer_contract,
)
from mr_vqa.selfcheck import _llm_judge_largest_group


ANCHORS = {
    "r000": ("red", (255, 0, 0)),
    "r001": ("blue", (0, 0, 255)),
    "r002": ("green", (0, 255, 0)),
}


class AnswerDiversityTest(unittest.TestCase):
    def test_template_registry_and_semantic_pool(self):
        self.assertTrue(validate_template_registry())
        semantic = [
            spec for spec in QUESTION_TEMPLATES
            if spec["final_answer_kind"] in {"semantic", "semantic_set", "semantic_pair", "text"}
        ]
        self.assertGreaterEqual(len(semantic), 30)

    def test_template_selection_balances_anchor_and_numeric(self):
        selected = select_question_templates(4, 8, "stable-seed")
        kinds = [spec["final_answer_kind"] for spec in selected]
        self.assertGreaterEqual(kinds.count("anchor"), 1)
        self.assertLessEqual(kinds.count("anchor"), 3)
        self.assertLessEqual(kinds.count("numeric"), 2)
        self.assertGreaterEqual(
            sum(kind in {"semantic", "semantic_set", "semantic_pair", "text"} for kind in kinds),
            4,
        )

    def test_answer_kind_classifier(self):
        self.assertEqual(classify_final_answer_kind("blue", ANCHORS), "anchor")
        self.assertEqual(classify_final_answer_kind("red + green", ANCHORS), "anchor")
        self.assertEqual(
            classify_final_answer_kind("the soldier in the red box", ANCHORS),
            "anchor",
        )
        self.assertEqual(classify_final_answer_kind("8", ANCHORS), "numeric")
        self.assertEqual(classify_final_answer_kind("both true", ANCHORS), "boolean")
        self.assertEqual(classify_final_answer_kind("striped umbrella", ANCHORS), "semantic")

    def test_anchor_attribute_leak(self):
        self.assertFalse(
            has_anchor_attribute_leak("Which object is in the red box?", "red", ANCHORS)
        )
        self.assertFalse(
            has_anchor_attribute_leak(
                "Among the red, blue, and green outlined regions, which is unique?",
                "red",
                ANCHORS,
            )
        )
        self.assertTrue(
            has_anchor_attribute_leak(
                "Which marked region contains a distinct red textile object?",
                "red",
                ANCHORS,
            )
        )

    def test_declared_kind_is_rechecked_after_voting(self):
        self.assertTrue(answer_kind_matches_declared("semantic", "semantic"))
        self.assertFalse(answer_kind_matches_declared("semantic", "anchor"))
        self.assertTrue(answer_kind_matches_declared("text", "numeric"))

    def test_schema_aliases_and_answer_semantics_are_normalized(self):
        self.assertEqual(
            normalize_question_type("comparison"), "relational_composition"
        )
        self.assertEqual(
            normalize_question_type("multi-hop"), "multi_hop_reasoning"
        )
        self.assertEqual(
            infer_answer_semantics("cross_region_pairing", "semantic_pair"),
            "ordered_pair",
        )
        self.assertEqual(
            infer_answer_semantics("set_filtering", "semantic_set"),
            "unordered_set",
        )

    def test_question_answer_contract_rejects_location_for_text_request(self):
        passed, reasons = validate_question_answer_contract(
            "Identify the text or number visible on the selected curved surface.",
            "Image 1",
            "text",
            ANCHORS,
        )
        self.assertFalse(passed)
        self.assertIn("text_request_answered_with_location", reasons)

    def test_question_answer_contract_accepts_numeric_text_and_anchor(self):
        self.assertTrue(validate_question_answer_contract(
            "How many people are visible across all marked regions?", "7", "numeric", ANCHORS
        )[0])
        self.assertTrue(validate_question_answer_contract(
            "What text appears in the selected marked region?", "CAFE", "text", ANCHORS
        )[0])
        self.assertTrue(validate_question_answer_contract(
            "Which marked region contains the only hatchback?", "red", "anchor", ANCHORS
        )[0])

    def test_scalar_contract_rejects_two_explicit_output_requests(self):
        passed, reasons = validate_question_answer_contract(
            "Arrange the three marked grilles from left to right, and state "
            "the shape of the badge on the middle vehicle.",
            "oval",
            "semantic",
            ANCHORS,
            answer_semantics="scalar",
        )
        self.assertFalse(passed)
        self.assertIn("multiple_output_requests_for_scalar_answer", reasons)

    def test_scalar_contract_accepts_reasoning_as_a_modifier(self):
        passed, reasons = validate_question_answer_contract(
            "After arranging the three marked grilles from left to right, "
            "what shape is the badge on the middle vehicle?",
            "oval",
            "semantic",
            ANCHORS,
            answer_semantics="scalar",
        )
        self.assertTrue(passed, reasons)

    def test_pair_contract_allows_a_coordinated_pair_request(self):
        passed, reasons = validate_question_answer_contract(
            "Identify the matching person and vehicle.",
            "the driver + the white van",
            "semantic_pair",
            ANCHORS,
            answer_semantics="ordered_pair",
        )
        self.assertTrue(passed, reasons)

    def test_only_scalar_semantic_and_text_answers_are_open_vocabulary(self):
        self.assertTrue(uses_open_answer_space("scalar", "semantic"))
        self.assertTrue(uses_open_answer_space("scalar", "text"))
        self.assertFalse(uses_open_answer_space("unordered_set", "semantic_set"))
        self.assertFalse(uses_open_answer_space("ordered_pair", "semantic_pair"))
        self.assertFalse(uses_open_answer_space("scalar", "anchor"))
        self.assertFalse(uses_open_answer_space("scalar", "numeric"))

    def test_open_answers_allow_any_expression_under_ten_words(self):
        self.assertEqual(
            open_answer_word_count("the woman is holding a cigarette"), 6
        )
        self.assertEqual(open_answer_word_count("红衣女子正在打电话"), 5)
        self.assertLess(open_answer_word_count("two people face the same direction"), 10)
        self.assertEqual(
            open_answer_word_count(
                "the woman is probably waiting because the bus is late"
            ),
            10,
        )

    @patch("mr_vqa.selfcheck.call_chat")
    def test_semantic_judge_returns_equivalent_group_members(self, call_chat):
        call_chat.return_value = (
            '{"groups": [["kneeling soldier", [1, 2, 3, 5, 6, 7]], '
            '["standing soldier", [4, 8]]], "largest_group_size": 6, '
            '"canonical": "kneeling soldier"}'
        )
        size, canonical, members = _llm_judge_largest_group(
            "What is the selected person doing?",
            [
                "kneeling soldier", "soldier kneeling", "kneeling",
                "standing soldier", "person kneeling", "kneeling person",
                "soldier is kneeling", "standing",
            ],
            "http://judge", "key", "judge-model",
        )
        self.assertEqual(size, 6)
        self.assertEqual(canonical, "kneeling soldier")
        self.assertEqual(members, [1, 2, 3, 5, 6, 7])


if __name__ == "__main__":
    unittest.main()
