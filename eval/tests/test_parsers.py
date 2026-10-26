"""Execute selected preserved pure functions, without importing torch/vLLM."""
import ast
import math
import re
import string
import unittest

from zwz_eval.protocols import RESOURCES


def functions(relative, names, constants=None):
    source = RESOURCES / 'vendor' / relative
    tree = ast.parse(source.read_text())
    picked = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in picked} != set(names):
        raise AssertionError('Preserved function identity changed')
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *picked], type_ignores=[])
    ast.fix_missing_locations(tree)
    ns = {'re': re, 'math': math, 'string': string, **(constants or {})}
    exec(compile(tree, str(source), 'exec'), ns)
    return ns


class PreservedParsers(unittest.TestCase):
    def test_mml_answer_extraction(self):
        m = functions('mmlongbench/utils.py', ['remove_articles', 'white_space_fix', 'remove_punc', 'lower',
            'normalize_answer', 'substring_exact_match_score', 'parse_output', 'normalize_answer_with_punc',
            'extract_binary_label', 'extract_choice_letter'])
        self.assertEqual(m['parse_output']('<answer>A</answer> text <answer>B</answer>'), 'B')
        self.assertEqual(m['parse_output']('Answer: first\nAnswer: final'), 'final')
        self.assertTrue(m['substring_exact_match_score']('It is the Eiffel Tower.', 'Eiffel Tower'))
        self.assertEqual(m['extract_choice_letter']('Answer: B'), 1)
        self.assertEqual(m['extract_binary_label']('yes, reconsider: no'), 0)

    def test_memlens_refusal_and_judge_parse(self):
        m = functions('memlens/judge.py', ['parse_score_from_judge', 'get_task_key'])
        self.assertEqual(m['parse_score_from_judge']('{"answer_score": 1}'), 1)
        self.assertEqual(m['parse_score_from_judge']('unparseable'), 0)
        self.assertEqual(m['get_task_key']('answer_refusal', '', 'anything'), 'AR_AnswerRefusal')

    def test_gui_inclusive_box_and_invalid_denominator(self):
        m = functions('gui/scripts/eval_grounding_topology.py', ['eval_sample_positive_gt',
            'collect_results_to_eval', 'calc_metric_for_result_list'])
        sample = {'bbox': [200, 100, 400, 200], 'img_size': [1000, 500]}
        self.assertEqual(m['eval_sample_positive_gt'](sample, {'point': [0.2, 0.2]}), 'correct')
        self.assertEqual(m['eval_sample_positive_gt'](sample, {'point': None}), 'wrong_format')
        score = m['calc_metric_for_result_list']([{'correctness': 'correct', 'ui_type': 'text'}, {'correctness': 'wrong_format', 'ui_type': 'icon'}])
        self.assertEqual(score['num_total'], 2)
        self.assertEqual(score['action_acc'], 0.5)


    def test_common_last_answer_box(self):
        m = functions('common/infer.py', ['normalize_model_answer'])
        self.assertEqual(m['normalize_model_answer']('<think>analysis</think><answer>A</answer>'), 'A')
        self.assertEqual(m['normalize_model_answer']('<answer>A</answer><answer>B</answer>'), 'B')


if __name__ == '__main__':
    unittest.main()
