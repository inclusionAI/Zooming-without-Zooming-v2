import unittest
from pathlib import Path

from zwz_eval.references import load
from zwz_eval.scope import REQUESTED, EXCLUDED, coverage


class AuditedExpansion(unittest.TestCase):
    def test_final_script_selection_covers_requested_benches_without_default_substitution(self):
        root = Path(__file__).resolve().parents[1]
        text = (root / 'docs/PROTOCOLS.md').read_text().split('# Final project-adapted script selection')[-1]
        rows = [line for line in text.splitlines() if line.startswith('| ')][2:]
        self.assertEqual(len(rows), len(REQUESTED))
        self.assertEqual(len({line.split('|')[1].strip() for line in rows}), len(REQUESTED))
        self.assertIn('**5,000 Counting**', text)
        self.assertIn('**robust-output disabled**', text)
        self.assertIn('not commands that depend on', text)
        guide = (root / 'README.md').read_text()
        benchmark_table = guide.split('## Supported benchmarks')[1].split('## Benchmark data format')[0]
        public_benches = {line.split('`')[1] for line in benchmark_table.splitlines()
                          if line.startswith('| `')}
        self.assertEqual(public_benches, set(REQUESTED))
        self.assertIn('scripts/evaluate.sh', guide)

    def test_scope_is_exact_and_does_not_overclaim(self):
        self.assertEqual(len(REQUESTED), 25)
        self.assertFalse(set(REQUESTED) & set(EXCLUDED))
        self.assertTrue(coverage()['mmsi']['adapter_implemented'])
        self.assertTrue(all(item['adapter_implemented'] for item in coverage().values()))

    def test_mmsi_final_not_intermediate(self):
        parser = load('mmsi_final').extract_choice
        self.assertEqual(parser('`A` then <answer>B</answer>')[0], 'B')
        self.assertEqual(parser('`A` revised to `D`')[0], 'D')
        self.assertEqual(parser('A. ' + 'reasoning ' * 30)[0], None)
        self.assertEqual(parser('**C**.')[0], 'C')

    def test_viewspatial_and_erqa(self):
        self.assertEqual(load('viewspatial').extract_answer('A\n<answer>D</answer>'), 'D')
        self.assertEqual(load('viewspatial').extract_answer('explanation\nB\n'), 'B')
        self.assertEqual(load('erqa').extract_choice('A </think> B'), 'B')
        self.assertIsNone(load('viewspatial').extract_answer(''))

    def test_platform_is_first_tag_not_last(self):
        m = load('platform')
        self.assertEqual(m.parse_answer_tag('<answer>A</answer><answer>B</answer>'), 'A')
        self.assertEqual(m.parse_first_letter('(C) option'), 'C')
        self.assertEqual(m.parse_answer_tag('Answer: B'), 'B')

    def test_mindcube_preserves_pattern_priority(self):
        m = load('mindcube')
        self.assertEqual(m.extract_answer('A. some analysis\n\\boxed{C}'), 'C')
        self.assertEqual(m.extract_answer('A. explanation\n<answer>B</answer>'), 'A')
        self.assertEqual(m.get_setting_from_id('case_translation_1'), 'translation')

    def test_spar_fractional_and_tag_rules(self):
        m = load('spar')
        self.assertEqual(m.official_score({'task': 'depth_prediction_oc', 'answer': '10'}, '10'), 1)
        self.assertEqual(m.official_score({'task': 'obj_spatial_relation_oo', 'answer': 'B'}, '<answer>B</answer>'), 1)
        self.assertEqual(m.extract_final_answer('<answer>A</answer><answer>D</answer>', 'obj_spatial_relation_oo'), 'A')
        self.assertGreater(m.mra(7, 10), 0)
        self.assertLess(m.mra(7, 10), 1)

    def test_mimic_listing_and_judge(self):
        m = load('mimic_listing')
        self.assertEqual(m.parse_model_answer_to_objects('<answer>cat, dog</answer><answer>cat</answer>'), ['cat', 'dog'])
        self.assertEqual(m.gt_objects_from_response('cat, dog'), ['cat', 'dog'])
        j = load('mimic_judge')
        self.assertEqual(j.parse_llm_judge_response('No, reconsider. Yes'), 'Yes')
        self.assertEqual(j.extract_answer('<answer>A</answer><answer>B</answer>'), 'A')

    def test_chart_guard_and_rubric(self):
        m = load('chartography')
        prompt = m._judge_prompt('question', 'gold', 'raw response')
        self.assertIn('Golden answer: gold', prompt)
        verdict = m._parse_verdict('{"scores":[{"score":1,"rationale":"does not match"}]}')
        self.assertFalse(load('chartography_guard').valid_verdict(verdict, 'gold'))
        self.assertIsNone(m._parse_verdict('API unavailable'))


if __name__ == '__main__':
    unittest.main()
