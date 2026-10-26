"""No implicit global thinking/default-token policy across heterogeneous suites."""
from __future__ import annotations

from pathlib import Path

RESOURCES = Path(__file__).resolve().parent / 'resources'
TEMPORAL = 'ActionLocalization ActionPrediction ActionSequence CharacterOrder CounterfactualInference EgocentricNavigation MovingAttribute MovingDirection ObjectExistence ObjectInteraction ObjectShuffle SceneTransition StateChange'.split()
SEMANTIC = 'ALFRED CLEVR-Change DocVQA IEdit MMCoQA MultiModalQA nuscenes OCR-VQA SlideVQA Spot-the-Diff TQA WebQA WikiVQA'.split()
DIAGNOSTIC = 'TextNeedleInAHaystack ImageNeedleInAHaystack GPR1200'.split()
MILE_TASKS = TEMPORAL + SEMANTIC + DIAGNOSTIC
MEM_SPLITS = ['32k', '64k', '128k', '256k']
MML_FAMILIES = ['infoseek', 'viquae', 'mm_niah_retrieval-image', 'mm_niah_reasoning-image', 'vh_multi']
LENGTHS = [8, 16, 32, 64, 128]
GUI_EXPECTED = {'screenspot_pro': {'all': 1581}, 'ui_vision': {'basic': 1772, 'functional': 1772, 'spatial': 1935}}
PLATFORM_COUNTS = {
    'mmstar': 1500, 'babyvision': 388,
    'vstar': 191, 'hr4k': 800, 'hr8k': 800,
    'mme_en': 23609, 'mme_cn': 5917, 'zoombench': 845,
    'perception': 3000, 'treebench': 405, 'o3': 345, 'countqa': 1528,
}
PLATFORM_BENCHMARKS = tuple(PLATFORM_COUNTS)
PLATFORM_SOURCE_FILES = {
    'mmstar': 'mmstar-shuf-imgplaceholder.jsonl',
    'babyvision': 'babyvision-shuf-imgplaceholder.jsonl',
    'vstar': 'vstar-shuf-imgplaceholder.jsonl',
    'hr4k': 'hrbench-4k-shuf-imgplaceholder.jsonl',
    'hr8k': 'hrbench-8k-shuf-imgplaceholder.jsonl',
    'mme_en': 'mme-realworld-shuf-NasPic-imgplaceholder.jsonl',
    'mme_cn': 'mme-realworld-cn-shuf-NasPic-imgplaceholder.jsonl',
    'zoombench': 'pbench-shuf-NasPic-imgplaceholder.jsonl',
    'perception': 'perceptionbench.jsonl',
    'treebench': 'treebench-shuf-imgplaceholder.jsonl',
    'o3': 'insight-o3.jsonl',
    'countqa': 'countqa-shuf-imgplaceholder.jsonl',
}

PROTOCOLS = {
    'mimic': {'version': 'project-four-task-declared-math-policy-v1', 'thinking': False, 'max_tokens': 8192, 'context': 32768, 'max_images': 256, 'judge_thinking': False, 'judge_context': 16384, 'judge_max_images': 0},
    'mmstar': {'version': 'project-evaluatev3-native16384-local96-v1', 'thinking': False, 'max_tokens': 16384, 'context': 32768, 'max_images': 10, 'judge_thinking': None, 'judge_context': 32768, 'judge_max_images': 0, 'judge_transport': 'native-raw-completion-empty-think'},
    'babyvision': {'version': 'project-evaluatev3-native16384-local96-v1', 'thinking': False, 'max_tokens': 16384, 'context': 32768, 'max_images': 10, 'judge_thinking': None, 'judge_context': 32768, 'judge_max_images': 0, 'judge_transport': 'native-raw-completion-empty-think'},
    'chartography': {'version': 'project-bare-prompt-a3b-guard-v1', 'thinking': False, 'max_tokens': 16384, 'context': 32768, 'max_images': 4, 'judge_thinking': False, 'judge_max_images': 0},
    'spar': {'version': 'project-answer2048-taskmacro-v2', 'thinking': False, 'max_tokens': 2048, 'context': 131072, 'max_images': 3, 'judge_thinking': False, 'judge_context': 32768, 'judge_max_images': 0},
    'erqa': {'version': 'project-native4096-bidirectional-a3b-v1', 'thinking': False, 'max_tokens': 4096, 'context': 131072, 'max_images': 16, 'judge_thinking': False, 'judge_context': 32768, 'judge_max_images': 0},
    'mmsi': {'version': 'project-final-answer-20261019-v1', 'thinking': False, 'max_tokens': 4096, 'context': 131072, 'max_images': 10, 'judge_thinking': None},
    'viewspatial': {'version': 'project-resize420x280-wrong-review-v1', 'thinking': False, 'max_tokens': 1024, 'context': 262144, 'max_images': 512, 'judge_thinking': False, 'judge_context': 32768, 'judge_max_images': 0},
    'mindcube': {'version': 'project-rawqa-pattern-priority-v1', 'thinking': False, 'max_tokens': 1536, 'context': 32768, 'max_images': 4, 'judge_thinking': None},
    'mmlongbench': {'version': 'project-25-vh-indexed-v2', 'thinking': False, 'max_tokens': 2048, 'context': 262144, 'max_images': 800, 'judge_thinking': None},
    'milebench': {'version': 'project-test29-v1', 'thinking': False, 'max_tokens': 8192, 'context': 262144, 'max_images': 256, 'judge_thinking': False},
    'memlens': {'version': 'project-official-rubric-v1', 'thinking': False, 'max_tokens': 8192, 'context': 262144, 'max_images': 256, 'judge_thinking': False},
    'agentvista': {'version': 'project-search-only-final209-v2', 'thinking': True, 'max_tokens': 1024, 'context': 131072, 'max_images': 100, 'judge_thinking': None},
    'screenspot_pro': {'version': 'project-point-in-box-v1', 'thinking': False, 'max_tokens': 8192, 'context': 32768, 'max_images': 1, 'max_pixels': 8294400, 'judge_thinking': None},
    'ui_vision': {'version': 'project-element-macro-v1', 'thinking': False, 'max_tokens': 8192, 'context': 32768, 'max_images': 1, 'max_pixels': 8294400, 'judge_thinking': None},
}

# Same evaluate_v3 source path as MMStar/BabyVision. Separate dictionary objects
# keep per-benchmark contracts independent; no upstream default is inferred.
for _bench in PLATFORM_BENCHMARKS:
    if _bench not in PROTOCOLS:
        PROTOCOLS[_bench] = {
            'version': 'project-evaluatev3-normalized', 'thinking': False,
            'max_tokens': 16384, 'context': 32768, 'max_images': 10,
            'judge_thinking': None, 'judge_context': 32768, 'judge_max_images': 0,
            'judge_transport': 'native-raw-completion-empty-think',
        }


def mml_specs():
    items = []
    for family in MML_FAMILIES:
        for k in LENGTHS:
            if family in ('infoseek', 'viquae'):
                depth = 3 if family == 'infoseek' else 6
                filename = f'vrag/{family}_K{k}_dep{depth}.jsonl'
                metric, cap = 'sub_em', 100
            elif family == 'vh_multi':
                filename, metric, cap = f'NIAH/vh_multi_test_1000_K{k}_dep3.jsonl', 'acc', 100
            else:
                task = family.removeprefix('mm_niah_')
                filename, metric, cap = f'NIAH/{task}_test_K{k}_dep6.jsonl', 'mc_acc', 50
            items.append({'family': family, 'k': k, 'file': filename, 'metric': metric, 'max_test_samples': cap})
    return items
