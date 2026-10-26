"""Requested publication scope, separate from executable adapter availability.

An audited benchmark must not silently become runnable through a generic prompt.
"""

REQUESTED = (
    'chartography', 'mimic', 'mmsi', 'spar', 'mmlongbench', 'milebench',
    'memlens', 'erqa', 'viewspatial', 'mindcube', 'agentvista',
    'screenspot_pro', 'ui_vision', 'mmstar', 'babyvision',
    'vstar', 'hr4k', 'hr8k', 'mme_en', 'mme_cn', 'zoombench',
    'countqa', 'perception', 'treebench', 'o3',
)
EXCLUDED = ()


def coverage():
    from .protocols import PROTOCOLS
    return {name: {
        'adapter_implemented': name in PROTOCOLS,
        'acceptance': 'integration acceptance pending',
    } for name in REQUESTED}
