"""Column-selection helpers for myfeatures.csv.

The CSV has a 3-level column MultiIndex: (feature, statistics, number),
e.g. ('mfcc', 'mean', '03'). The scalar features 'tempo' and 'beat_rate'
have a single column ('tempo', 'value', '01').

Everything here works on the MultiIndex directly, so the same helpers can
be reused later for group ablation / permutation importance.
"""
from collections.abc import Iterable

import numpy as np
import pandas as pd

# Statistics used for every feature of the compact set (same for all -> fair comparison).
# UNIFORM_STATS: tuple[str, ...] = ('mean', 'std', 'skew', 'kurtosis', 'median')
UNIFORM_STATS: tuple[str, ...] = ('mean', 'std', 'skew', 'kurtosis')

# One or two representatives per musical dimension.
COMPACT_FEATURES: tuple[str, ...] = (
    # timbre
    'mfcc',
    # spectral shape
    'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff_85',
    'spectral_contrast', 'spectral_flatness',
    # harmony
    'chroma_cens',
    # energy / percussive
    'rms', 'zcr', 'percussive_ratio',
    # rhythm
    'onset_strength', 'tempo', 'beat_rate',
)
MORE_COMPACT_FEATURES: tuple[str, ...] = (
    # timbre
    'mfcc',
    # spectral shape
    'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff_85',
    'spectral_contrast', 'spectral_flatness',
    # energy / percussive
    'rms', 'zcr',
    # rhythm
    'tempo',
)

# Feature-set name -> keyword arguments of select_columns (no arguments = all columns).
# This is the single place where the feature-set names are defined.
FEATURE_SET_SPECS: dict[str, dict] = {
    'compact': {'features': MORE_COMPACT_FEATURES, 'stats': UNIFORM_STATS},
    'full': {},
}


def select_columns(columns: pd.MultiIndex,
                   features: Iterable[str] | None = None,
                   stats: Iterable[str] | None = None) -> pd.MultiIndex:
    """Return the subset of ``columns`` matching the given features / statistics.

    Args:
        columns: The MultiIndex of the feature table.
        features: Feature names to keep (level 0). ``None`` keeps all.
        stats: Statistics to keep (level 1). ``None`` keeps all. The scalar
            features (whose statistic is ``'value'``) are never removed by
            this filter.

    Returns:
        The selected columns, in their original order.
    """
    feat = np.asarray(columns.get_level_values(0))
    stat = np.asarray(columns.get_level_values(1))
    keep = np.ones(len(columns), dtype=bool)
    if features is not None:
        keep &= np.isin(feat, list(features))
    if stats is not None:
        keep &= np.isin(stat, list(stats)) | (stat == 'value')
    return columns[keep]


def build_feature_sets(columns: pd.MultiIndex) -> dict[str, pd.MultiIndex]:
    """Build the feature sets compared in the model-selection step.

    Args:
        columns: The MultiIndex of the feature table.

    Returns:
        A dict mapping each name of ``FEATURE_SET_SPECS`` to its columns.
    """
    return {name: select_columns(columns, **spec) for name, spec in FEATURE_SET_SPECS.items()}
