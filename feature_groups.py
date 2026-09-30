"""Column-selection helpers for myfeatures.csv.

The CSV has a 3-level column MultiIndex: (feature, statistics, number),
e.g. ('mfcc', 'mean', '03'). The scalar features 'tempo' and 'beat_rate'
have a single column ('tempo', 'value', '01').

Everything here works on the MultiIndex directly, so the same helpers can
be reused later for group ablation / permutation importance.
"""
import numpy as np
import pandas as pd

ALL_STATS = ('mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max', '10th', '90th')
SCALAR_FEATURES = ('tempo', 'beat_rate')

# Statistics used for every feature of the compact set (same for all -> fair comparison).
UNIFORM_STATS = ('mean', 'std', 'skew', 'kurtosis', 'median')

# One or two representatives per musical dimension.
COMPACT_FEATURES = (
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


def select_columns(columns: pd.MultiIndex, features=None, stats=None) -> pd.MultiIndex:
    """Return the subset of ``columns`` matching the given features / statistics.

    Args:
        columns: The MultiIndex of the feature table.
        features: Feature names to keep (level 0). ``None`` keeps all.
        stats: Statistics to keep (level 1). ``None`` keeps all. The scalar
            features ('value' statistic) are never removed by this filter.
    """
    feat = np.asarray(columns.get_level_values(0))
    stat = np.asarray(columns.get_level_values(1))
    keep = np.ones(len(columns), dtype=bool)
    if features is not None:
        keep &= np.isin(feat, list(features))
    if stats is not None:
        keep &= np.isin(stat, list(stats)) | (stat == 'value')
    return columns[keep]


def build_feature_sets(columns: pd.MultiIndex) -> dict:
    """The two feature sets compared in the model-selection step."""
    return {
        'compact': select_columns(columns, COMPACT_FEATURES, UNIFORM_STATS),
        'full': columns,
    }
