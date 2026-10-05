"""Compare classifiers on the compact and full feature sets (FMA small, 'genre_top').

Protocol
--------
* The official training and validation sets are pooled into one development
  set. Models and hyperparameters are compared with k-fold cross-validation
  on it, using StratifiedGroupKFold: folds are stratified by genre and keep
  all the tracks of an artist in the same fold (no artist leakage).
* The same folds are used for every model and feature set, so comparisons
  are paired.
* The official test set is untouched unless --test is given. In that case
  the best model of each feature set is refit on the whole development set
  and evaluated once on the test set.
* Imputation and scaling are inside the Pipeline, so they are fitted on the
  training folds only (no leakage).
* Fold models are discarded right after scoring: only scores and out-of-fold
  predictions are kept. Use --save-models to keep the final (refit) models.

Usage
-----
    python compare_models.py                       # both sets, all models
    python compare_models.py --feature-sets compact --models svm rf
    python compare_models.py --save-models --test  # final step, once
"""
import argparse
import os
import time
from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, StandardScaler
from sklearn.svm import SVC

from feature_groups import FEATURE_SET_SPECS, build_feature_sets

DEFAULT_DATA_DIR = os.environ.get('MUSIC_DATA_DIR', '/run/media/thomas/Data/Documents/Database')
MODEL_NAMES = ('logreg', 'knn', 'svm', 'rf', 'hgb', 'mlp')
# Components that use randomness. Each one gets its own seed derived from --seed.
# (logreg/lbfgs, kNN and SVM are deterministic.)
SEED_KEYS = ('folds', 'scaler', 'rf', 'hgb', 'mlp')


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class Data(NamedTuple):
    """Development set (official training + validation, pooled) and held-out test set."""
    X_dev: pd.DataFrame
    y_dev: np.ndarray
    groups_dev: np.ndarray  # artist id of each development track
    X_test: pd.DataFrame
    y_test: np.ndarray
    encoder: LabelEncoder   # encoder.classes_ are the genre names


def load_data(data_dir: str | os.PathLike) -> Data:
    """Load features and 'genre_top' labels and split them with the official FMA split.

    The audio files are not read: the features are already computed.

    Args:
        data_dir: Directory containing ``fma_metadata/``.

    Returns:
        The development and test sets.
    """
    import utils  # imported here because it pulls in librosa

    meta = Path(data_dir) / 'fma_metadata'
    print(f"Reading data from {meta}...")
    tracks = utils.load(meta / 'tracks.csv')
    features = utils.load(meta / 'myfeatures.csv')

    features = features.replace([np.inf, -np.inf], np.nan)
    failed = features.index[features.isna().all(axis=1)]
    if len(failed) > 0:
        print(f"Dropping {len(failed)} tracks whose feature extraction failed.")
    usable = features.index.difference(failed)

    small = tracks['set', 'subset'] <= 'small'
    split = tracks['set', 'split']
    genre = tracks['track', 'genre_top']
    labels = genre.astype(str)
    artist = tracks['artist', 'id']

    def select(splits: tuple[str, ...]) -> pd.Index:
        mask = (small & split.isin(splits) & genre.notna()).to_numpy()
        ids = tracks.index[mask]
        return ids[ids.isin(usable)]

    dev_ids = select(('training', 'validation'))
    test_ids = select(('test',))

    encoder = LabelEncoder().fit(labels.loc[dev_ids])
    data = Data(X_dev=features.loc[dev_ids],
                y_dev=encoder.transform(labels.loc[dev_ids]),
                groups_dev=artist.loc[dev_ids].to_numpy(),
                X_test=features.loc[test_ids],
                y_test=encoder.transform(labels.loc[test_ids]),
                encoder=encoder)

    shared = np.intersect1d(data.groups_dev, artist.loc[test_ids].to_numpy())
    n_nan = int(data.X_dev.isna().sum().sum())
    print(f"{len(data.y_dev)} development tracks ({len(np.unique(data.groups_dev))} artists), "
          f"{len(data.y_test)} test tracks, {len(encoder.classes_)} classes: {list(encoder.classes_)}")
    print(f"{features.shape[1]} feature columns; {n_nan} NaN cells in the development set "
          f"(imputed with the training median inside the pipeline).")
    print(f"Artists present in both development and test sets: {len(shared)} (should be 0).")
    return data


# ---------------------------------------------------------------------------
# Seeds and models
# ---------------------------------------------------------------------------
def derive_seeds(seed: int) -> dict[str, int]:
    """Derive one independent, reproducible seed per random component.

    The seeds depend only on ``seed`` and on ``SEED_KEYS``, not on which models
    are run or in which order, so a given model gets the same seed whatever
    else is run with it.

    Args:
        seed: The master seed.

    Returns:
        A dict mapping each name of ``SEED_KEYS`` to an integer seed.
    """
    states = np.random.SeedSequence(seed).generate_state(len(SEED_KEYS))
    return {key: int(state) for key, state in zip(SEED_KEYS, states)}


def build_models(n_features: int, seeds: dict[str, int],
                 scaler: str) -> dict[str, tuple[Pipeline, dict[str, list]]]:
    """Build the pipelines and their hyperparameter grids.

    Args:
        n_features: Number of input columns (the SVM ``gamma`` grid depends on it).
        seeds: Seeds from :func:`derive_seeds`.
        scaler: 'standard' or 'quantile'; applied before the scale-sensitive models.

    Returns:
        A dict mapping each name of ``MODEL_NAMES`` to ``(pipeline, grid)``.
        The grid keys are Pipeline-style ('clf__C', ...).
    """

    def pipe(clf, scale: bool) -> Pipeline:
        steps = [('impute', SimpleImputer(strategy='median'))]
        if scale:
            steps.append(('scale', QuantileTransformer(output_distribution='normal',
                                                       random_state=seeds['scaler'])
                          if scaler == 'quantile' else StandardScaler()))
        steps.append(('clf', clf))
        return Pipeline(steps)

    # After scaling, features have unit variance, so SVC's gamma='scale' is 1/n_features.
    # The grid is expressed relative to that so it is comparable across feature sets.
    gammas = [m / n_features for m in (0.3, 1.0, 3.0)]

    return {
        'logreg': (pipe(LogisticRegression(max_iter=2000), True),
                   {'clf__C': [0.01, 0.1, 1.0]}),
        'knn': (pipe(KNeighborsClassifier(), True),
                {'clf__n_neighbors': [5, 15, 30], 'clf__weights': ['uniform', 'distance']}),
        'svm': (pipe(SVC(kernel='rbf'), True),
                {'clf__C': [1, 10, 100], 'clf__gamma': gammas}),
        'rf': (pipe(RandomForestClassifier(n_estimators=300, n_jobs=1,
                                           random_state=seeds['rf']), False),
               {'clf__max_features': ['sqrt', 0.1], 'clf__min_samples_leaf': [1, 3]}),
        'hgb': (pipe(HistGradientBoostingClassifier(max_iter=300, early_stopping=True,
                                                    validation_fraction=0.1,
                                                    random_state=seeds['hgb']), False),
                {'clf__learning_rate': [0.05, 0.1], 'clf__max_leaf_nodes': [15, 31]}),
        'mlp': (pipe(MLPClassifier(max_iter=300, early_stopping=True,
                                   random_state=seeds['mlp']), True),
                {'clf__hidden_layer_sizes': [(128,), (256, 128)],
                 'clf__alpha': [1e-4, 1e-2, 1.0]}),
    }


def format_params(params: dict) -> str:
    """Readable one-line description of a hyperparameter dict."""
    items = []
    for key, value in params.items():
        name = key.removeprefix('clf__')
        items.append(f"{name}={value:.3g}" if isinstance(value, float) else f"{name}={value}")
    return ', '.join(items)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
def _fit_predict(pipeline: Pipeline, params: dict, X: np.ndarray, y: np.ndarray,
                 train_idx: np.ndarray, val_idx: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit one configuration on one training fold. Returns (predictions on the
    validation fold, fit time in seconds). The fitted model is discarded."""
    model = clone(pipeline).set_params(**params)
    t0 = time.perf_counter()
    model.fit(X[train_idx], y[train_idx])
    fit_time = time.perf_counter() - t0
    return model.predict(X[val_idx]), fit_time


def cross_validate_grid(pipeline: Pipeline, grid: dict[str, list], X: np.ndarray, y: np.ndarray,
                        folds: list[tuple[np.ndarray, np.ndarray]],
                        n_jobs: int) -> tuple[list[dict], list[dict], np.ndarray]:
    """Cross-validate every configuration of ``grid``.

    All (configuration, fold) fits are independent, so they are distributed
    over worker processes. Only scores and predictions are returned.

    Args:
        pipeline: Unfitted pipeline.
        grid: Hyperparameter grid (Pipeline-style keys).
        X: Feature matrix of the development set.
        y: Integer labels.
        folds: List of (train_idx, val_idx) pairs.
        n_jobs: Number of worker processes (-1 = all cores).

    Returns:
        configs: the hyperparameter dicts, indexed by config_id.
        rows: one dict per fit (config_id, config, fold, acc, f1, fit_time).
        oof: array (n_configs, n_samples) of out-of-fold predictions.
    """
    configs = list(ParameterGrid(grid))
    tasks = [(c, f) for c in range(len(configs)) for f in range(len(folds))]
    outputs = Parallel(n_jobs=n_jobs)(
        delayed(_fit_predict)(pipeline, configs[c], X, y, *folds[f]) for c, f in tasks)

    oof = np.empty((len(configs), len(y)), dtype=np.int64)
    rows = []
    for (c, f), (pred, fit_time) in zip(tasks, outputs):
        val_idx = folds[f][1]
        oof[c, val_idx] = pred
        rows.append({'config_id': c, 'config': format_params(configs[c]), 'fold': f,
                     'acc': accuracy_score(y[val_idx], pred),
                     'f1': f1_score(y[val_idx], pred, average='macro'),
                     'fit_time': fit_time})
    return configs, rows, oof


def summarize(fold_scores: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-fold scores: one row per (feature set, model, configuration)."""
    return (fold_scores
            .groupby(['feature_set', 'n_features', 'model', 'config_id', 'config'], sort=False)
            .agg(acc_mean=('acc', 'mean'), acc_std=('acc', 'std'),
                 f1_mean=('f1', 'mean'), f1_std=('f1', 'std'),
                 fit_time_s=('fit_time', 'mean'))
            .reset_index())


def best_configs(summary: pd.DataFrame) -> pd.DataFrame:
    """Best configuration (highest mean accuracy) of each (feature set, model)."""
    idx = summary.groupby(['feature_set', 'model'], sort=False).acc_mean.idxmax()
    return summary.loc[idx]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def report_winner(y: np.ndarray, oof_pred: np.ndarray, classes: np.ndarray,
                  title: str, path: Path) -> None:
    """Print a per-class report and save a normalized confusion matrix (out-of-fold predictions)."""
    print(f"\n--- {title} (out-of-fold predictions) ---")
    print(classification_report(y, oof_pred, target_names=classes, digits=3))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y, oof_pred, display_labels=classes, normalize='true',
        values_format='.2f', xticks_rotation=45, ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def run_comparison(data: Data, fsets: dict[str, pd.MultiIndex], model_names: list[str],
                   out_dir: Path, seed: int, scaler: str, n_folds: int, n_jobs: int,
                   do_test: bool, save_models: bool) -> None:
    """Cross-validate every model on every feature set and write the results to ``out_dir``.

    Args:
        data: Development and test sets.
        fsets: Feature-set name -> selected columns.
        model_names: Names from ``MODEL_NAMES`` to run.
        out_dir: Output directory (created if needed).
        seed: Master seed (see :func:`derive_seeds`).
        scaler: 'standard' or 'quantile'.
        n_folds: Number of cross-validation folds.
        n_jobs: Number of worker processes (-1 = all cores).
        do_test: Evaluate the best model of each feature set on the test set.
        save_models: Refit the best configuration of every (feature set, model)
            on the whole development set and save it.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = derive_seeds(seed)

    # Folds are built once and reused for every model / feature set (paired comparison).
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seeds['folds'])
    folds = list(cv.split(data.X_dev, data.y_dev, data.groups_dev))

    fold_rows: list[dict] = []
    configs_by_key: dict[tuple[str, str], list[dict]] = {}
    oof_by_key: dict[tuple[str, str], np.ndarray] = {}

    for fs_name, cols in fsets.items():
        X = data.X_dev.loc[:, cols].to_numpy(np.float32)
        models = build_models(len(cols), seeds, scaler)

        for name in model_names:
            pipeline, grid = models[name]
            t0 = time.perf_counter()
            configs, rows, oof = cross_validate_grid(pipeline, grid, X, data.y_dev, folds, n_jobs)
            elapsed = time.perf_counter() - t0

            rows = [{'feature_set': fs_name, 'n_features': len(cols), 'model': name, **r}
                    for r in rows]
            fold_rows.extend(rows)
            configs_by_key[(fs_name, name)] = configs
            oof_by_key[(fs_name, name)] = oof

            best = best_configs(summarize(pd.DataFrame(rows))).iloc[0]
            print(f"[{fs_name:7s} | {len(cols):4d} feats] {name:6s} "
                  f"acc={best.acc_mean:.3f}±{best.acc_std:.3f}  macro-F1={best.f1_mean:.3f}  "
                  f"({elapsed:.0f}s, {len(configs)} configs x {n_folds} folds)  {best.config}")

    # Raw per-fold scores and per-configuration summary: every trained model is in there.
    fold_scores = pd.DataFrame(fold_rows)
    summary = summarize(fold_scores)
    fold_scores.to_csv(out_dir / 'fold_scores.csv', index=False)
    summary.to_csv(out_dir / 'config_scores.csv', index=False)

    best = best_configs(summary)
    spread = summary.groupby(['feature_set', 'model'], sort=False).acc_mean.agg(
        acc_worst='min', n_configs='size')
    table = best.join(spread, on=['feature_set', 'model'])
    print("\n=== Best configuration per model (cross-validated; acc_worst = worst config of that model) ===")
    print(table.sort_values(['feature_set', 'acc_mean'], ascending=[True, False])
          [['feature_set', 'n_features', 'model', 'acc_mean', 'acc_std', 'f1_mean',
            'acc_worst', 'n_configs', 'fit_time_s', 'config']]
          .to_string(index=False, float_format='%.3f'))

    winners = best.loc[best.groupby('feature_set', sort=False).acc_mean.idxmax()]
    for row in winners.itertuples():
        oof_pred = oof_by_key[(row.feature_set, row.model)][row.config_id]
        report_winner(data.y_dev, oof_pred, data.encoder.classes_,
                      f'{row.feature_set} / {row.model}',
                      out_dir / f'confusion_{row.feature_set}_{row.model}.png')

    if not (save_models or do_test):
        return

    # Refit on the whole development set: all best configurations to save them,
    # otherwise only the winner of each feature set (for the test evaluation).
    models_dir = out_dir / 'models'
    if save_models:
        models_dir.mkdir(exist_ok=True)
    for row in (best if save_models else winners).itertuples():
        cols = fsets[row.feature_set]
        pipeline, _ = build_models(len(cols), seeds, scaler)[row.model]
        params = configs_by_key[(row.feature_set, row.model)][row.config_id]
        model = clone(pipeline).set_params(**params).fit(
            data.X_dev.loc[:, cols].to_numpy(np.float32), data.y_dev)

        if save_models:
            path = models_dir / f'{row.feature_set}_{row.model}.joblib'
            dump({'pipeline': model, 'columns': list(cols),
                  'classes': list(data.encoder.classes_)}, path, compress=3)
            print(f"Saved {path}")
        if do_test and row.Index in winners.index:
            pred = model.predict(data.X_test.loc[:, cols].to_numpy(np.float32))
            print(f"TEST  {row.feature_set}/{row.model}: acc={accuracy_score(data.y_test, pred):.3f}  "
                  f"macro-F1={f1_score(data.y_test, pred, average='macro'):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    ap.add_argument('--out-dir', type=Path, default=Path('results'))
    ap.add_argument('--feature-sets', nargs='+', choices=list(FEATURE_SET_SPECS),
                    default=list(FEATURE_SET_SPECS))
    ap.add_argument('--models', nargs='+', choices=MODEL_NAMES, default=list(MODEL_NAMES))
    ap.add_argument('--scaler', choices=['standard', 'quantile'], default='standard',
                    help="scaler for logreg/knn/svm/mlp (quantile is more robust to skew/kurtosis outliers)")
    ap.add_argument('--seed', type=int, default=0, help="master seed, see derive_seeds()")
    ap.add_argument('--n-folds', type=int, default=5)
    ap.add_argument('--n-jobs', type=int, default=-1,
                    help="worker processes over the (configuration, fold) fits; -1 = all cores")
    ap.add_argument('--test', action='store_true',
                    help="evaluate the best model of each feature set on the test set (do this once, at the end)")
    ap.add_argument('--save-models', action='store_true',
                    help="refit the best configuration of each (feature set, model) on the whole development set and save it")
    args = ap.parse_args()

    data = load_data(args.data_dir)
    all_sets = build_feature_sets(data.X_dev.columns)
    fsets = {name: all_sets[name] for name in args.feature_sets}
    print({name: len(cols) for name, cols in fsets.items()}, "columns per feature set\n")

    run_comparison(data, fsets, args.models, args.out_dir, seed=args.seed, scaler=args.scaler,
                   n_folds=args.n_folds, n_jobs=args.n_jobs, do_test=args.test,
                   save_models=args.save_models)


if __name__ == '__main__':
    main()
