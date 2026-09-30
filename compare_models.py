"""Compare classifiers on the compact and full feature sets (FMA small, 'genre_top').

Protocol
--------
* Official FMA split (training / validation / test, artist-aware).
* Hyperparameters and models are chosen on the VALIDATION set only.
* The test set is untouched unless you pass --test, in which case only the
  single best (by validation) model of each feature set is evaluated on it.
* Imputation and scaling live inside the Pipeline, so they are fitted on the
  training data only (no leakage).

Usage
-----
    python compare_models.py                       # both sets, all models
    python compare_models.py --feature-sets compact --models svm rf
    python compare_models.py --test                # final check, once, at the end
"""
import argparse
import os
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, StandardScaler
from sklearn.svm import SVC

from feature_groups import build_feature_sets

DEFAULT_DATA_DIR = os.environ.get('MUSIC_DATA_DIR', '/run/media/thomas/Data/Documents/Database')
MODEL_NAMES = ('logreg', 'knn', 'svm', 'rf', 'hgb', 'mlp')


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data(data_dir):
    """Load features + labels and split them with the official FMA split.

    Audio files are NOT read: the features are already computed.

    Returns:
        data: dict split -> (features DataFrame, integer labels array)
        encoder: fitted LabelEncoder (encoder.classes_ are the genre names)
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

    ids = {}
    for key, name in (('train', 'training'), ('val', 'validation'), ('test', 'test')):
        mask = (small & (split == name) & genre.notna()).to_numpy()
        sel = tracks.index[mask]
        ids[key] = sel[sel.isin(usable)]

    encoder = LabelEncoder().fit(genre.loc[ids['train']].astype(str))
    data = {k: (features.loc[v], encoder.transform(genre.loc[v].astype(str)))
            for k, v in ids.items()}
    # data is a dictionnary with 3 keys: train, val and test and each of
    # these keys contains 2 arrays, the full feature set (from the csv file)
    # and the encoded labels.
    # so data['train'][0] is the features array
    # and data['train'][1] is the corresponding encoded labels

    n_nan = int(data['train'][0].isna().sum().sum())
    print({k: len(v[1]) for k, v in data.items()}, '| classes:', list(encoder.classes_))
    print(f"{features.shape[1]} feature columns; {n_nan} remaining NaN cells in train "
          f"(imputed with the training median inside the pipeline).")
    return data, encoder


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def build_models(n_features, seed, scaler='standard'):
    """Return {name: (pipeline, param_grid)}. Grid keys are Pipeline-style ('clf__...')."""

    def pipe(clf, scale):
        steps = [('impute', SimpleImputer(strategy='median'))]
        if scale:
            steps.append(('scale', QuantileTransformer(output_distribution='normal',
                                                       random_state=seed)
                          if scaler == 'quantile' else StandardScaler()))
        steps.append(('clf', clf))
        return Pipeline(steps)

    # After scaling, features have unit variance, so SVC's gamma='scale' is 1/n_features.
    # The grid is expressed relative to that so it is comparable across feature sets.
    gammas = [m / n_features for m in (0.3, 1.0, 3.0)]

    return {
        'logreg': (pipe(LogisticRegression(max_iter=2000, random_state=seed), True),
                   {'clf__C': [0.01, 0.1, 1.0]}),
        'knn': (pipe(KNeighborsClassifier(), True),
                {'clf__n_neighbors': [5, 15, 30], 'clf__weights': ['uniform', 'distance']}),
        'svm': (pipe(SVC(kernel='rbf'), True),
                {'clf__C': [1, 10, 100], 'clf__gamma': gammas}),
        'rf': (pipe(RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=seed), False),
               {'clf__max_features': ['sqrt', 0.1], 'clf__min_samples_leaf': [1, 3]}),
        'hgb': (pipe(HistGradientBoostingClassifier(max_iter=300, early_stopping=True,
                                                    validation_fraction=0.1,
                                                    random_state=seed), False),
                {'clf__learning_rate': [0.05, 0.1], 'clf__max_leaf_nodes': [15, 31]}),
        'mlp': (pipe(MLPClassifier(max_iter=300, early_stopping=True, random_state=seed), True),
                {'clf__hidden_layer_sizes': [(128,), (256, 128)],
                 'clf__alpha': [1e-4, 1e-2, 1.0]}),
    }


def _fit_eval(pipeline, params, X_tr, y_tr, X_va, y_va):
    model = clone(pipeline).set_params(**params)
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    fit_time = time.perf_counter() - t0
    pred = model.predict(X_va)
    scores = {'params': params,
              'val_acc': accuracy_score(y_va, pred),
              'val_f1': f1_score(y_va, pred, average='macro'),
              'fit_time': fit_time}
    return model, scores


def tune(pipeline, grid, X_tr, y_tr, X_va, y_va, n_jobs):
    """Grid search: fit on train, score on validation. Returns (best_model, best_scores, n_combos)."""
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_eval)(pipeline, p, X_tr, y_tr, X_va, y_va) for p in ParameterGrid(grid))
    best = max(results, key=lambda r: r[1]['val_acc'])
    return best[0], best[1], len(results)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def run_comparison(data, encoder, fsets, model_names, out_dir, seed=0, scaler='standard',
                   n_jobs=-1, do_test=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (F_tr, y_tr), (F_va, y_va), (F_te, y_te) = data['train'], data['val'], data['test']
    rows, fitted = [], {}

    for fs_name, cols in fsets.items():
        X_tr = F_tr.loc[:, cols].to_numpy(np.float32)
        X_va = F_va.loc[:, cols].to_numpy(np.float32)
        models = build_models(X_tr.shape[1], seed, scaler)

        for name in model_names:
            pipeline, grid = models[name]
            t0 = time.perf_counter()
            model, sc, n_combos = tune(pipeline, grid, X_tr, y_tr, X_va, y_va, n_jobs)
            total = time.perf_counter() - t0
            fitted[(fs_name, name)] = model
            rows.append({'feature_set': fs_name, 'n_features': X_tr.shape[1], 'model': name,
                         'val_acc': sc['val_acc'], 'val_macro_f1': sc['val_f1'],
                         'best_fit_time_s': sc['fit_time'], 'search_time_s': total,
                         'n_configs': n_combos, 'best_params': str(sc['params'])})
            print(f"[{fs_name:7s} | {X_tr.shape[1]:4d} feats] {name:6s} "
                  f"val acc={sc['val_acc']:.3f}  macro-F1={sc['val_f1']:.3f}  "
                  f"({total:.0f}s, {n_combos} configs)  {sc['params']}")

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / 'model_comparison.csv', index=False)

    print("\n=== Validation results (sorted) ===")
    print(results.sort_values(['feature_set', 'val_acc'], ascending=[True, False])
          .drop(columns=['best_params']).to_string(index=False, float_format='%.3f'))

    # Detailed look at the winner of each feature set
    for fs_name in fsets:
        sub = results[results.feature_set == fs_name]
        best = sub.loc[sub.val_acc.idxmax()]
        model = fitted[(fs_name, best.model)]
        cols = fsets[fs_name]

        pred_va = model.predict(F_va.loc[:, cols].to_numpy(np.float32))
        print(f"\n--- {fs_name}: best model = {best.model} (validation) ---")
        print(classification_report(y_va, pred_va, target_names=encoder.classes_, digits=3))
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_va, pred_va, display_labels=encoder.classes_, normalize='true',
            values_format='.2f', xticks_rotation=45, ax=ax, colorbar=False)
        ax.set_title(f'{fs_name} / {best.model} (validation)')
        fig.tight_layout()
        fig.savefig(out_dir / f'confusion_{fs_name}_{best.model}.png', dpi=120)
        plt.close(fig)

        if do_test:
            pred_te = model.predict(F_te.loc[:, cols].to_numpy(np.float32))
            print(f"TEST  {fs_name}/{best.model}: acc={accuracy_score(y_te, pred_te):.3f}  "
                  f"macro-F1={f1_score(y_te, pred_te, average='macro'):.3f}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    ap.add_argument('--out-dir', default='results')
    ap.add_argument('--feature-sets', nargs='+', default=['compact', 'full'],
                    choices=['compact', 'full'])
    ap.add_argument('--models', nargs='+', default=list(MODEL_NAMES), choices=MODEL_NAMES)
    ap.add_argument('--scaler', default='standard', choices=['standard', 'quantile'],
                    help="scaler for logreg/knn/svm/mlp (quantile is more robust to skew/kurtosis outliers)")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-jobs', type=int, default=-1, help="parallel workers over the hyperparameter grid")
    ap.add_argument('--test', action='store_true',
                    help="also evaluate the best-by-validation model of each set on the test set (do this once, at the end)")
    args = ap.parse_args()

    warnings.filterwarnings('ignore', category=UserWarning)
    np.random.seed(args.seed)

    data, encoder = load_data(args.data_dir)
    all_sets = build_feature_sets(data['train'][0].columns)
    fsets = {k: all_sets[k] for k in args.feature_sets}
    print({k: len(v) for k, v in fsets.items()}, "columns per feature set\n")

    run_comparison(data, encoder, fsets, args.models, args.out_dir, seed=args.seed,
                   scaler=args.scaler, n_jobs=args.n_jobs, do_test=args.test)


if __name__ == '__main__':
    main()
